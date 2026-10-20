"""기존 원본·SFT·LoRA를 재학습 없이 같은 조건으로 평가하고 JSON에 저장합니다."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ROWS = [
    ("train", "Device gpu status? Reply with one word.", "failed"),
    ("train", "Device network status? Reply with one word.", "healthy"),
    ("probe", "What is the gpu status? Reply with one word.", "failed"),
    ("probe", "What is the network status? Reply with one word.", "healthy"),
    ("diagnostic", "GPU status? Reply with one word.", "failed"),
    ("diagnostic", "Network status? Reply with one word.", "healthy"),
    ("diagnostic", "Report the gpu status in one word.", "failed"),
    ("diagnostic", "Report the network status in one word.", "healthy"),
]


def classify(text, answer):
    text = text.strip()
    if text == answer:
        return "correct"
    if text in ("failed", "healthy"):
        return "wrong_label"
    if re.search(r"\b" + re.escape(answer) + r"\b", text):
        return "extra_text"
    return "other"


def summarize(records):
    groups = {}
    for group in dict.fromkeys(row["group"] for row in records):
        rows = [row for row in records if row["group"] == group]
        correct = sum(row["correct"] for row in rows)
        groups[group] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                         "mean_loss": sum(row["loss"] for row in rows) / len(rows)}
    return groups


def example(tokenizer, prompt, answer, device):
    import torch
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True)
    assert full[:len(prefix)] == prefix, "대화 형식 접두 불일치"
    end = full.index(tokenizer.eos_token_id, len(prefix)) + 1
    ids = torch.tensor([full[:end]], device=device)
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    assert (labels[:, 1:] != -100).any(), "응답/eos 정답 없음"
    return ids, labels, torch.tensor([prefix], device=device)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftConfig, PeftModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("../smollm2_sft_v1"))
    parser.add_argument("--lora", type=Path, default=Path("../smollm2_lora_v1"))
    parser.add_argument("--output", type=Path, default=Path("../smollm2_evaluation_v1.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"평가 결과 덮어쓰기 거부: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    for path in (args.sft, args.lora):
        if not path.is_dir():
            raise FileNotFoundError(f"기존 저장 폴더 없음: {path}")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True,
                                             trust_remote_code=False)
    for path in (args.sft, args.lora):
        saved = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        assert (saved.get_vocab(), saved.eos_token_id, saved.chat_template) == (
            tokenizer.get_vocab(), tokenizer.eos_token_id, tokenizer.chat_template), "토크나이저 불일치"
    adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
    assert adapter.base_model_name_or_path == MODEL_ID and adapter.revision == REVISION, "LoRA 원본/revision 불일치"
    generation = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                  bos_token_id=tokenizer.bos_token_id, use_cache=True)
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "device": args.device, "dtype": "float32",
                "torch": version("torch"), "transformers": version("transformers"), "peft": version("peft"),
                "generation": generation.to_dict(), "training_executed": False,
                "scoring": "strip-only, case-sensitive exact match; per-example mean response/eos CE",
                "sources": {"sft": str(args.sft), "lora": str(args.lora)}}
    records, summaries = [], {}
    print(f"장치: {args.device} | float32 | 재학습 없음 | 모델당 {len(ROWS)}문항")
    # ponytail: 모델을 순서대로 한 개씩 적재; 대규모 평가는 batching/전용 평가 도구로 확장합니다.
    for name in ("base", "sft", "lora"):
        location = args.sft if name == "sft" else MODEL_ID
        options = {} if name == "sft" else {"revision": REVISION}
        model = AutoModelForCausalLM.from_pretrained(
            location, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager", **options).to(args.device)
        if name == "lora":
            model = PeftModel.from_pretrained(model, args.lora, is_trainable=False, local_files_only=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for index, (group, prompt, answer) in enumerate(ROWS, 1):
                ids, labels, inputs = example(tokenizer, prompt, answer, args.device)
                loss = model(ids, labels=labels, use_cache=False).loss.item()
                assert math.isfinite(loss), "유한하지 않은 loss"
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
                text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
                error = classify(text, answer)
                records.append({"model": name, "id": index, "group": group, "prompt": prompt,
                                "answer": answer, "text": text, "correct": error == "correct",
                                "error": error, "loss": loss})
                print(f"[{name}/{group}/{index}] {error} | 정답={answer} | 생성={text!r} | loss={loss:.4f}")
        summaries[name] = summarize([row for row in records if row["model"] == name])
        for group, result in summaries[name].items():
            print(f"[{name}/{group}] 정답 {result['correct']}/{result['total']} | 평균 loss={result['mean_loss']:.4f}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "records": records, "summary": summaries}, stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"평가 저장: {args.output} | 체크포인트 저장·수정 없음")


if __name__ == "__main__":
    main()
