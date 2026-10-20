"""작은 합성 데이터로 SFT·기본 모델 비교·별도 저장·새 프로세스 복원을 실행합니다."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["train", "check"])
parser.add_argument("output", nargs="?", type=Path, default=Path("../smollm2_sft_v1"))
args = parser.parse_args()
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
train_rows = [("Device gpu status? Reply with one word.", "failed"),
              ("Device network status? Reply with one word.", "healthy")]
probe_rows = [("What is the gpu status? Reply with one word.", "failed"),
              ("What is the network status? Reply with one word.", "healthy")]
torch.set_num_threads(1)
torch.manual_seed(0)
if args.mode == "train":
    args.output.mkdir(parents=True, exist_ok=False)
    location, options = model_id, {"revision": revision}
else:
    location, options = args.output, {"local_files_only": True}
tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=False, **options)
model = AutoModelForCausalLM.from_pretrained(
    location, trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
    attn_implementation="eager", **options,
).to("cpu")


def example(prompt, answer=None):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if answer is None:
        return torch.tensor([prefix])
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True)
    assert full[:len(prefix)] == prefix
    end = full.index(tokenizer.eos_token_id, len(prefix)) + 1
    ids = torch.tensor([full[:end]])
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    assert torch.all(labels[:, :len(prefix)] == -100)
    assert torch.equal(labels[:, len(prefix):], ids[:, len(prefix):])
    assert (labels != -100).any()
    return ids, labels


@torch.inference_mode()
def report(name):
    model.eval()
    records, scores = [], []
    for split, rows in (("train", train_rows), ("probe", probe_rows)):
        for prompt, answer in rows:
            ids, labels = example(prompt, answer)
            loss = model(ids, labels=labels, use_cache=False).loss.item()
            inputs = example(prompt)
            logits = model(inputs).logits[:, -1, :]
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                       max_new_tokens=8, do_sample=False, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
            assert torch.isfinite(logits).all()
            print(f"[{name}/{split}] {answer} | loss={loss:.4f} | 생성={text!r}")
            records.append((split, answer, text))
            scores.append(logits)
    return records, torch.cat(scores)


if args.mode == "train":
    report("base")
    before = model.get_input_embeddings().weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    # ponytail: CPU·전체 파라미터·두 예제·20회 갱신; 확장 시 GPU/LoRA를 별도 비교합니다.
    for step in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model(ids, labels=labels, use_cache=False).loss
                   for ids, labels in (example(*row) for row in train_rows)) / len(train_rows)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert not torch.equal(before, model.get_input_embeddings().weight)
    del before
    records, logits = report("sft")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    torch.save({"records": records, "logits": logits}, args.output / "reference.pt")
    print("추가 학습·별도 저장 완료: smollm2_sft_v1 | 원본 캐시 저장 없음")
else:
    saved = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    records, logits = report("restored")
    assert records == saved["records"], "생성 결과 불일치"
    assert torch.allclose(logits, saved["logits"], atol=1e-5, rtol=1e-5), "logits 불일치"
    print("복원 검사 통과: logits·생성 결과 일치·재학습 없음")
