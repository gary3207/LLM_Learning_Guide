"""같은 합성 과제를 LoRA로 학습하고 어댑터만 저장·복원합니다. CPU 또는 CUDA 명시 선택."""
import argparse
import hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["train", "check"])
parser.add_argument("output", nargs="?", type=Path, default=Path("../smollm2_lora_v1"))
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA 사용 불가: GPU/PyTorch 상태를 확인하세요. CPU로 자동 전환하지 않습니다.")
device = torch.device(args.device)
torch.set_num_threads(1)
torch.manual_seed(0)
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
train_rows = [("Device gpu status? Reply with one word.", "failed"),
              ("Device network status? Reply with one word.", "healthy")]
probe_rows = [("What is the gpu status? Reply with one word.", "failed"),
              ("What is the network status? Reply with one word.", "healthy")]
if args.mode == "train":
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.output, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, use_safetensors=True,
    dtype=torch.float32, attn_implementation="eager", local_files_only=(args.mode == "check"),
).to(device)
print(f"실행 장치: {model.device} | dtype: {model.dtype}")


def example(prompt, answer=None):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if answer is None:
        return torch.tensor([prefix], device=device)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True)
    assert full[:len(prefix)] == prefix
    end = full.index(tokenizer.eos_token_id, len(prefix)) + 1
    ids = torch.tensor([full[:end]], device=device)
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
            scores.append(logits.cpu())
    return records, torch.cat(scores)


def frozen_digest():
    digest = hashlib.sha256()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if args.mode == "train":
    _, base_logits = report("base")
    full_count = model.num_parameters()
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                        target_modules=["q_proj", "v_proj"], bias="none", init_lora_weights=True,
                        revision=revision)
    model = get_peft_model(model, config).to(device).eval()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    count = sum(p.numel() for p in trainable.values())
    print(f"전체 SFT 학습 대상: {full_count:,} | LoRA 학습 대상: {count:,} | 비율: {count/full_count:.4%}")
    with torch.inference_mode():
        initial = torch.cat([model(example(prompt)).logits[:, -1, :].cpu()
                             for prompt, _ in train_rows + probe_rows])
    assert torch.allclose(initial, base_logits, atol=1e-5, rtol=1e-5), "초기 LoRA가 원본 출력을 바꿈"
    frozen_before = frozen_digest()
    adapter_before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=0.01, momentum=0.0, weight_decay=0.0)
    # ponytail: 두 예제·20회·q/v rank4만 비교; 최적 설정 탐색·속도 벤치마크는 별도입니다.
    for _ in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model(ids, labels=labels, use_cache=False).loss
                   for ids, labels in (example(*row) for row in train_rows)) / len(train_rows)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert any(not torch.equal(adapter_before[name], p) for name, p in trainable.items())
    assert frozen_digest() == frozen_before, "원본 동결 가중치 변경"
    records, logits = report("lora")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    torch.save({"records": records, "logits": logits}, args.output / "reference.pt")
    size = (args.output / "adapter_model.safetensors").stat().st_size
    print(f"LoRA 별도 저장: {args.output.name} | 어댑터 가중치: {size} bytes | 원본 동결 가중치 불변")
else:
    model = PeftModel.from_pretrained(model, args.output, is_trainable=False, local_files_only=True).to(device)
    assert not any(p.requires_grad for p in model.parameters())
    saved = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    records, logits = report("restored")
    assert records == saved["records"], "생성 결과 불일치"
    assert torch.allclose(logits, saved["logits"], atol=1e-4, rtol=1e-4), "logits 불일치"
    print("LoRA 복원 검사 통과: 원본+어댑터·logits/생성 일치·재학습 없음")
