"""NF4 원본 + q/v LoRA: 같은 두 예제 학습·어댑터만 저장·새 프로세스 복원."""
import argparse
import hashlib
import json
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["train", "check"])
parser.add_argument("output", nargs="?", type=Path, default=Path("../smollm2_qlora_v1"))
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args = parser.parse_args()
# Refuse an existing candidate before loading/downloading anything.
if args.mode == "train" and args.output.exists():
    raise FileExistsError(f"기존 출력 보존: {args.output}")
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
if not bnb.supported_torch_devices or args.device not in bnb.supported_torch_devices:
    raise RuntimeError(f"bitsandbytes backend unavailable: {args.device}")
device = torch.device(args.device)
torch.set_num_threads(1)
torch.manual_seed(0)
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
train_rows = [("Device gpu status? Reply with one word.", "failed"),
              ("Device network status? Reply with one word.", "healthy")]
probe_rows = [("What is the gpu status? Reply with one word.", "failed"),
              ("What is the network status? Reply with one word.", "healthy")]
settings = {
    "base_model": model_id, "revision": revision, "dtype": "float32", "attention": "eager",
    "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                     "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float32"},
    "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
             "target_modules": ["q_proj", "v_proj"], "bias": "none"},
    "training": {"steps": 20, "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0,
                 "clip_norm": 1.0, "seed": 0, "gradient_checkpointing": False},
    "generation": {"max_new_tokens": 8, "do_sample": False, "num_beams": 1,
                   "use_cache": True, "repetition_penalty": 1.0},
}
if args.mode == "check":
    # Validate provenance and fixed settings BEFORE loading the cached base.
    if json.loads((args.output / "qlora_settings.json").read_text(encoding="utf-8")) != settings:
        raise ValueError("QLoRA settings mismatch")
    adapter_config = json.loads((args.output / "adapter_config.json").read_text(encoding="utf-8"))
    expected = dict(settings["lora"], base_model_name_or_path=model_id, revision=revision,
                    peft_type="LORA", task_type="CAUSAL_LM")
    adapter_config["target_modules"] = sorted(adapter_config.get("target_modules", []))
    if any(adapter_config.get(key) != value for key, value in expected.items()):
        raise ValueError("adapter config mismatch")
else:
    # New directory only; a failed new run may leave a partial directory to inspect.
    args.output.mkdir(parents=True, exist_ok=False)

tokenizer = AutoTokenizer.from_pretrained(
    model_id if args.mode == "train" else args.output,
    revision=revision if args.mode == "train" else None,
    local_files_only=(args.mode == "check"), trust_remote_code=False)
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, use_safetensors=True,
    dtype=torch.float32, attn_implementation="eager", quantization_config=quantization,
    device_map={"": args.device}, local_files_only=(args.mode == "check"))
# Use identical preparation on both paths: non-4bit parameters stay float32.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False).eval()
quantized = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
assert quantized, "실제 Linear4bit 없음"
assert all(module.weight.quant_state is not None
           and module.weight.quant_state.quant_type == "nf4"
           and module.weight.quant_state.nested for module in quantized), "NF4/double quant 상태 없음"
assert all(p.device.type == args.device for p in model.parameters()), "자동 offload 금지"
assert all(p.dtype == torch.float32 for p in model.parameters() if not isinstance(p, bnb.nn.Params4bit))
print(f"실행 장치: {model.device} | compute: float32 | NF4 double quant Linear4bit: {len(quantized)}")
generation = GenerationConfig(**settings["generation"], bos_token_id=tokenizer.bos_token_id,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)


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
    assert torch.equal(labels[:, len(prefix):], ids[:, len(prefix):]) and (labels != -100).any()
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
            logits = model(inputs, use_cache=False).logits[:, -1, :]
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
            text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
            assert torch.isfinite(logits).all()
            print(f"[{name}/{split}] {answer} | loss={loss:.4f} | 생성={text!r}")
            records.append((split, answer, text))
            scores.append(logits.cpu())
    return records, torch.cat(scores)


def frozen_digest():
    digest = hashlib.sha256()

    def add(name, value):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            # Bytes, not float conversion: packed uint8 weights must remain exact.
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            add(name, parameter)
    for name, buffer in model.named_buffers():
        add(name, buffer)
    for index, module in enumerate(quantized):
        for key, value in sorted(module.weight.quant_state.as_dict().items()):
            add(f"quant_state.{index}.{key}", value)
    return digest.hexdigest()


if args.mode == "train":
    _, base_logits = report("quantized-base")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, **settings["lora"],
                        init_lora_weights=True, revision=revision)
    # Never call .to() on the quantized whole model; device_map already placed it.
    model = get_peft_model(model, config).eval()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name and (".q_proj." in name or ".v_proj." in name)
                             for name in trainable)
    count = sum(p.numel() for p in trainable.values())
    assert count == 230400
    print(f"QLoRA 학습 대상: {count:,} | q/v rank4 alpha8 dropout0")
    with torch.inference_mode():
        initial = torch.cat([model(example(prompt), use_cache=False).logits[:, -1, :].cpu()
                             for prompt, _ in train_rows + probe_rows])
    assert torch.allclose(initial, base_logits, atol=1e-5, rtol=1e-5), "초기 어댑터가 양자화 원본 출력을 바꿈"
    frozen_before = frozen_digest()
    adapter_before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=0.01, momentum=0.0, weight_decay=0.0)
    # ponytail: 두 예제·20회·q/v만 비교; all-linear 재현·튜닝·성능 벤치마크는 별도입니다.
    for _ in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model(ids, labels=labels, use_cache=False).loss
                   for ids, labels in (example(*row) for row in train_rows)) / len(train_rows)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert any(not torch.equal(adapter_before[name], p) for name, p in trainable.items())
    assert frozen_digest() == frozen_before, "원본/양자화 상태 변경"
    records, logits = report("qlora")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    torch.save({"records": records, "logits": logits}, args.output / "reference.pt")
    (args.output / "qlora_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    size = (args.output / "adapter_model.safetensors").stat().st_size
    print(f"QLoRA 별도 저장: {args.output.name} | 어댑터 가중치: {size} bytes | 원본/양자화 상태 불변")
else:
    model = PeftModel.from_pretrained(model, args.output, is_trainable=False, local_files_only=True)
    assert not any(p.requires_grad for p in model.parameters())
    saved = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    records, logits = report("restored")
    assert records == saved["records"], "생성 결과 불일치"
    assert torch.allclose(logits, saved["logits"], atol=1e-4, rtol=1e-4), "logits 불일치"
    print("QLoRA 복원 검사 통과: NF4 원본+어댑터·logits/생성 일치·재학습 없음")
