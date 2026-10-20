# 20 · 4비트 원본으로 QLoRA 학습·저장·복원

[← 이전: 모델 비교 평가](19-model-evaluation.md) · [목차](README.md) · [다음: API 서빙 →](21-api-serving.md)

## 1. 이번에 완성할 결과물

18장에서는 float32 원본을 동결하고 q/v LoRA만 학습했습니다. 이번에는 **같은 원본을 4비트 NF4로 로딩 → 원본을 동결한 채 새 LoRA 학습 → 어댑터만 별도 저장 → 새 프로세스에서 동일한 4비트 원본과 결합해 복원**합니다. 두 학습 질문과 두 probe의 loss·생성도 전후로 관찰합니다.

기존 SFT나 LoRA 후보에 이어서 학습하지 않습니다. 같은 고정 revision의 원본에서 새 어댑터를 시작하고 `smollm2_qlora_v1/`에만 저장합니다. 19장 평가 JSON과 기존 후보는 그대로 둡니다. API 서빙은 이 묶음 뒤의 별도 주제입니다.

| 구분 | 18장 LoRA | 이번 교육용 QLoRA |
|---|---|---|
| Transformer 선형층 원본 | float32 | 4비트 NF4 + 양자화 메타데이터 |
| 갱신 대상 | q/v rank4 어댑터 | 같은 q/v rank4 어댑터 |
| 연산 정밀도 | float32 | float32로 명시 |
| optimizer | SGD · lr0.01 · 20회 · clip1 | 동일 |
| 저장 | 어댑터·토크나이저·복원 기준 | 왼쪽 + 양자화/원본 설정 |

**원본 QLoRA 논문의 전체 실험을 재현하는 코드는 아닙니다.** 앞 장과 연결하기 위해 q/v만 대상으로 하며, 모든 선형층에 LoRA를 붙이거나 paged optimizer를 쓰지 않습니다. 이 작은 모델에서 메모리 절감이 필요해서라기보다 구조를 배우기 위한 실습입니다. 하이퍼파라미터 탐색·속도/메모리 벤치마크·독립 품질 평가도 하지 않습니다.

## 2. 4비트 저장과 학습 계산은 다르다

### 원본을 압축해 보관한다

4비트에는 16가지 코드가 들어갑니다. NF4는 실수 가중치를 16개 표현 수준과 블록별 스케일로 근사합니다. 단순히 모든 가중치를 정수로 반올림하는 예제가 아닙니다. `bnb_4bit_use_double_quant=True`는 양자화에 필요한 스케일 정보도 다시 양자화하는 nested/double quantization을 켭니다.

실제 연산에서는 압축한 가중치를 필요한 정밀도로 해석합니다. **4비트로 저장한다고 loss·gradient·모든 텐서가 4비트가 되는 것은 아닙니다.** 이번 `bnb_4bit_compute_dtype=torch.float32`는 양자화 선형층의 연산 dtype 설정입니다. bf16 자동 혼합정밀도나 GradScaler는 사용하지 않아 이전 실습과 정밀도 변경을 분리합니다.

임베딩·출력 head·정규화·어댑터·활성값·gradient·양자화 메타데이터까지 모두 4비트가 되는 것도 아닙니다. 따라서 가중치의 비트 수만 나누어 전체 모델 파일이나 GPU 메모리 절감률이라고 말할 수 없습니다.

### 동결돼도 gradient는 통과한다

```text
질문 토큰 → 동결된 4비트 원본의 계산 + 학습 가능한 LoRA 변경분
          → 응답/eos loss → 역전파 → LoRA만 SGD 갱신
```

원본에 gradient를 저장하거나 optimizer로 갱신하지 않지만, 원본을 거친 계산에서 LoRA까지 역전파할 수 있습니다. `prepare_model_for_kbit_training()`은 원본 동결과 학습 준비를 맡고, 그 뒤 `get_peft_model()`로 새 어댑터를 붙입니다. 이 실습에서는 gradient checkpointing을 끕니다. 짧은 두 예제에 별도 메모리 절약 기법을 겹치지 않습니다.

양자화로 원본의 값이 근사되므로 **18장의 float32 원본과 출력이 같아야 한다는 검사는 하지 않습니다.** 대신 LoRA 부착 직전의 준비된 4비트 원본과, zero-B 어댑터를 막 붙인 모델이 같은 logits를 내는지 검사합니다.

## 3. 실행 환경·파일 보호

**기존 Ubuntu A10 VM 터미널**, `~/llm-lab/source`의 `.venv`에서 실행합니다. 기존 CUDA PyTorch는 유지하고 Transformers4.57.1·PEFT0.17.1과 **bitsandbytes0.48.2**를 사용합니다. bitsandbytes는 이번 4비트 로딩·계산을 담당합니다. 별도 Trainer·TRL·새 CUDA Toolkit 설치는 없습니다.

- VM 코드: `~/llm-lab/source/smollm2_qlora.py`.
- VM 새 결과: `~/llm-lab/smollm2_qlora_v1/`.
- 원본: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- `train`은 원본이 캐시에 없으면 다운로드할 수 있습니다. `check`는 같은 고정 원본을 로컬 캐시에서 읽습니다.
- `--device cuda`를 명시하고 모델 전체를 해당 장치에 배치합니다. `device_map="auto"`로 CPU에 일부를 넘기지 않습니다. CUDA가 없으면 CPU로 자동 전환하지 않습니다.
- 기존 출력 폴더는 덮어쓰지 않습니다. 실패한 새 폴더에는 일부 파일이 남을 수 있으므로 자동 삭제하지 않습니다. 기존 원본·SFT·LoRA를 지우거나 보호 검사를 제거해 우회하지 않습니다.
- CPU 실행은 개발자 검증용 별도 경로입니다. CPU 실행 성공을 A10 CUDA 실행 성공으로 해석하지 않습니다.

공식 bitsandbytes0.48.2 설치 표에는 Linux x86-64의 CUDA12.8/12.9와 sm86 바이너리가 포함됩니다. 이것은 패키지 지원 정보이며, 특정 VM의 실제 실행 성공은 아래 명령의 출력으로 확인합니다. 드라이버·CUDA 오류가 나면 마지막 오류와 패키지/GPU 출력을 먼저 확인하고 임의로 재설치하지 않습니다.

소스: [01-qlora.py](source/20-qlora/01-qlora.py). 아래 전체 명령은 같은 소스를 새 VM 파일로 만들고 학습·새 프로세스 복원을 연속 실행합니다. `&&`는 앞 명령이 실패하면 뒤 명령을 실행하지 않는다는 뜻이고, `set -C`는 기존 코드 파일 덮어쓰기를 막습니다. `python -O`는 검사문을 끄므로 사용하지 않습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA 사용 불가'; print('GPU:', torch.cuda.get_device_name(0))" &&
python -m pip install 'transformers==4.57.1' 'peft==0.17.1' 'bitsandbytes==0.48.2' &&
(set -C; cat > smollm2_qlora.py <<'PY'
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
PY
) && python smollm2_qlora.py train --device cuda && python smollm2_qlora.py check --device cuda
```

이미 정상 저장했다면 재학습 없이 같은 폴더에서 복원만 확인합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python smollm2_qlora.py check --device cuda
```

## 4. 코드 흐름과 검사 의미

### 설정·4비트 로딩

`BitsAndBytesConfig`의 `load_in_4bit`, `nf4`, double quantization, float32 compute 설정을 명시합니다. 원본 로딩 시 장치를 정하고, 로딩 후 실제 `Linear4bit` 모듈과 NF4 양자화 상태가 있는지 확인합니다. 일반 float32 모델을 불러온 뒤 이름만 QLoRA라고 표시하지 않습니다. 양자화된 모델 전체에 별도의 `.to()`를 호출하는 흐름도 사용하지 않습니다.

`prepare_model_for_kbit_training(..., use_gradient_checkpointing=False)`를 학습과 복원 양쪽에 적용해 비양자화 층의 정밀도도 맞춥니다. 학습 대상은 `lora_` 이름의 어댑터뿐이며 optimizer에도 그 대상만 전달합니다.

### 데이터·loss·자유 생성

앞 장의 `gpu → failed`, `network → healthy`는 **실제 장비 상태가 아닌 합성 규칙**입니다. 질문 토큰은 입력에 남기고 labels만 -100으로 가립니다. 응답과 eos가 학습 대상이며 CausalLM 내부 정답 이동을 다시 수동 적용하지 않습니다. 두 예제의 토큰 평균 loss를 각각 계산해 평균합니다.

정답을 붙여 계산하는 teacher-forced loss와 질문만 넣고 생성하는 greedy 응답을 나눠 출력합니다. 생성 길이·sampling·beam·특수 토큰 설정도 명시합니다. loss가 낮아져도 `failed`/`healthy`가 생성되지 않을 수 있습니다. 나쁜 생성 결과를 실행 오류로 취급하거나 설정을 자동 조정하지 않습니다.

### 원본 동결·어댑터 갱신

압축 원본 가중치와 양자화 상태의 해시를 전후 비교하고, 원본 파라미터에 gradient가 없는지 검사합니다. 어댑터는 유한한 gradient로 실제 갱신됐는지 확인합니다. 원본 상태를 CPU로 복사해 검사하는 비용이 있으므로 전체 실행 시간을 학습 속도 측정치라고 읽지 않습니다.

### 저장·새 프로세스 복원

저장물은 어댑터·토크나이저·복원 기준과 설정입니다. **양자화한 원본 전체를 독립 완성 모델로 저장하는 것이 아닙니다.** 복원에는 고정 revision 원본 캐시와 같은 양자화 설정이 필요합니다. 저장된 어댑터의 base/revision과 실행 설정을 확인해 다른 원본·양자화 방식으로 조용히 복원하지 않습니다.

`check`는 학습하지 않으며 동결된 원본+어댑터로 네 질문의 생성 문자열과 마지막 위치 전체 어휘 logits를 다시 계산해 저장 전 결과와 비교합니다. 같은 장치의 train/check를 기본으로 하며 다른 장치·라이브러리 버전까지 완전한 문자열 일치를 보장하지 않습니다. optimizer·RNG 상태가 없으므로 정확한 학습 재개 체크포인트도 아닙니다.

## 5. 검증 범위와 출력 읽기

### 실제 CPU 실행 예시 — A10 CUDA 결과가 아님

Windows·Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2 격리 환경에서 실제 NF4 학습과 새 프로세스의 오프라인 복원을 실행했습니다. 가짜 양자화나 미학습 fixture 결과가 아닙니다. 아래는 실제 train/check stdout이며 환경·장치에 따라 수치나 생성이 달라질 수 있습니다. **별도 A10 CUDA 실행 예시는 아래에 구분하며, 속도·GPU 메모리는 측정하지 않았습니다.**

```text
실행 장치: cpu | compute: float32 | NF4 double quant Linear4bit: 210
[quantized-base/train] failed | loss=14.5004 | 생성='Device GPU status:\n\nDevice GPU'
[quantized-base/train] healthy | loss=11.4819 | 생성='Device network status:\n\nDevice network'
[quantized-base/probe] failed | loss=14.3683 | 생성='The gpu status is the status of'
[quantized-base/probe] healthy | loss=8.5856 | 생성='The network status is "Not Available".'
QLoRA 학습 대상: 230,400 | q/v rank4 alpha8 dropout0
[qlora/train] failed | loss=12.7709 | 생성='Device GPU status:\n\nDevice GPU'
[qlora/train] healthy | loss=9.8336 | 생성='Device network status:\n\nDevice network'
[qlora/probe] failed | loss=12.4268 | 생성='The gpu status is "not available'
[qlora/probe] healthy | loss=7.4256 | 생성='The network status is currently "Not Available'
QLoRA 별도 저장: smollm2_qlora_v1 | 어댑터 가중치: 937056 bytes | 원본/양자화 상태 불변
실행 장치: cpu | compute: float32 | NF4 double quant Linear4bit: 210
[restored/train] failed | loss=12.7709 | 생성='Device GPU status:\n\nDevice GPU'
[restored/train] healthy | loss=9.8336 | 생성='Device network status:\n\nDevice network'
[restored/probe] failed | loss=12.4268 | 생성='The gpu status is "not available'
[restored/probe] healthy | loss=7.4256 | 생성='The network status is currently "Not Available'
QLoRA 복원 검사 통과: NF4 원본+어댑터·logits/생성 일치·재학습 없음
```

이 설정에서는 네 질문 모두 정답 응답 loss가 낮아졌지만, 요구한 한 단어 정답을 생성하지 못했습니다. 18장의 937,056bytes와 같은 어댑터 크기도 정상입니다. **양자화한 것은 원본이고 저장한 q/v 어댑터의 shape는 같기 때문**입니다. 이 크기는 4비트 원본 크기나 GPU 메모리 사용량이 아닙니다.

로컬 통합 검사는 실제 원본 q/v shape로 어댑터 파라미터 수를 독립 계산하고 LoRA-only 키·유한값·0이 아닌 B, 원본/양자화 상태 불변, 새 프로세스 복원, 캐시·후보 파일 불변, 기존 폴더 거부, 복원 기준/설정/base/revision 변조 탐지와 CUDA 부재 시 출력 생성 전 실패를 확인했습니다. 이 통합 실행은 학습을 포함하며 아래 정적 검사기의 `training_executed=false`와 구분합니다.

### A10 CUDA 실행 로그 예시

Ubuntu VM에서 bitsandbytes0.48.2 설치 후 실행한 제공 로그에서도 CUDA·float32·실제 NF4 학습, 원본/양자화 상태 불변, 어댑터 별도 저장과 새 프로세스 복원 통과를 확인했습니다. 아래는 해당 로그의 실행 결과 부분이며 위 CPU 검사와 별도 근거입니다. 일부 loss 끝자리는 CPU와 다르지만 생성 문자열과 어댑터 크기는 같았습니다. 이 로그만으로 실행 속도·GPU 메모리·독립 품질 평가까지 확인한 것은 아닙니다.

```text
실행 장치: cuda:0 | compute: float32 | NF4 double quant Linear4bit: 210
[quantized-base/train] failed | loss=14.5004 | 생성='Device GPU status:\n\nDevice GPU'
[quantized-base/train] healthy | loss=11.4788 | 생성='Device network status:\n\nDevice network'
[quantized-base/probe] failed | loss=14.3651 | 생성='The gpu status is the status of'
[quantized-base/probe] healthy | loss=8.5710 | 생성='The network status is "Not Available".'
QLoRA 학습 대상: 230,400 | q/v rank4 alpha8 dropout0
[qlora/train] failed | loss=12.7704 | 생성='Device GPU status:\n\nDevice GPU'
[qlora/train] healthy | loss=9.8301 | 생성='Device network status:\n\nDevice network'
[qlora/probe] failed | loss=12.4212 | 생성='The gpu status is "not available'
[qlora/probe] healthy | loss=7.4152 | 생성='The network status is currently "Not Available'
QLoRA 별도 저장: smollm2_qlora_v1 | 어댑터 가중치: 937056 bytes | 원본/양자화 상태 불변
실행 장치: cuda:0 | compute: float32 | NF4 double quant Linear4bit: 210
[restored/train] failed | loss=12.7704 | 생성='Device GPU status:\n\nDevice GPU'
[restored/train] healthy | loss=9.8301 | 생성='Device network status:\n\nDevice network'
[restored/probe] failed | loss=12.4212 | 생성='The gpu status is "not available'
[restored/probe] healthy | loss=7.4152 | 생성='The network status is currently "Not Available'
QLoRA 복원 검사 통과: NF4 원본+어댑터·logits/생성 일치·재학습 없음
```

- 학습 질문·probe 모두 정답 응답 loss가 낮아졌지만 한 단어 정답은 생성하지 못했습니다. 저장·복원 성공과 과제 품질을 구분합니다.
- 실행 중 bitsandbytes CUDA 코드에서 `_check_is_size`가 향후 제거될 예정이라는 `FutureWarning`이 출력됐지만, 이어진 학습·저장·복원 검사는 완료됐습니다. 이 로그는 해당 경고 때문에 실행이 실패했다는 증거가 아닙니다. 경고를 숨기거나 패키지를 임의 교체해 성공 출력을 다시 만들 필요는 없습니다.
- 원본/양자화 상태 불변과 생성/logits 일치는 코드 검사 통과 로그로 확인한 범위입니다. 이 로그가 저장 파일의 별도 다운로드·내용 검사나 성능 측정을 대신하지는 않습니다.

실행 결과에서 다음을 구분합니다.

1. **구조 성공:** 실제 NF4 원본이 로딩됐고 어댑터만 갱신됨.
2. **저장·복원 성공:** 새 프로세스의 생성/logits가 저장 전과 일치함.
3. **과제 품질:** train/probe의 실제 단어 응답과 loss가 어떻게 바뀌었는지 관찰함.

앞의 두 성공이 세 번째 성공을 보장하지 않습니다. 이 장의 float32 compute·작은 q/v 어댑터·두 예제·SGD20회 결과를 QLoRA 전체의 성능으로 일반화하지 않습니다. 19장의 float32 평가 스크립트에 새 4비트 어댑터 경로만 바꿔 넣는 것도 동일 설정 평가가 아닙니다. 이번에는 이 장의 고정된 train/check 관찰만 합니다.

개발자 검사는 저장소 루트에서 같은 의존성 환경으로 실행합니다. 실제 CPU NF4 학습을 수행하는 검사이며 정적 문서 검사나 CUDA 검사와 구분합니다.

```bash
python source/check_guide.py
python source/20-qlora/test_qlora.py
```

검사 소스: [test_qlora.py](source/20-qlora/test_qlora.py). 오류가 나면 전체 마지막 traceback과 장치·패키지 정보를 확인합니다. SSL 검증을 끄거나, 원본 캐시를 삭제하거나, CPU fallback으로 CUDA 성공을 대신하지 않습니다.

## 6. 복습과 다음 묶음

- [ ] 4비트 원본 저장과 float32 연산을 구분한다.
- [ ] 원본이 동결돼도 LoRA까지 역전파할 수 있는 이유를 설명한다.
- [ ] LoRA 초기 출력 비교의 기준이 float32 원본이 아니라 준비된 4비트 원본임을 이해한다.
- [ ] 어댑터 복원에 같은 원본 revision·양자화 설정이 필요한 이유를 설명한다.
- [ ] 저장·복원 성공과 정답률·메모리·속도를 별개로 해석한다.

다음 [21장 저장 모델의 API 서빙](21-api-serving.md)은 모델을 로딩하고 로컬 HTTP 요청을 받아 응답하는 경로를 배웁니다. 이 장의 학습·저장 코드와 분리하며 외부 노출·운영 배포는 포함하지 않습니다.

## 공식 참고자료

- [Transformers4.57.1 · bitsandbytes 양자화](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes)
- [PEFT0.17 · 양자화 모델 학습 준비](https://huggingface.co/docs/peft/v0.17.0/developer_guides/quantization)
- [bitsandbytes0.48.2 · 설치·하드웨어 지원](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)

[← 이전: 모델 비교 평가](19-model-evaluation.md) · [목차](README.md) · [다음: API 서빙 →](21-api-serving.md)
