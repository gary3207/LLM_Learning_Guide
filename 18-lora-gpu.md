# 18 · GPU에서 LoRA: 작은 어댑터 학습·저장·복원

[← 이전: 전체 파라미터 SFT](17-sft-compare.md) · [목차](README.md) · [다음: 모델 비교 평가 →](19-model-evaluation.md)

## 1. 이번에 완성할 결과물

17장은 전체 가중치를 갱신했습니다. 이번에는 **원본은 동결하고 일부 층에 작은 LoRA 가중치만 학습 → 같은 질문으로 비교 → 어댑터만 별도 저장 → 원본과 결합해 새 프로세스에서 복원**합니다. 한 통합 실습이며 GPU 속도 벤치마크나 최적 학습 설정 탐색은 하지 않습니다.

기존 17장의 학습 모델에서 이어가는 것이 아닙니다. **같은 고정 revision의 원본 SmolLM2-135M-Instruct에서 다시 출발**합니다. 17장의 두 학습 질문·정답, 두 probe, 응답/eos 전용 loss, SGD·학습률0.01·20회·gradient norm 상한1을 유지하고 학습 대상을 바꿉니다. 일반 장비 상태 조회가 아니라 `gpu → failed`, `network → healthy`라는 합성 규칙입니다.

| 방식 | 갱신 대상 | 별도 저장되는 가중치 |
|---|---|---|
| 17장 전체 파라미터 SFT | 전체 134,515,008개 | 수정된 전체 모델 |
| 18장 LoRA | q_proj·v_proj에 붙인 230,400개 | 작은 어댑터만 |

LoRA 학습 대상 수는 원본 파라미터 수의 **0.1713%**였습니다. 이 비율은 원본 모델을 분모로 사용하며, 어댑터를 더한 총 파라미터 수가 분모인 표시와 구분합니다. 줄어드는 것은 학습 대상과 저장할 변경분입니다. 원본 모델의 추론 계산·GPU 적재가 사라지는 것이 아닙니다.

## 2. LoRA의 작은 행렬은 무엇인가

PyTorch 선형층 가중치가 `W: (출력좌표, 입력좌표)`라면 LoRA는 다음 변경분을 추가합니다.

```text
W는 동결
A: (r, 입력좌표)      B: (출력좌표, r)
변경분 ΔW = (alpha / r) × B @ A
출력 = 원래 선형층 출력 + LoRA 변경분의 출력
```

이번에는 `r=4`, `alpha=8`, dropout0입니다. 예를 들어 첫 층의 q_proj는 W가 `(576, 576)`이며 A는 `(4, 576)`, B는 `(576, 4)`입니다. 첫 층 v_proj의 W는 `(192, 576)`이므로 같은 r에서도 크기가 다릅니다. 모델의 모든 q_proj·v_proj를 대상으로 하고, k_proj·FFN·임베딩 등 다른 원본 가중치는 동결합니다.

`r`은 단순한 학습률이 아니라 변경분의 저차원 크기입니다. `alpha/r`는 변경분의 배율이고 SGD의 학습률과 별개입니다. 기본 초기화는 A를 초기화하고 B를 0으로 두므로 시작 시 변경분이 0입니다. 실제 원본과 LoRA 부착 직후 네 질문의 logits가 같은지도 확인합니다.

17장에서 배운 labels 마스크·정답 이동·자유 생성 방식은 유지합니다. 별도 소스는 단독 실행이 가능하도록 같은 함수 흐름을 재사용한 교육 예제이며, 이전 실행 코드를 import해서 의도치 않게 재학습하지 않습니다.

## 3. Ubuntu A10 VM에서 전체 실행

**기존 Ubuntu VM 터미널**, `~/llm-lab/source/.venv`에서 실행합니다. 기존 PyTorch와 Transformers 4.57.1을 사용하고 **PEFT 0.17.1과 그 의존성**을 추가합니다. torch 재설치·드라이버·CUDA 변경 명령은 없습니다. PEFT는 LoRA 부착·저장·복원 기능을 제공하며 별도 Trainer·TRL은 설치하지 않습니다.

- 새 코드: `~/llm-lab/source/smollm2_lora.py`.
- 새 결과: `~/llm-lab/smollm2_lora_v1/`.
- 17장 `smollm2_sft_v1/`와 원본 모델 캐시는 유지합니다.
- GPU와 float32를 사용합니다. 양자화·QLoRA·mixed precision은 포함하지 않습니다.
- **CPU 검증과 별도로 Ubuntu·NVIDIA A10의 CUDA 학습·복원 실행 로그도 확인했습니다. 두 환경의 출력 예시는 아래에서 구분합니다.** 각 환경의 성공 여부는 자신의 실행 출력으로 확인합니다.
- CUDA가 없으면 출력 폴더를 만들기 전에 명확히 실패합니다. CPU로 조용히 전환하지 않습니다.
- 코드·출력 폴더가 있으면 덮어쓰지 않습니다. 중단된 새 폴더는 일부 파일만 있을 수 있으므로 자동 삭제하지 말고 확인합니다.

소스: [01-lora-gpu.py](source/18-lora/01-lora-gpu.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA 사용 불가'; print('GPU:', torch.cuda.get_device_name(0))" &&
python -m pip install 'transformers==4.57.1' 'peft==0.17.1' &&
(set -C; cat > smollm2_lora.py <<'PY'
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
PY
) && python smollm2_lora.py train --device cuda && python smollm2_lora.py check --device cuda
```

`--device cuda`는 명시적으로 GPU를 선택합니다. 코드의 기본 장치는 로컬 검증용 CPU이므로, GPU 명령에서는 train/check 양쪽에 옵션을 붙입니다. 실패 시 오류를 보고 장치를 확인하며 CUDA를 임의로 재설치하지 않습니다. `&&`는 실패 시 뒤 명령을 막습니다. `python -O`는 사용하지 않습니다.

이미 저장된 후보의 복원만 검사하려면 같은 폴더·환경에서 `python smollm2_lora.py check --device cuda`를 실행합니다. 출력 폴더를 삭제해 다시 학습할 필요는 없습니다.

## 4. 무엇을 검사하고 무엇을 저장하는가

### 동결과 실제 갱신

`get_peft_model()`이 원본에 어댑터를 부착합니다. `requires_grad=True`인 파라미터가 실제로 LoRA뿐인지 확인하고, optimizer에는 그 파라미터만 전달합니다. 매회 원본 파라미터에 gradient가 없는지, 학습 후 어댑터가 바뀌었는지 검사합니다.

`frozen_digest()`는 동결된 전체 가중치를 CPU로 읽어 SHA-256을 계산하고 전후를 비교합니다. 원본 값의 불변 확인이며 타이밍 측정이 아닙니다. GPU에서는 이 검사에 CPU 복사 비용이 포함되므로 전체 실행 시간을 LoRA 학습 속도라고 보고하면 안 됩니다.

### 어댑터 저장과 복원

PEFT 모델의 `save_pretrained()`는 원본 전체가 아니라 `adapter_model.safetensors`와 `adapter_config.json`을 저장합니다. 토크나이저와 교육용 복원 기준 `reference.pt`도 같은 폴더에 둡니다. 어댑터 가중치 파일의 실제 크기는 **937,056 bytes**였습니다. 이 수치는 전체 폴더 크기나 GPU 메모리 사용량이 아닙니다.

**어댑터만으로는 완전한 LLM을 실행할 수 없습니다.** `check`는 고정 revision의 원본 모델을 기존 로컬 캐시에서 다시 읽고 `PeftModel.from_pretrained(..., is_trainable=False)`로 어댑터를 부착합니다. 원본 캐시가 없으면 local-only 복원은 실패합니다. 원본을 다운로드하지 않아도 되는 독립 완성 모델로 저장한 것이 아닙니다.

저장 전후 네 질문의 실제 생성 문자열과 마지막 위치 전체 어휘 logits를 비교합니다. CPU로 옮겨 저장한 logits에 수치 허용오차를 적용하며, 생성 문자열은 일치해야 합니다. GPU/CPU를 바꿔도 어떤 입력이든 동일한 문자열이 나온다는 보장은 아닙니다. 기본 실습은 같은 장치에서 train/check합니다. optimizer/RNG 상태는 저장하지 않으므로 정확한 학습 재개용 체크포인트는 아닙니다.

## 5. 실제 CPU 검증 출력 — CUDA 실행 예시가 아님

Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1의 격리된 Windows 로컬 환경에서 `train --device cpu`, 별도 프로세스 `check --device cpu`를 실행했습니다. 같은 원본 캐시를 사용했고, 아래는 실제 stdout입니다. **A10 실행·GPU 메모리·속도를 검증한 출력이 아닙니다.** 장치·패키지·계산 조건에 따라 값·문장이 달라질 수 있습니다.

```text
실행 장치: cpu | dtype: torch.float32
[base/train] failed | loss=14.4325 | 생성='Device GPU status: 1'
[base/train] healthy | loss=11.1671 | 생성='Device network status:\n\n- Device'
[base/probe] failed | loss=12.3040 | 생성="The GPU status is currently '0'."
[base/probe] healthy | loss=7.1799 | 생성='The network status is currently stable and functioning'
전체 SFT 학습 대상: 134,515,008 | LoRA 학습 대상: 230,400 | 비율: 0.1713%
[lora/train] failed | loss=11.8943 | 생성='Device GPU status: 0\n\n'
[lora/train] healthy | loss=8.2563 | 생성='Device network status: Disconnected'
[lora/probe] failed | loss=8.6117 | 생성="The GPU status is currently '0'."
[lora/probe] healthy | loss=6.0967 | 생성='The network status is currently stable and functioning'
LoRA 별도 저장: smollm2_lora_v1 | 어댑터 가중치: 937056 bytes | 원본 동결 가중치 불변
실행 장치: cpu | dtype: torch.float32
[restored/train] failed | loss=11.8943 | 생성='Device GPU status: 0\n\n'
[restored/train] healthy | loss=8.2563 | 생성='Device network status: Disconnected'
[restored/probe] failed | loss=8.6117 | 생성="The GPU status is currently '0'."
[restored/probe] healthy | loss=6.0967 | 생성='The network status is currently stable and functioning'
LoRA 복원 검사 통과: 원본+어댑터·logits/생성 일치·재학습 없음
```

### A10 GPU 실행 로그 예시

Ubuntu·NVIDIA A10·PyTorch2.11.0+cu128·Transformers4.57.1·PEFT0.17.1 환경의 제공된 실행 로그에서도 CUDA 학습·어댑터 저장·새 프로세스 복원 검사 통과를 확인했습니다. CPU 예시와 일부 loss 끝자리는 다르지만 생성 문자열과 어댑터 크기는 같았습니다. 아래 로그는 GPU 메모리나 실행 속도 측정 결과가 아닙니다.

```text
torch: 2.11.0+cu128 CUDA: True
GPU: NVIDIA A10
실행 장치: cuda:0 | dtype: torch.float32
[base/train] failed | loss=14.4325 | 생성='Device GPU status: 1'
[base/train] healthy | loss=11.1671 | 생성='Device network status:\n\n- Device'
[base/probe] failed | loss=12.3041 | 생성="The GPU status is currently '0'."
[base/probe] healthy | loss=7.1800 | 생성='The network status is currently stable and functioning'
전체 SFT 학습 대상: 134,515,008 | LoRA 학습 대상: 230,400 | 비율: 0.1713%
[lora/train] failed | loss=11.8943 | 생성='Device GPU status: 0\n\n'
[lora/train] healthy | loss=8.2563 | 생성='Device network status: Disconnected'
[lora/probe] failed | loss=8.6117 | 생성="The GPU status is currently '0'."
[lora/probe] healthy | loss=6.0968 | 생성='The network status is currently stable and functioning'
LoRA 별도 저장: smollm2_lora_v1 | 어댑터 가중치: 937056 bytes | 원본 동결 가중치 불변
실행 장치: cuda:0 | dtype: torch.float32
[restored/train] failed | loss=11.8943 | 생성='Device GPU status: 0\n\n'
[restored/train] healthy | loss=8.2563 | 생성='Device network status: Disconnected'
[restored/probe] failed | loss=8.6117 | 생성="The GPU status is currently '0'."
[restored/probe] healthy | loss=6.0968 | 생성='The network status is currently stable and functioning'
LoRA 복원 검사 통과: 원본+어댑터·logits/생성 일치·재학습 없음
```

### 이번에는 loss가 낮아져도 정답 단어를 생성하지 못함

train/probe의 정답 응답 loss는 낮아졌지만, 이 설정의 LoRA는 `failed`·`healthy` 정답을 생성하지 못했습니다. 17장의 CPU 전체 파라미터 SFT는 같은 네 질문에서 정답 단어를 생성했습니다. **같은 학습률·갱신 수가 서로 다른 파라미터화에서 같은 학습 효과를 주는 것은 아닙니다.**

이 결과를 보고 학습률·r·갱신 수를 자동 조정해 성공 출력을 만들지 않습니다. 이번 목표는 LoRA의 동결·학습 대상 감소·작은 저장물·원본과 결합하는 복원 구조를 배우고, 품질과 별도로 확인하는 것입니다. LoRA가 항상 전체 SFT보다 나쁘거나 좋다는 결론도 낼 수 없습니다.

로컬 CPU 비교는 같은 모델·데이터·optimizer 설정을 유지한 작은 관찰입니다. 독자가 17장 CPU와 18장 GPU 결과를 비교하면 장치까지 달라지므로 더 엄격한 동일 조건 품질 실험이나 속도 벤치마크가 아닙니다. 추후 설정 탐색이 필요하다면 probe를 개발용으로 구분하고 별도 독립 평가를 설계해야 합니다. 현재 두 probe는 같은 기기·정답을 공유하는 진단 입력이며 넓은 일반화·기존 생성 유지·실제 장비 상태 판단을 검증하지 않습니다.

## 6. 개발자 검사와 마무리

동일한 패키지가 있는 개발 환경의 저장소 루트에서 [test_lora.py](source/18-lora/test_lora.py)를 실행할 수 있습니다. CPU를 명시해 실제 LoRA 학습을 수행하는 검사이며 GPU 테스트로 세지 않습니다.

```bash
python source/18-lora/test_lora.py
```

실제 원본 q/v 가중치의 shape로 rank4 어댑터 파라미터 수를 독립 계산하고, 저장된 키·텐서 수·유한값·0이 아닌 B를 검사합니다. 어댑터만 저장되고 원본 전체 바이너리가 포함되지 않는지, 원본 캐시·정상 복원 후 후보 파일 해시 불변인지, 기존 폴더 재저장 거부와 의도적으로 바꾼 복원 기준의 불일치 탐지가 동작하는지 확인합니다. CUDA 없는 호스트에서는 CUDA 선택 시 CPU로 전환하거나 폴더를 만들지 않고 실패하는 것도 검사합니다.

CUDA·모듈·다운로드 오류는 마지막 오류 메시지를 기준으로 구분합니다. SSL 검증을 끄거나, 원본/17장 폴더를 지우거나, 출력 폴더 보호를 제거해서 우회하지 않습니다.

- [ ] 원본 동결과 어댑터 학습을 구분한다.
- [ ] rank r·배율 alpha/r·optimizer 학습률이 다른 값임을 설명한다.
- [ ] 어댑터만 저장해도 복원에는 원본 모델이 필요한 이유를 설명한다.
- [ ] 학습 대상 감소·파일 크기·GPU 메모리·실행 속도를 구분한다.
- [ ] loss 감소·정답 생성·저장복원 성공을 별개로 해석한다.

## 공식 참고자료

- [PEFT LoRA 개념·초기화](https://huggingface.co/docs/peft/v0.17.0/developer_guides/lora)
- [PEFT 어댑터 체크포인트 형식](https://huggingface.co/docs/peft/v0.17.0/developer_guides/checkpoint)
- [PEFT 0.17.1 소스](https://github.com/huggingface/peft/tree/v0.17.1)

[← 이전: 전체 파라미터 SFT](17-sft-compare.md) · [목차](README.md) · [다음: 모델 비교 평가 →](19-model-evaluation.md)
