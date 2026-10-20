# 12 · 새 입력으로 모델의 한계 관찰

[← 이전: 저장·복원·생성](11-checkpoint-generation.md) · [목차](README.md) · [다음: 정답이 있는 평가 →](13-next-token-evaluation.md)

## 1. 목표와 선행 조건

11장에서는 학습한 두 문장을 저장 모델이 스스로 생성하는지 확인했습니다. 이번에는 **모델·가중치를 고정하고 입력만 바꿔**, 익숙한 문맥의 재현과 새로운 입력의 반응을 비교합니다. 재학습이나 데이터 보완은 하지 않습니다.

필요한 파일은 11장에서 만든 `~/llm-lab/source/tiny_transformer_checkpoint.py`와 `~/llm-lab/tiny_transformer_v1.pt`입니다. 자신이 만든 신뢰할 수 있는 체크포인트만 읽습니다.

| 구분 | 입력 | 확인할 질문 |
|---|---|---|
| 학습 접두 문맥 | `gpu current status`, `network current status` | 이전 결과가 그대로 나오는가? |
| 단어 생략 | `gpu status`, `network status` | 등록 단어라도 조합이 바뀌면 어떤 문장을 만드는가? |
| 순서 변경 | `status current gpu` | 같은 단어가 다른 위치에 오면 어떻게 반응하는가? |
| 상태 조합 변경 | `gpu current status healthy` | 학습의 GPU 예문과 다른 상태도 그대로 끝내는가? |
| 미등록 단어 | `storage status` | 모델 계산 전에 입력 검사가 거부하는가? |

여기서 새 입력은 **두 학습 문장의 접두 부분으로 사용하지 않은 입력**입니다. 모든 단어가 새롭다는 뜻은 아닙니다. 위의 상태 표현은 교육용 문자열이며 실제 GPU·네트워크 상태를 측정한 데이터가 아닙니다.

## 2. 관찰과 평가의 차이

- **재현 검사:** 학습한 문맥에서 정해진 다음 토큰을 만드는지 검사합니다.
- **정성 관찰:** 정답을 미리 지정하지 않은 입력에서 무엇을 만드는지 읽어 봅니다.
- **일반화 평가:** 목적에 맞는 별도 입력·정답·측정 기준으로 성능을 확인합니다. 이번 실습에는 포함하지 않습니다.

새 입력에 정답을 붙이지 않았으므로 정확도를 계산하지 않습니다. `storage` 거부는 신경망의 의미 판단이 아니라 기존 `generate` 함수의 어휘 검사입니다. `<eos>` 종료도 문장이 옳다는 뜻이 아닙니다.

## 3. 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source/.venv`에서 실행합니다. CPU를 사용하며 패키지를 설치하지 않습니다. 아래 명령은 코드를 파일로 기록하지 않고 바로 실행합니다. 체크포인트를 읽고, 실행 전후 파일 바이트가 같은지 확인합니다. 코드의 저장·학습 함수를 호출하지 않습니다.

`cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 `PY` 사이의 코드를 표준 입력에서 실행한다는 뜻입니다. `python -O`는 assert 검사를 생략하므로 사용하지 않습니다.

소스: [04-probe-inputs.py](source/09-transformer/04-probe-inputs.py). 저장소의 원본 소스도 **11장의 VM 배치 폴더에서** 실행하는 코드입니다. 저장소 루트에서 바로 실행하면 VM 파일을 찾지 못합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
"""11장의 VM 배치 폴더에서 기존 모델을 바꾸지 않고 새 입력을 관찰합니다."""
from pathlib import Path
import runpy
import torch

torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original = checkpoint.read_bytes()
model = lm["restore"](checkpoint)
generate = lm["generate"]

controls = {"gpu current status": ["gpu", "current", "status", "failed", "<eos>"],
            "network current status": ["network", "current", "status", "healthy", "<eos>"]}
# ponytail: 선택한 입력의 정성 관찰뿐; 일반화 점수는 별도 정답 데이터로 평가합니다.
for prompt in [*controls, "gpu status", "network status", "status current gpu",
               "gpu current status healthy"]:
    tokens, reason = generate(model, prompt.split())
    if prompt in controls:
        assert tokens == controls[prompt] and reason == "eos"
    print(f"입력: {prompt} -> {' '.join(tokens)} | 종료: {reason}")

try:
    generate(model, ["storage", "status"])
except ValueError as exc:
    print(f"미등록 입력 거부: storage status | {exc}")
else:
    raise AssertionError("미등록 토큰을 허용했습니다")
assert checkpoint.read_bytes() == original, "체크포인트가 변경됐습니다"
print("검사 통과: 학습 접두 문맥 재현·미등록 입력 거부·체크포인트 불변")
PY
```

## 4. 코드의 흐름

```text
기존 코드 로드 → 체크포인트 바이트 보관 → 모델 복원
                                 ↓
문자열 → split으로 토큰 목록 → 기존 generate → 전체 토큰·종료 이유 출력
                                 ↓
미등록 입력 거부 검사 → 체크포인트 바이트 재비교
```

| 코드 | 역할 |
|---|---|
| `runpy.run_path(...)` | 기존 파일의 정의를 딕셔너리로 가져옵니다. 기본 실행 이름은 `__main__`이 아니므로 기존 CLI의 save/generate 분기는 실행하지 않습니다. |
| `lm["restore"]` | 기존 구조·어휘·가중치·저장 전후 logits 검사로 복원합니다. |
| `controls` | 학습 접두 문맥 두 개의 재현 기준입니다. 생성 함수에 정답을 전달하지 않습니다. |
| `[*controls, ...]` | 딕셔너리의 두 입력 문자열에 관찰용 입력 네 개를 이어 붙입니다. |
| `prompt.split()` | 공백 기준으로 나눈 문자열 리스트입니다. 새 tokenizer를 학습하지 않습니다. |
| `generate(...)` | 기존 함수가 토큰 ID로 바꿔 마지막 위치 점수의 argmax를 반복해서 붙입니다. |
| `if prompt in controls` | 기준 입력에만 정답 검사를 적용합니다. 새 입력 출력을 강제로 맞추지 않습니다. |
| `original` | 작은 체크포인트 전체 바이트입니다. 이 교육용 파일의 변경 여부만 검사하며 대용량 파일 검증 방식은 아닙니다. |

길이 L의 입력은 `1×L` ID 텐서, 출력 점수는 `1×L×7`입니다. 생성 함수는 마지막 위치 후보 일곱 개 중 하나를 고릅니다. L은 1~4이며 최대 생성 수는 기존 기본값 4입니다. 미등록 토큰은 ID 변환과 모델 계산 전에 거부합니다.

## 5. 실제 로컬 실행 예시

Python 3.12·CPU PyTorch 격리 환경에서 11장 코드로 **임시 체크포인트를 별도 재현 학습·저장한 뒤**, 위 관찰 코드를 실제 실행한 결과입니다. 아래 관찰 코드 자체에는 학습·저장이 없습니다. VM 실행 결과와는 구분하며, 환경·학습된 가중치에 따라 새 입력의 예측은 달라질 수 있습니다.

```text
입력: gpu current status -> gpu current status failed <eos> | 종료: eos
입력: network current status -> network current status healthy <eos> | 종료: eos
입력: gpu status -> gpu status status healthy <eos> | 종료: eos
입력: network status -> network status status healthy <eos> | 종료: eos
입력: status current gpu -> status current gpu current status | 종료: context
입력: gpu current status healthy -> gpu current status healthy <eos> | 종료: eos
미등록 입력 거부: storage status | 시작 문맥은 eos를 제외한 등록 토큰 1~4개의 리스트여야 합니다
검사 통과: 학습 접두 문맥 재현·미등록 입력 거부·체크포인트 불변
```

### 결과 읽기

1. **앞의 두 줄:** 학습 접두 문맥에서 앞 장의 문장을 재현했습니다.
2. **단어 생략:** `status`를 반복하고 두 시작 단어 모두 `healthy`로 이어졌습니다. 단어가 어휘에 있다는 사실만으로 임의 조합을 기대대로 처리한다고 볼 수 없습니다. 이 관찰에는 미리 정한 정답이나 정확도가 없습니다.
3. **순서 변경:** `current status`를 덧붙인 뒤 다음 입력이 다섯 토큰이 되어 `context`로 멈췄습니다. 출력에 다섯 토큰이 보이는 것은 다섯 토큰을 모델에 넣었다는 뜻이 아닙니다. 네 토큰 입력에서 다섯 번째를 예측하는 것은 가능하지만, 그 전체를 다시 넣지는 않습니다.
4. **상태 조합 변경:** 입력으로 준 `healthy`를 고치지 않고 `<eos>`를 붙였습니다. 이 함수는 앞 문장을 교정하거나 학습 예문과 모순을 검증하는 기능이 없습니다. 또한 실제 GPU는 healthy일 수도 있으므로 이 교육용 학습 문장과 다르다는 이유만으로 현실의 오답이라고 판단하지 않습니다.
5. **미등록 단어:** 입력 검사에서 `ValueError`를 받았습니다. 어휘가 고정되어 `storage`를 처리할 수 없으며, 별도 unknown 토큰으로 바꾸지도 않습니다.

종료 이유 `eos`는 종료 토큰 예측, `limit`는 새 토큰 수 제한, `context`는 다음 계산에 필요한 문맥 길이 제한입니다. 어느 것도 정확도나 신뢰도 점수가 아닙니다.

## 6. 검사·한계·다음 단계

[독립 검사](source/09-transformer/test_probe.py)는 개발용 PyTorch 환경의 **저장소 루트**에서 실행합니다.

```bash
python source/09-transformer/test_probe.py
```

이 검사는 임시 VM형 배치 폴더를 만들고 11장의 save를 실제 실행한 뒤 관찰 코드를 별도 프로세스에서 실행합니다. 기준 문맥 두 개·관찰 입력 네 개·미등록 입력 거부·기존 체크포인트 불변과 새 체크포인트 미생성을 확인합니다. VM이나 기존 체크포인트에는 접근하지 않습니다.

- 이 소수의 선택 입력은 엄격한 블라인드 테스트 세트가 아닙니다. 출력에 맞춰 모델을 조정하면 이후에도 독립 테스트로 사용할 수 없습니다.
- 같은 체크포인트와 입력의 argmax 생성은 확률 샘플링이 아닙니다. 여러 번 돌린 결과를 서로 다른 평가 표본처럼 세지 않습니다.
- 체크포인트 파일 불변은 메모리의 모든 상태가 불변이라는 일반 증명이 아닙니다. 이번 경로는 기존 eval/no_grad 생성만 사용하며 optimizer를 실행하지 않습니다.
- 정답 데이터·정확도·평가 loss·데이터 보완·재학습은 포함하지 않습니다.
- [13장](13-next-token-evaluation.md)에서는 합성 과제의 정답을 먼저 정하고 형식별 train/validation/test의 마지막 상태 위치를 평가합니다. 단순히 관찰 출력을 정답으로 복사하지 않습니다.

## 7. 복습

- [ ] 등록된 단어와 학습에서 사용한 입력 조합을 구분한다.
- [ ] 학습 접두 문맥 재현과 새 입력 관찰을 구분한다.
- [ ] 새 입력에 정답 검사가 없는 이유를 설명한다.
- [ ] eos 종료가 내용의 옳음을 보장하지 않는 이유를 설명한다.
- [ ] 미등록 단어 거부가 모델의 의미 판단이 아님을 설명한다.
- [ ] 정성 관찰과 별도 정답 데이터에 대한 평가를 구분한다.

[← 이전: 저장·복원·생성](11-checkpoint-generation.md) · [목차](README.md) · [다음: 정답이 있는 평가 →](13-next-token-evaluation.md)
