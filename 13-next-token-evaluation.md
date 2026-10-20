# 13 · 정답이 있는 다음 토큰 평가

[← 이전: 새 입력 관찰](12-input-probes.md) · [목차](README.md)

## 1. 목표와 평가 과제

12장에서는 새 입력에 무엇을 생성하는지 관찰했습니다. 정답을 지정하지 않았으므로 정확도를 계산하지 않았습니다. 이번에는 **예측을 보기 전에 입력·정답·분리를 고정**하고 기존 체크포인트의 다음 토큰 정답률과 평균 cross-entropy loss를 계산합니다. 가중치를 바꾸거나 새 모델을 학습하지 않습니다.

평가 목적은 **단어 순서가 바뀌어도 교육용 기기→상태 연결을 유지하는가**입니다. 합성 규칙을 다음처럼 정합니다.

- `gpu`가 있는 입력의 상태 정답: `failed`
- `network`가 있는 입력의 상태 정답: `healthy`
- 이번 세 형식에서는 단어 순서가 달라도 같은 상태를 정답으로 지정합니다.

이것은 실제 장비 상태나 일반 영어 문법에 대한 정답이 아닙니다. 실제 GPU가 항상 failed라는 뜻도 아닙니다. 기존 모델이 그런 순서 불변 규칙을 학습하도록 별도 훈련됐다는 전제도 없습니다. **새 과제를 정의하고 고정 모델이 얼마나 맞히는지 측정하는 작은 실험**입니다.

## 2. 입력·정답·분리부터 고정

| 분리 | 입력 | 정답 |
|---|---|---|
| train | `gpu current status` | `failed` |
| train | `network current status` | `healthy` |
| validation | `gpu status current` | `failed` |
| validation | `network status current` | `healthy` |
| test | `current status gpu` | `failed` |
| test | `current status network` | `healthy` |

- **train:** 기존 두 학습 문장의 상태 예측 위치만 재검사합니다. 원래 학습에는 문장마다 여덟 위치가 아니라 **두 문장 합계 여덟 위치**가 있었고, 이번에는 그중 상태 위치 두 개만 사용합니다. 기존 학습 데이터를 새로 나누거나 나머지 여섯 위치를 없앤 것이 아닙니다.
- **validation:** 같은 단어 집합에서 뒤의 두 단어 순서를 바꾼 형식입니다. 보통 학습 설정·모델 선택에 사용하지만, 이번에는 고정 모델의 진단에만 사용합니다.
- **test:** 기기 단어를 마지막으로 옮긴 또 다른 형식입니다. 이번 모델과 정답을 고정한 채 한 번의 평가 흐름을 배우기 위해 사용합니다.

기기 단어 두 개를 무작위로 섞는 대신 **형식별로 분리**했습니다. 동일한 입력 문자열은 분리 사이에 중복되지 않습니다. 그러나 같은 어휘·기기·합성 정답 규칙을 공유하고 각 분리가 두 예제뿐이므로, 중복이 없다는 것만으로 독립성이나 넓은 일반화가 입증되지는 않습니다. 이전 장에서 이미 순서 변경 반응도 관찰했으므로 엄격한 블라인드 벤치마크가 아닙니다.

이번에는 모델 선택·학습률 조절·데이터 보완을 하지 않아 validation 결과를 본 뒤 모델을 바꾸지 않고 test까지 계산합니다. **여러 모델을 비교하는 실제 절차에서는 validation으로 선택한 후 test를 마지막에 사용합니다.** 이 test 출력을 보고 조정하기 시작하면 이 입력들은 개발에 사용한 데이터가 됩니다. 새 독립 평가가 필요합니다.

## 3. 정확도와 loss는 무엇이 다른가

각 입력에서 **마지막 위치 다음 토큰 하나**만 평가합니다. 두 단어 `failed`·`healthy`에만 후보를 제한하지 않고 기존 어휘 일곱 개 전체에서 예측합니다.

- **정확도:** 가장 높은 점수의 토큰이 미리 정한 정답과 같은 예제 수 / 전체 예제 수
- **한 예제 loss:** 정답 토큰의 확률을 p라고 할 때 `-ln(p)`
- **평균 loss:** 예제마다 구한 loss의 합 / 예제 수

정답이 1등이어도 다른 후보에 확률이 나뉘면 loss는 0이 아닙니다. 정확도는 1등의 일치 여부만 보지만 loss는 정답에 준 확률도 반영합니다. 확률을 보정된 현실의 신뢰도나 장비 장애 확률로 해석하지 않습니다.

코드에서는 logits에 softmax를 먼저 붙이지 않습니다. 기존 학습처럼 `CrossEntropyLoss`에 **원래 점수와 정답 ID**를 넣습니다. 정답은 모델의 입력으로 주거나 예측에 끼워 넣지 않고 결과를 채점할 때만 사용합니다.

## 4. 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source/.venv`에서 실행합니다. 11장의 `tiny_transformer_checkpoint.py`와 `../tiny_transformer_v1.pt`가 필요합니다. 자신이 만든 신뢰할 수 있는 체크포인트만 읽습니다.

`cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 `PY` 사이 코드를 바로 실행한다는 뜻입니다. CPU를 사용하고 설치·재학습·코드 파일 생성·체크포인트 덮어쓰기는 하지 않습니다. assert 검사가 필요한 예제이므로 `python -O`를 사용하지 않습니다.

소스: [05-evaluate-next-token.py](source/09-transformer/05-evaluate-next-token.py). 원본 파일도 저장소 루트가 아니라 **위 VM 배치 폴더를 현재 폴더로 두고** 실행하는 코드입니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
"""고정 모델의 마지막 위치 다음 토큰을 미리 정한 정답으로 평가합니다."""
from pathlib import Path
import runpy
import torch

# 교육용 규칙: 순서를 바꿔도 gpu의 상태 정답은 failed, network는 healthy.
# 실제 장비 상태·일반 영어 문법의 정답이 아니라 합성 평가 과제입니다.
splits = {
    "train": [("gpu current status", "failed"), ("network current status", "healthy")],
    "validation": [("gpu status current", "failed"), ("network status current", "healthy")],
    "test": [("current status gpu", "failed"), ("current status network", "healthy")],
}
prompts = [prompt for rows in splits.values() for prompt, _ in rows]
assert len(prompts) == len(set(prompts)), "분리 사이 입력이 중복됐습니다"
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original = checkpoint.read_bytes()
model = lm["restore"](checkpoint)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)

# ponytail: split마다 두 예제·마지막 한 위치만 평가; 더 큰 정답 데이터는 후속 범위입니다.
with torch.no_grad():
    for split, rows in splits.items():
        correct, total_loss = 0, 0.0
        assert rows
        for prompt, target in rows:
            tokens = prompt.split()
            assert 1 <= len(tokens) <= 4 and all(t in lm["to_id"] for t in tokens)
            assert target in ("failed", "healthy")
            ids = torch.tensor([[lm["to_id"][t] for t in tokens]], dtype=torch.long)
            logits, _ = model(ids)
            scores = logits[:, -1, :]  # (1, 7): 마지막 위치의 모든 어휘 후보
            assert scores.shape == (1, len(lm["vocab"])) and torch.isfinite(scores).all()
            label = torch.tensor([lm["to_id"][target]], dtype=torch.long)
            loss = loss_fn(scores, label)
            assert torch.isfinite(loss)
            guess = lm["vocab"][scores.argmax(-1).item()]
            correct += int(guess == target)
            total_loss += loss.item()
            print(f"{prompt} -> {guess} | 정답: {target} | loss: {loss.item():.4f}")
        print(f"[{split}] 정답: {correct}/{len(rows)} | 정확도: {correct/len(rows)*100:.1f}% | 평균 loss: {total_loss/len(rows):.4f}")
assert checkpoint.read_bytes() == original, "체크포인트가 변경됐습니다"
print("검사 통과: 입력 중복 없음·유한한 loss·체크포인트 불변. 재학습 없음")
PY
```

## 5. 코드와 텐서 크기

| 단계 | 의미와 크기 |
|---|---|
| `splits` | 결과를 보기 전에 정한 입력 문자열·정답 문자열 쌍. 이번 실행에서 학습하지 않음 |
| `set(prompts)` | 완전히 같은 입력 문자열의 중복만 검사. 의미적 중복·누출 전체를 검사하는 기능은 아님 |
| `lm["restore"](...)` | 이전 구조·어휘·저장 전후 logits 검사 후 eval 모드 모델 반환 |
| `ids` | `1×3`, 입력 한 문장의 토큰 ID 세 개 |
| `logits` | `1×3×7`, 입력의 각 위치마다 다음 토큰 후보 일곱 개의 점수 |
| `logits[:, -1, :]` | `1×7`, 문장 축을 유지하고 마지막 위치만 선택 |
| `label` | 길이 1의 정수 ID 텐서. 입력과 별개인 채점용 정답 |
| `scores.argmax(-1)` | 후보 일곱 개 중 가장 높은 점수의 ID |
| `loss.item()` | 텐서 loss를 Python 숫자로 꺼냄. 역전파나 업데이트가 아님 |
| `correct`, `total_loss` | 각 split 시작 때 0으로 초기화하는 합계 |

`model.eval()`은 복원 함수에서 적용하고 `no_grad()`는 평가 계산의 미분 그래프를 만들지 않습니다. optimizer·backward·step·save를 호출하지 않습니다. 이번에는 자신의 예측을 다음 입력으로 붙이지 않으며 eos까지 생성하지도 않습니다. **한 위치 다음 토큰 평가**와 **문장 전체 생성 평가**는 다른 대상입니다.

## 6. 실제 로컬 실행 예시와 해석

Python 3.12·CPU PyTorch 격리 환경에서 임시 VM형 폴더를 만들고 11장 코드로 별도 체크포인트를 재현 학습·저장한 뒤 이 평가를 실제 실행했습니다. 아래 평가 코드 자체는 재학습하지 않습니다. 사용자 VM 실행 완료를 뜻하지 않으며 환경·가중치에 따라 수치가 달라질 수 있습니다.

```text
gpu current status -> failed | 정답: failed | loss: 0.6282
network current status -> healthy | 정답: healthy | loss: 0.6501
[train] 정답: 2/2 | 정확도: 100.0% | 평균 loss: 0.6391
gpu status current -> failed | 정답: failed | loss: 0.7915
network status current -> failed | 정답: healthy | loss: 2.4366
[validation] 정답: 1/2 | 정확도: 50.0% | 평균 loss: 1.6141
current status gpu -> current | 정답: failed | loss: 1.9934
current status network -> current | 정답: healthy | loss: 1.9889
[test] 정답: 0/2 | 정확도: 0.0% | 평균 loss: 1.9912
검사 통과: 입력 중복 없음·유한한 loss·체크포인트 불변. 재학습 없음
```

1. **train 2/2:** 학습한 상태 예측 위치에서는 기존 정답을 재현합니다. 이전의 `8/8`과는 분모가 다릅니다.
2. **validation 1/2:** 같은 단어를 사용했어도 순서 변경 뒤 network의 상태를 틀렸습니다. 두 예제에서 한 개를 틀린 결과이지 전체 입력에 대한 50% 성능 추정이 아닙니다.
3. **test 0/2:** 두 입력 모두 `current`를 예측해 미리 정한 상태 정답과 다릅니다. **평가 과제에서 실패한 것이지 프로그램 실행이 실패한 것이 아닙니다.** 테스트 정답을 예측 출력에 맞춰 current로 바꾸면 평가 목적이 바뀝니다.
4. **loss:** 같은 후보·같은 마지막 상태 위치 평가에서 분리별 정답에 부여한 확률을 비교할 수 있습니다. 다만 입력 분포가 다르고 사례가 적으므로 일반적인 성능 차이로 확대하지 않습니다.

10~11장의 loss `0.1662`는 두 문장의 **모든 여덟 정답 위치 평균**입니다. 이번 train loss `0.6391`은 **상태 위치 두 개의 평균**입니다. 모델이 그대로여도 계산 대상이 달라 값이 다릅니다. 따라서 두 숫자의 차이를 학습 퇴보로 해석하지 않습니다.

두 학습 문장의 성공과 순서 변경 과제의 실패는 이 설정에서 재현과 전이의 차이를 보여 줍니다. 이 결과 하나로 모든 언어 이해·일반화 능력이 없다고 단정하거나, 실패 원인을 특정 한 층에만 돌리지는 않습니다. 어순별 학습 데이터, 위치 표현 등은 후속 통제 실험으로 구분해야 합니다.

## 7. 개발용 검사와 한계

[독립 검사](source/09-transformer/test_evaluation.py)는 PyTorch가 있는 개발 환경의 **저장소 루트**에서 실행합니다.

```bash
python source/09-transformer/test_evaluation.py
```

임시 VM 배치 폴더에서 실제 저장 프로세스와 평가 코드를 실행합니다. 입력·정답 쌍의 고정, 실제 모델 점수에서 Python 표준 라이브러리의 안정적인 logsumexp 수식으로 구한 loss와 출력의 일치, 정답 수·평균, eval/no_grad 결과, 모든 가중치와 체크포인트 불변을 확인합니다. 임시 모델 준비에는 재현 학습이 포함되지만 기존 VM·체크포인트는 건드리지 않습니다.

- split마다 두 예제뿐이며 미등록 어휘·새 기기·자연어 문장 전체를 대표하지 않습니다.
- 원래 두 학습 문장에서 진정한 독립 테스트를 뒤늦게 떼어낸 것이 아닙니다. 새 합성 형식에 대한 진단 평가입니다.
- 이번에는 후보 모델 하나가 고정되어 validation으로 모델을 선택하지 않습니다. 여러 모델 선택 절차의 완료를 주장하지 않습니다.
- 테스트 결과를 보고 보완하는 순간 이후의 비교는 개발 데이터 비교입니다. 독립 최종 평가용 입력은 따로 준비합니다.
- perplexity·자동 평가 프레임워크·배포 기준·데이터 보완·재학습은 추가하지 않습니다.
- 후속 실습에서는 테스트 점수만 올리기보다 학습 데이터의 표현 범위를 바꾼 모델을 같은 개발 기준으로 비교하고, 별도 최종 평가를 남기는 방법을 다룰 수 있습니다.

## 8. 복습

- [ ] 모델 출력을 보기 전에 입력과 정답을 정하는 이유를 설명한다.
- [ ] 실제 상태 정답과 이 합성 과제의 규칙을 구분한다.
- [ ] 후보 일곱 개 전체에서 argmax를 고르는 이유를 설명한다.
- [ ] 마지막 한 위치 loss와 모든 위치 평균 loss를 구분한다.
- [ ] 정확도 0%여도 평가 코드의 검사가 통과할 수 있는 이유를 설명한다.
- [ ] 문자열 중복 없음이 독립 테스트의 충분조건이 아님을 설명한다.
- [ ] validation 진단과 모델 선택, test 이후 보완과 독립 평가를 구분한다.

[← 이전: 새 입력 관찰](12-input-probes.md) · [목차](README.md)
