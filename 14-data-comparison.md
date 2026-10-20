# 14 · 학습 표현을 늘리고 같은 기준으로 비교하기

[← 이전: 다음 토큰 평가](13-next-token-evaluation.md) · [목차](README.md) · [다음: 작은 모델 마무리 →](15-finish-tiny-model.md)

## 1. 목표와 선행 조건

13장에서는 기존 모델을 고정해 순서 변경 과제를 채점했습니다. 이번에는 **같은 원본에서 복사본 둘을 만들고 학습 표현만 다르게 추가 학습**합니다. 원본 모델·파일은 보존하고, 세 모델을 같은 입력과 정답으로 비교합니다.

11장의 `tiny_transformer_checkpoint.py`와 `tiny_transformer_v1.pt`, 기존 가상환경이 필요합니다. 모델 구조·어휘·CPU·float64는 그대로입니다. 기기→상태 규칙은 교육용 합성 정답이지 실제 장비 상태가 아닙니다.

## 2. 무엇을 같게 두고 무엇을 바꾸는가

| 모델 | 출발점 | 이번 추가 학습 입력 | 추가 갱신 |
|---|---|---|---|
| base | 기존 체크포인트 복원 | 없음 | 0회 |
| control | base의 독립 복사본 | 기존 형식 두 개 | 100회 |
| augmented | base의 또 다른 독립 복사본 | 기존 두 개 + 변경 형식 두 개 | 100회 |

두 후보 모두 **SGD·학습률0.1·momentum0·weight decay0·full-batch 평균 loss·마지막 위치 다음 토큰 정답**을 사용합니다. 후보 전체 파라미터를 갱신하며 층을 추가하거나 출력 후보를 failed/healthy 두 개로 제한하지 않습니다.

왜 control도 필요한가요? base와 augmented만 비교하면 **더 학습한 효과**와 **표현을 추가한 효과**가 섞입니다. control과 augmented는 같은 가중치에서 같은 목적함수와 갱신 수로 시작하므로 이번 데이터 구성 차이를 비교하기 좋습니다. 다만 배치가 두 개와 네 개로 다릅니다. 같은 갱신 수이지 같은 계산량·같은 예제 노출 수는 아니며, 평균 loss에서 기존/추가 형식의 가중 비율도 달라집니다. 보편적 인과 효과나 최적 설정을 입증하는 실험은 아닙니다.

> [!IMPORTANT]
> 10~11장은 문장 전체 여덟 위치의 다음 토큰을 학습했습니다. 이번에는 **입력 세 토큰 뒤 상태 한 위치만** 학습합니다. 같은 구조·가중치에서 시작하지만 목적함수가 달라집니다. 저장된 optimizer/RNG 상태를 복원한 정확한 학습 재개가 아니라 **새 optimizer를 사용한 과제별 추가 학습**입니다. 문장 생성이나 다른 위치의 성능 유지는 이번 검사 대상이 아닙니다.

## 3. 입력과 정답의 역할

| 그룹 | gpu 입력 예시 | control 학습 | augmented 학습 | 비교 역할 |
|---|---|---|---|---|
| original | `gpu current status` | 포함 | 포함 | 기존 표현의 상태 예측 유지 확인 |
| added | `gpu status current` | 미포함 | 포함 | augmented에는 학습 데이터 적합 검사 |
| dev_order | `current status gpu` | 미포함 | 미포함 | 이미 관찰한 별도 순서의 개발용 비교 |

각 그룹은 gpu→failed, network→healthy 한 개씩입니다. 13장의 validation을 added로, test를 dev_order로 다시 이름 붙입니다. 이미 본 결과로 보완 방향을 정했으므로 **이전 test를 독립 최종 평가라고 부르지 않습니다.** dev_order는 이번 gradient 계산에는 들어가지 않지만 개발에 참고한 데이터입니다. added의 개선은 학습 입력 재현이며 새로운 입력 일반화의 증거가 아닙니다. 새 독립 최종 평가 데이터의 구성·실행은 이번 범위가 아닙니다.

## 4. 실행 위치와 전체 명령

**Ubuntu VM 터미널**에서 실행합니다. `cd`는 폴더 이동, `source`는 기존 환경 활성화, `python -`는 PY 사이 코드의 즉시 실행입니다. 새 설치·GPU 사용은 없습니다. **복사본 두 개를 메모리에서 실제 학습하지만 코드 파일·체크포인트는 저장하거나 덮어쓰지 않습니다.** 프로세스가 끝나면 복사본도 사라집니다. 검사에 assert를 사용하므로 `python -O`는 사용하지 않습니다.

신뢰하는 본인 체크포인트만 읽습니다. 소스: [06-compare-data.py](source/09-transformer/06-compare-data.py). 이 소스를 직접 실행할 때도 현재 폴더는 저장소 루트가 아니라 위 파일들이 배치된 VM의 `~/llm-lab/source`여야 합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
"""같은 원본의 복사본 둘을 추가 학습해 데이터 표현 보완을 비교합니다."""
import copy
from pathlib import Path
import runpy
import torch

# 13장의 test도 결과를 이미 보았으므로 이제 모두 개발용입니다.
groups = {
    "original": [("gpu current status", "failed"), ("network current status", "healthy")],
    "added": [("gpu status current", "failed"), ("network status current", "healthy")],
    "dev_order": [("current status gpu", "failed"), ("current status network", "healthy")],
}
prompts = [prompt for rows in groups.values() for prompt, _ in rows]
assert len(prompts) == len(set(prompts))
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original_bytes = checkpoint.read_bytes()
base = lm["restore"](checkpoint)
original_weights = {k: v.clone() for k, v in base.state_dict().items()}
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)


def batch(rows):
    assert rows
    for prompt, target in rows:
        tokens = prompt.split()
        assert len(tokens) == 3 and all(t in lm["to_id"] for t in tokens)
        assert target in ("failed", "healthy")
    x = torch.tensor([[lm["to_id"][t] for t in prompt.split()] for prompt, _ in rows])
    y = torch.tensor([lm["to_id"][target] for _, target in rows])
    return x, y


def continue_training(rows):
    model = copy.deepcopy(base)  # 가중치를 공유하지 않고 매번 같은 출발점에서 시작합니다.
    model.train()
    x, y = batch(rows)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    # ponytail: 각 후보100회 full-batch·마지막 상태 위치만; 설정 탐색은 후속 범위입니다.
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits[:, -1, :], y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(v, original_weights[k]) for k, v in model.state_dict().items())
    return model.eval()


models = {
    "base": base,
    "control": continue_training(groups["original"]),
    "augmented": continue_training(groups["original"] + groups["added"]),
}
print("후보마다 같은 원본에서 SGD 100회·lr 0.1·마지막 상태 위치만 추가 학습")
with torch.no_grad():
    for name, model in models.items():
        for group, rows in groups.items():
            x, y = batch(rows)
            scores = model(x)[0][:, -1, :]
            assert scores.shape == (len(rows), len(lm["vocab"])) and torch.isfinite(scores).all()
            loss = loss_fn(scores, y)
            assert torch.isfinite(loss)
            guesses = scores.argmax(-1)
            correct = (guesses == y).sum().item()
            print(f"[{name}/{group}] 정답: {correct}/{len(rows)} | 평균 loss: {loss.item():.4f}")
            print("  " + " / ".join(f"{prompt} -> {lm['vocab'][guess]}" for (prompt, _), guess in zip(rows, guesses.tolist())))
assert all(torch.equal(v, original_weights[k]) for k, v in base.state_dict().items())
assert checkpoint.read_bytes() == original_bytes
print("검사 통과: 원본 가중치·체크포인트 불변. 복사본만 학습·저장 없음")
PY
```

## 5. 코드와 텐서 흐름

| 코드 | 역할 |
|---|---|
| `copy.deepcopy(base)` | 모델과 가중치를 독립 복사. `model = base`는 같은 객체에 이름만 붙여 원본까지 바꾸므로 사용하지 않음 |
| `original_weights` | 원본 state_dict 값의 clone. 마지막에 원본이 변하지 않았는지 비교 |
| `batch(rows)` | 문자열을 어휘 ID로 변환. 입력 x는 `B×3`, 정답 y는 `B` |
| `model(x)[0]` | 모든 입력 위치의 점수 `B×3×7`. 두 번째 반환값인 Attention은 이번 loss 계산에 직접 사용하지 않음 |
| `logits[:, -1, :]` | 마지막 위치의 모든 어휘 점수 `B×7`. 정답 상태는 입력에 넣지 않고 채점·loss에만 사용 |
| `CrossEntropyLoss` | softmax를 미리 붙이지 않은 logits와 정수 정답 ID로 배치 평균 loss 계산 |
| `zero_grad → backward → step` | 이전 기울기 초기화 → 현재 기울기 계산 → 복사본 가중치 갱신 |
| `model.eval()` / `no_grad()` | 평가 모드와 평가 중 미분 그래프 비활성화. 학습 루프 바깥에서 사용 |
| `checkpoint.read_bytes()` | 실행 전후 원본 파일 바이트 비교. 복사본을 저장하는 기능이 아님 |

control을 만든 뒤에도 base는 그대로입니다. augmented는 control에서 이어 학습하는 것이 아니라 다시 base를 복사합니다. 복원 함수는 같은 구조·어휘와 저장 전후 점수도 확인합니다.

## 6. 실제 로컬 실행 예시

Python 3.12·CPU PyTorch 격리 환경의 임시 VM형 폴더에서 11장 코드로 별도 원본을 재현 학습한 뒤, 위 코드를 실제 실행했습니다. 검사 준비용 재현 학습과 본문의 복사본 추가 학습은 서로 다른 작업입니다. 아래는 실행 예시이며 독자의 VM 완료를 뜻하지 않습니다. 패키지·원본 가중치·계산 환경에 따라 결과가 달라질 수 있습니다.

```text
후보마다 같은 원본에서 SGD 100회·lr 0.1·마지막 상태 위치만 추가 학습
[base/original] 정답: 2/2 | 평균 loss: 0.6391
  gpu current status -> failed / network current status -> healthy
[base/added] 정답: 1/2 | 평균 loss: 1.6141
  gpu status current -> failed / network status current -> failed
[base/dev_order] 정답: 0/2 | 평균 loss: 1.9912
  current status gpu -> current / current status network -> current
[control/original] 정답: 2/2 | 평균 loss: 0.0068
  gpu current status -> failed / network current status -> healthy
[control/added] 정답: 1/2 | 평균 loss: 0.8290
  gpu status current -> failed / network status current -> failed
[control/dev_order] 정답: 1/2 | 평균 loss: 1.0814
  current status gpu -> failed / current status network -> failed
[augmented/original] 정답: 2/2 | 평균 loss: 0.0084
  gpu current status -> failed / network current status -> healthy
[augmented/added] 정답: 2/2 | 평균 loss: 0.0099
  gpu status current -> failed / network status current -> healthy
[augmented/dev_order] 정답: 2/2 | 평균 loss: 0.3215
  current status gpu -> failed / current status network -> healthy
검사 통과: 원본 가중치·체크포인트 불변. 복사본만 학습·저장 없음
```

| 그룹 | base 정답 / loss | control 정답 / loss | augmented 정답 / loss |
|---|---|---|---|
| original | 2/2 / 0.6391 | 2/2 / 0.0068 | 2/2 / 0.0084 |
| added | 1/2 / 1.6141 | 1/2 / 0.8290 | 2/2 / 0.0099 |
| dev_order | 0/2 / 1.9912 | 1/2 / 1.0814 | 2/2 / 0.3215 |

- **원본 보존:** base 점수는 13장과 같습니다. 마지막 검사도 메모리 원본·파일 보존을 확인했습니다.
- **더 학습한 효과:** control도 dev_order에서 0/2→1/2로 바뀌었습니다. base와 augmented 차이 전부를 데이터 추가 때문이라고 말할 수 없습니다.
- **데이터 구성 비교:** 이 설정에서는 augmented가 control보다 added와 dev_order에서 높은 정답 수·낮은 평균 loss를 보였습니다. added는 augmented의 학습 입력이므로 성공을 일반화라고 부르지 않습니다.
- **정확도와 loss:** original은 두 후보 모두 2/2여도 loss가 다릅니다. 세 그룹을 합친 점수로 학습 적합과 개발용 순서 비교를 섞지 않고 그룹별로 읽습니다.
- **한계:** dev_order 두 개의 성공은 임의의 단어 순서·새 기기·자연어·실제 장애 판단 성능을 입증하지 않습니다. 기존 모든 위치 loss0.1662와 직접 비교하거나 문장 생성 능력이 유지됐다고 주장하지 않습니다.

출력이 다른 경우 결과를 맞추려고 정답·반복 수를 바꾸지 말고 먼저 원본 체크포인트와 환경을 확인합니다. 검사를 통과했어도 정답률이 반드시 좋아야 하는 것은 아닙니다.

## 7. 개발용 검사와 다음 경계

PyTorch가 있는 **개발 환경의 저장소 루트**에서 실행합니다. 학습 VM에서 추가로 실행할 필수 명령이 아닙니다.

```bash
python source/09-transformer/test_data_comparison.py
```

[검사 코드](source/09-transformer/test_data_comparison.py)는 임시 폴더에서 실제 원본을 만들고 비교를 실행합니다. 데이터·정답 고정, 두 후보의 독립 저장 공간, 원본 가중치/파일 보존을 확인하고, 선언한 100회 마지막 위치 SGD를 별도로 재계산해 모든 후보 가중치를 비교합니다. 실제 점수에서 표준 라이브러리 logsumexp로 그룹별 평균 loss·정답 수를 교차 검사합니다. 운영 모델·외부 서비스·넓은 일반화를 검사하는 테스트는 아닙니다.

모델 등록·새 체크포인트·학습률 탐색·LoRA·서빙·새 독립 최종 평가를 추가하지 않습니다. 다음 확장에서 저장이나 최종 평가를 다룬다면, 원본과 구분되는 파일명과 개발 과정에 사용하지 않은 평가 설계를 먼저 정합니다.

공식 참고: [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD), [저장·복원과 상태 복사](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

## 8. 복습

- [ ] base와 복사본 둘이 필요한 이유를 설명한다.
- [ ] 기존 전체 위치 학습과 이번 마지막 위치 추가 학습을 구분한다.
- [ ] 같은 갱신 수와 같은 계산량의 차이를 설명한다.
- [ ] added와 dev_order가 독립 최종 평가가 아닌 이유를 설명한다.
- [ ] 정답률이 같아도 loss가 다른 이유를 설명한다.
- [ ] 저장하지 않은 학습 가중치는 프로세스 종료 후 사라짐을 설명한다.

[← 이전: 다음 토큰 평가](13-next-token-evaluation.md) · [목차](README.md) · [다음: 작은 모델 마무리 →](15-finish-tiny-model.md)
