# 15 · 작은 모델 실습 마무리: 추가 학습부터 저장·복원까지

[← 이전: 학습 표현 보완](14-data-comparison.md) · [목차](README.md) · [다음: 사전학습 LLM →](16-pretrained-inference.md)

## 1. 이번에 완성할 결과물

**기존 원본을 보존하면서 14장의 보완 후보를 재현 학습하고 → 기존 문장 생성과 비교하고 → 별도 파일에 저장하고 → 새 Python 프로세스에서 복원·검사**합니다. 이 전체 흐름을 하나의 실습으로 실행합니다.

14장의 복사본은 프로세스 종료로 사라졌습니다. 이번에는 기존 체크포인트에서 같은 네 입력·마지막 상태 위치 loss·SGD100회·학습률0.1로 다시 만듭니다. 사라진 프로세스나 optimizer 상태를 복구하는 것이 아닙니다. 원본 구조·어휘·CPU·float64는 그대로이며 기존 모델 정의·복원·생성 함수를 재사용합니다.

| 파일 | 역할 | 이번 동작 |
|---|---|---|
| `source/tiny_transformer_checkpoint.py` | 11장 구조·복원·생성 코드 | 읽기만 함 |
| `tiny_transformer_v1.pt` | 기존 원본 | 읽기만 함, 불변 검사 |
| `source/tiny_transformer_finish.py` | 이번 통합 실행 코드 | 새 파일로 작성 |
| `tiny_transformer_augmented_v1.pt` | 보완 후보와 복원 검사용 값 | 새 파일로만 저장 |

이 저장 파일은 **추론 재현용 교육 산출물**입니다. optimizer/RNG 상태가 없으므로 임의 시점의 정확한 학습 재개용이 아니고, 아래 생성 회귀 때문에 운영용 개선 모델로 승인한 것도 아닙니다.

## 2. Ubuntu VM에서 한 번에 실행

기존 `~/llm-lab/source/.venv`와 11장 파일들이 필요합니다. 새 설치나 GPU 사용은 없습니다. `cat`은 코드를 새 파일에 기록하고, `(set -C; ...)`는 이미 있는 코드 파일의 덮어쓰기를 거부합니다. 두 Python 명령은 별도 프로세스입니다. `save`만 실제 추가 학습하고 `check`는 재학습하지 않습니다.

기존 파일이 있다면 내용을 확인하고 자동 삭제하지 않습니다. 학습 중 중단되면 새 체크포인트가 불완전할 수 있습니다. 이 경우 원본은 유지되며 실패 파일을 확인한 뒤 별도로 처리합니다. 검사에 assert를 사용하므로 `python -O`는 사용하지 않고, 본인이 만든 신뢰하는 체크포인트만 읽습니다.

소스: [07-finish-model.py](source/09-transformer/07-finish-model.py). 저장소에서 직접 실행할 때도 아래 VM형 파일 배치와 현재 폴더가 필요합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

(set -C; cat > tiny_transformer_finish.py <<'PY'
"""14장 보완 후보를 재현하고 생성 비교·별도 저장·새 프로세스 복원을 확인합니다."""
import argparse
import copy
from pathlib import Path
import runpy
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["save", "check"])
args = parser.parse_args()
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
original = Path("../tiny_transformer_v1.pt")
checkpoint = Path("../tiny_transformer_augmented_v1.pt")
if args.mode == "save" and (checkpoint.exists() or checkpoint.is_symlink()):
    raise FileExistsError(f"기존 파일 보호: {checkpoint}")
original_bytes = original.read_bytes()


def generations(model, name):
    results = {}
    for start in ("gpu", "network"):
        tokens, reason = lm["generate"](model, [start])
        results[start] = (tokens, reason)
        print(f"[{name}] {' '.join(tokens)} | 종료: {reason}")
    return results


if args.mode == "save":
    base = lm["restore"](original)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model = copy.deepcopy(base).train()
    rows = [("gpu current status", "failed"), ("network current status", "healthy"),
            ("gpu status current", "failed"), ("network status current", "healthy")]
    x = torch.tensor([[lm["to_id"][t] for t in prompt.split()] for prompt, _ in rows])
    y = torch.tensor([lm["to_id"][target] for _, target in rows])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    # xb: 기존 파일을 거부하고 학습 전에 새 파일만 확보합니다.
    # ponytail: 중단 시 새 파일이 불완전할 수 있음; 운영 저장은 원자적 교체로 확장합니다.
    with checkpoint.open("xb") as stream:
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x)[0][:, -1, :], y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
        with torch.no_grad():
            scores = model(x)[0][:, -1, :]
            print(f"보완 학습 입력 상태 정답: {(scores.argmax(-1) == y).sum().item()}/4 | loss: {loss_fn(scores, y).item():.4f}")
            probes = torch.tensor([[lm["to_id"][t] for t in s.split()] for s in
                                  ("gpu current status failed", "network current status healthy")])
            reference_logits = model(probes)[0]
        assert torch.isfinite(reference_logits).all()
        generations(base, "base")
        generated = generations(model, "augmented")
        assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())
        assert original.read_bytes() == original_bytes, "원본 파일 변경"
        torch.save({"architecture": lm["architecture"], "vocab": lm["vocab"],
                    "state_dict": model.state_dict(), "reference_logits": reference_logits,
                    "generation": generated}, stream)
    print(f"별도 저장 완료: {checkpoint.name} | 원본 가중치·파일 불변")
else:
    saved_bytes = checkpoint.read_bytes()
    model = lm["restore"](checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actual = generations(model, "restored")
    assert actual == payload["generation"], "저장 전·복원 후 생성 결과 불일치"
    assert checkpoint.read_bytes() == saved_bytes, "보완 파일 변경"
    assert original.read_bytes() == original_bytes, "원본 파일 변경"
    print("복원 검사 통과: logits·생성 결과 일치·재학습 없음·원본 파일 불변·보완 파일 불변")
PY
) && python tiny_transformer_finish.py save && python tiny_transformer_finish.py check
```

`&&`는 앞 명령이 성공했을 때만 다음 명령을 실행합니다. 코드 파일이나 보완 체크포인트가 이미 있으면 덮어쓰지 않고 중단합니다. 저장이 이미 완료된 뒤 복원만 다시 검사하려면 다음 명령을 사용합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python tiny_transformer_finish.py check
```

## 3. 핵심 코드 해설

- **재현 학습:** 기존 모델을 `deepcopy`하고 14장 augmented와 같은 네 입력의 마지막 상태 위치만 학습합니다. 입력은 `4×3`, logits는 `4×3×7`, loss에는 마지막 점수 `4×7`과 정답 ID `4`를 넣습니다. 원본과 공유하지 않는 복사본의 전체 파라미터를 갱신합니다.
- **생성 비교:** gpu/network 한 토큰에서 출발해 자신의 예측을 붙이는 기존 `generate`를 호출합니다. 정답 문장을 입력으로 주는 teacher-forcing 평가가 아닙니다. 최대 새 토큰4개·기존 eos/context/limit 규칙을 유지합니다.
- **저장:** 구조 ID·어휘 순서·가중치와 복원 점검용 `reference_logits`, 실제 생성 결과를 별도 파일에 넣습니다. reference_logits는 기존 restore와 같은 두 개의 네 토큰 점검 입력에서 계산합니다. 이 값은 정답 점수가 아니라 저장 시점의 점수입니다.
- **복원:** 기존 `restore`가 같은 구조·어휘·유한한 가중치·저장 전후 점수를 검사합니다. 그 다음 생성 함수를 실제로 다시 실행해 저장 전 결과와 비교합니다. 저장된 문자열을 답으로 출력하는 방식이 아닙니다.
- **검사 통과의 뜻:** 저장 전 모델을 재현했다는 뜻입니다. 문장 품질이나 과제 성능까지 좋다는 뜻이 아닙니다. 저장된 생성 결과는 재현 검사 기준이지 정답 문장이 아닙니다.

## 4. 실제 로컬 실행 예시와 중요한 결과

Python3.12·CPU PyTorch 격리 환경의 임시 VM형 폴더에서 원본을 재현 학습해 준비한 뒤, 위 `save`와 `check`를 별도 프로세스로 실제 실행했습니다. 독자 VM 완료를 의미하지 않으며 환경·원본 가중치에 따라 결과가 달라질 수 있습니다.

```text
보완 학습 입력 상태 정답: 4/4 | loss: 0.0092
[base] gpu current status failed <eos> | 종료: eos
[base] network current status healthy <eos> | 종료: eos
[augmented] gpu current status failed <eos> | 종료: eos
[augmented] network healthy failed failed <eos> | 종료: eos
별도 저장 완료: tiny_transformer_augmented_v1.pt | 원본 가중치·파일 불변
[restored] gpu current status failed <eos> | 종료: eos
[restored] network healthy failed failed <eos> | 종료: eos
복원 검사 통과: logits·생성 결과 일치·재학습 없음·원본 파일 불변·보완 파일 불변
```

**상태 위치의 4/4와 문장 생성 유지 실패가 동시에 나타났습니다.** network를 시작으로 생성하면 원본은 `current status healthy`를 이어 가지만 보완 후보는 `healthy failed failed`를 만듭니다. 보완 후보의 첫 토큰부터 예측이 달라졌습니다. 이번 loss는 세 토큰 뒤 상태만 점검했고, network만 주었을 때 current를 예측하는 위치나 문장 전체 생성을 유지하도록 직접 학습하지 않았습니다. 공유 가중치의 갱신이 다른 입력의 출력에도 영향을 준 결과입니다.

이 예제에서 관찰한 것은 **과제별 추가 학습 뒤 기존 생성의 회귀**입니다. 원인을 한 층으로 특정하거나 모든 추가 학습이 반드시 같은 문제를 만든다고 일반화하지 않습니다. eos로 끝났다는 사실도 문장 내용이 맞다는 뜻은 아닙니다.

복원된 모델이 같은 잘못된 문장을 생성하는 것은 저장·복원이 그 모델을 재현했다는 증거입니다. 품질을 고쳤다는 증거가 아닙니다. 목표에 맞춰 상태 평가와 생성 회귀 검사를 함께 남기는 이유를 여기서 확인합니다. 상태 입력 네 개는 학습 적합 검사이며 독립 최종 평가가 아닙니다.

## 5. 검증과 마무리

PyTorch가 있는 개발 환경의 **저장소 루트**에서 다음 [독립 검사](source/09-transformer/test_finish_model.py)를 실행할 수 있습니다. VM 학습자에게 추가로 요구하는 필수 단계는 아닙니다.

```bash
python source/09-transformer/test_finish_model.py
```

실제 save/check 별도 프로세스, 14장 augmented와 전체 가중치 일치, 원본·보완 파일 불변, 재저장 거부, 저장된 생성 기준을 바꿨을 때 불일치 탐지를 확인합니다. 모델 품질을 개선하는 자동 수정기는 아닙니다.

**작은 모델을 직접 만드는 기본 흐름은 여기서 하나의 결과물로 정리합니다.** 원본·보완 후보·실행 코드와 재현 검사가 남습니다. 확인된 생성 회귀는 숨기지 않고 한계로 유지합니다. 다음 [16장](16-pretrained-inference.md)은 사전학습 모델 불러오기·토큰화·응답 생성을 다룹니다. 추가 학습 비교는 그 뒤 범위이며 이 장은 새 패키지 설치나 사전학습 모델 다운로드를 수행하지 않습니다.

- [ ] 상태 한 위치 성공과 문장 전체 생성 성공을 구분한다.
- [ ] 저장·복원 성공과 모델 품질 개선을 구분한다.
- [ ] 원본을 보존하고 보완 후보를 별도 파일로 복원할 수 있다.

[← 이전: 학습 표현 보완](14-data-comparison.md) · [목차](README.md) · [다음: 사전학습 LLM →](16-pretrained-inference.md)
