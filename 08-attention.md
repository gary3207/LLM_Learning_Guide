# 08 · Attention 입문 — 펼치기와 가중합

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)

## 목표와 진행 순서

세 토큰 모델은 `Embedding → Flatten → Linear`로 다음 토큰을 예측했습니다. 이번에는 여러 토큰의 벡터를 **가중합**하는 계산을 먼저 관찰합니다.

| 단계 | 내용 | 범위 |
|---|---|---|
| 1/3 | 같은 입력, 다른 가중치 | 이 장의 실행 실습 |
| 2/3 | Query·Key 점수와 softmax | 후속 실습 예정 |
| 3/3 | Value 가중합과 고정 문맥 비교 | 후속 실습 예정 |

이번 코드는 **Attention 전체 구현이나 모델 학습이 아닙니다.** 가중치를 직접 정해 가중합의 의미만 살펴봅니다. PyTorch·GPU·체크포인트가 필요하지 않으며 파일을 생성하거나 수정하지 않습니다.

## 1. 펼치기와 가중합의 차이

교육용 토큰 세 개에 다음 벡터를 임의로 정합니다. 학습된 임베딩도, 실제 단어 의미를 나타낸 숫자도 아닙니다.

| 토큰 | 벡터 |
|---|---|
| gpu | `[1.0, 0.0]` |
| current | `[0.0, 1.0]` |
| status | `[1.0, 1.0]` |

- **펼치기:** 세 벡터를 이어 붙여 숫자 여섯 개로 만듭니다. 이 연산 자체는 숫자를 섞지 않습니다. 기존 모델에서는 뒤의 `Linear`가 학습된 계수로 이를 조합했습니다.
- **가중합:** 각 벡터에 비중을 곱하고 같은 좌표끼리 더해 숫자 두 개로 만듭니다.

예를 들어 비중 `[0.8, 0.1, 0.1]`을 주면 다음처럼 계산합니다.

```text
첫 좌표 = 0.8 × 1.0 + 0.1 × 0.0 + 0.1 × 1.0 = 0.9
둘째 좌표 = 0.8 × 0.0 + 0.1 × 1.0 + 0.1 × 1.0 = 0.2
```

Attention은 Query와 Key의 관계로 비중을 계산하고 Value 벡터를 가중합합니다. 이번에는 **그 비중을 계산하는 부분을 생략**하고 직접 지정합니다. 근거: *Attention Is All You Need*, §3.2–3.2.1, arXiv:1706.03762.

## 2. 실행 위치와 전체 명령

**Ubuntu VM 터미널**에서 실행합니다. 기존 실습 폴더 `~/llm-lab/source`와 `.venv`를 사용합니다. Python 3.10 이상이 필요하며 기존 실습의 Python 3.12에서도 실행할 수 있습니다. 새 패키지는 설치하지 않습니다.

- `cd`: 기존 실습 폴더로 이동합니다.
- `source .venv/bin/activate`: 기존 Python 가상환경을 활성화합니다.
- `python -`: 아래 `PY`까지의 코드를 표준 입력으로 받아 실행합니다. Python 파일로 저장하지 않습니다.
- `python`을 찾을 수 없으면 먼저 가상환경 활성화 오류를 확인합니다. 이 오류만으로 환경을 새로 만들거나 패키지를 다시 설치하지 않습니다.

**소스:** [01-weighted-sum.py](source/08-attention/01-weighted-sum.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import isclose

# 교육용 숫자입니다. 학습된 임베딩이나 단어의 실제 의미가 아닙니다.
tokens = ["gpu", "current", "status"]
values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
flattened = [number for vector in values for number in vector]
assert flattened == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
print("토큰:", tokens)
print("펼친 입력:", flattened)

# ponytail: 가중치를 직접 지정; 다음 단계에서 Query·Key로 계산합니다.
cases = [
    ("gpu 쪽 비중", [0.8, 0.1, 0.1], [0.9, 0.2]),
    ("current 쪽 비중", [0.1, 0.8, 0.1], [0.2, 0.9]),
]
for label, weights, expected in cases:
    assert len(weights) == len(values) and isclose(sum(weights), 1.0)
    assert all(weight >= 0 for weight in weights)
    weighted = [sum(weight * vector[column] for weight, vector in zip(weights, values, strict=True))
                for column in range(len(values[0]))]
    assert len(weighted) == 2
    assert all(isclose(actual, target) for actual, target in zip(weighted, expected, strict=True))
    print(f"{label}: {weights} -> {[round(number, 3) for number in weighted]}")

print("검사 통과: 같은 입력도 가중치에 따라 가중합이 달라집니다")
PY
```

## 3. 코드 해설

| 코드 | 역할 |
|---|---|
| `values` | 토큰마다 두 숫자를 가진 벡터 목록 |
| `flattened` | 벡터 세 개를 순서대로 연결한 길이 6의 입력 |
| `cases` | 가중치 두 가지와 각 계산의 검사 기준 |
| `zip(weights, values, strict=True)` | 비중과 벡터를 하나씩 묶음; 개수가 다르면 오류 |
| `vector[column]` | 벡터의 첫째 또는 둘째 좌표 선택 |
| `sum(weight * ...)` | 선택한 좌표에 비중을 곱해 합산 |
| `isclose` | 부동소수점의 작은 오차를 허용한 수치 비교 |
| `assert` | 가중치 합·계산 결과·출력 차원 검사; 틀리면 중단 |

`expected`는 계산을 끝낸 후 비교하는 검사 기준입니다. 가중합을 계산할 때 정답을 가져다 쓰지 않습니다. `round`는 출력 표시만 정리하며 실제 계산에는 사용하지 않습니다.

## 4. 실행 예시와 해석

다음은 위 코드를 Python으로 실행해 확인한 출력 예시입니다. 모델 학습·추론의 결과가 아니라 직접 지정한 숫자의 계산 결과입니다.

```text
토큰: ['gpu', 'current', 'status']
펼친 입력: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
gpu 쪽 비중: [0.8, 0.1, 0.1] -> [0.9, 0.2]
current 쪽 비중: [0.1, 0.8, 0.1] -> [0.2, 0.9]
검사 통과: 같은 입력도 가중치에 따라 가중합이 달라집니다
```

**입력 벡터는 같고 비중만 달라졌는데 결과가 달라집니다.** 이것이 이번 단계에서 확인하는 전부입니다. `gpu 쪽 비중`은 사람이 붙인 실험 이름이지, 모델이 GPU의 중요성을 알아냈다는 뜻이 아닙니다.

## 5. 한계와 복습

- 직접 정한 가중치는 학습된 Attention 가중치가 아닙니다.
- Attention이 입력에서 잘려 나간 토큰까지 복원하는 것은 아닙니다. 어떤 토큰을 입력으로 제공하는지는 별도 문제입니다.
- 이번 가중합은 벡터 하나만 계산합니다. Transformer의 모든 위치·여러 head·위치 정보·causal mask는 구현하지 않습니다.
- 단어 예측·loss 계산·역전파·저장 모델 변경은 없습니다.

- [ ] 펼치기는 숫자 여섯 개, 이번 가중합은 숫자 두 개인 이유를 설명한다.
- [ ] 첫 사례에서 `[0.9, 0.2]`가 나오는 계산을 따라간다.
- [ ] 가중치가 달라지는 것과 입력 자체가 달라지는 것을 구분한다.
- [ ] 사람이 지정한 비중과 Query·Key로 계산한 비중을 구분한다.

다음 단계에서는 **비중을 직접 쓰는 대신 Query·Key 점수를 softmax로 바꾸는 방법**을 배웁니다. 이번 코드의 출력과 위 항목을 확인한 뒤 연결합니다.

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)
