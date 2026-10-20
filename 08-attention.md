# 08 · Attention 입문 — 펼치기와 가중합

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)

## 목표와 진행 순서

세 토큰 모델은 `Embedding → Flatten → Linear`로 다음 토큰을 예측했습니다. 이번에는 여러 토큰의 벡터를 **가중합**하는 계산을 먼저 관찰합니다.

| 단계 | 내용 | 범위 |
|---|---|---|
| 1/3 | 같은 입력, 다른 가중치 | 직접 정한 비중의 가중합 |
| 2/3 | Query·Key 점수와 softmax | 내적·크기 조정·비중 계산 |
| 3/3 | Value 가중합과 고정 문맥 비교 | 후속 실습 예정 |

이번 코드는 **Attention 전체 구현이나 모델 학습이 아닙니다.** 1/3에서는 비중을 직접 정하고, 2/3에서는 직접 정한 Query·Key에서 비중을 계산합니다. PyTorch·GPU·체크포인트가 필요하지 않으며 파일을 생성하거나 수정하지 않습니다.

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

다음 2/3단계에서는 **비중을 직접 쓰는 대신 Query·Key 점수를 softmax로 바꾸는 방법**을 배웁니다.

## 6. 2/3단계 · Query·Key에서 비중 계산하기

1/3에서는 비중 `[0.8, 0.1, 0.1]`을 직접 정했습니다. 이제는 Query 한 개를 Key 세 개와 비교하여 비중 세 개를 계산합니다.

- **Query(Q):** 어떤 정보를 참고할지 비교할 때 사용하는 벡터입니다.
- **Key(K):** 각 토큰을 Query와 비교하기 위한 벡터입니다.
- **내적:** 같은 좌표끼리 곱한 뒤 더합니다. 이번 방식에서는 큰 내적일수록 큰 비중을 받습니다.
- **크기 조정:** 내적 점수를 Key 차원의 제곱근으로 나눕니다. 여기서는 차원이 2이므로 `sqrt(2)`입니다.
- **softmax:** 각 점수의 지수함수 값을 전체 합으로 나누어, 합이 1인 비중을 만듭니다.

이것은 *Attention Is All You Need*, §3.2.1의 scaled dot-product 계산 순서입니다. 실제 Transformer에서는 학습되는 선형 변환으로 Q·K·V를 만들지만, 여기서는 계산 흐름에 집중하려고 Q·K를 직접 정합니다. 기존 모델을 Attention 모델로 바꾸거나 학습하지 않습니다.

Key는 앞서 Value에 쓴 숫자와 같게 정하되, **역할은 다릅니다.** Key는 비중을 정하는 비교용이고 Value는 그 비중으로 섞을 정보입니다. 이번 단계에서는 Value를 아직 섞지 않습니다.

### 전체 명령

이전과 같은 **Ubuntu VM 터미널**에서 실행합니다. Python 표준 라이브러리만 사용하며 파일 저장·수정이 없습니다. 이전 코드를 파일로 저장할 필요도 없습니다.

**소스:** [02-query-key.py](source/08-attention/02-query-key.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import exp, isclose, sqrt

# 교육용 Q·K를 직접 지정합니다. 학습된 벡터가 아닙니다.
tokens = ["gpu", "current", "status"]
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
cases = [
    ("Q1", [1.0, 0.0], [1.0, 0.0, 1.0], 0, 1),
    ("Q2", [0.0, 1.0], [0.0, 1.0, 1.0], 1, 0),
]
print("Key 순서:", tokens)
for label, query, expected_scores, higher, lower in cases:
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    assert scores == expected_scores
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert len(weights) == len(keys)
    assert all(0.0 < weight < 1.0 for weight in weights)
    assert isclose(sum(weights), 1.0)
    assert weights[higher] > weights[lower]
    assert isclose(weights[higher], weights[2])
    assert isclose(weights[higher] / weights[lower], exp(1.0 / sqrt(2.0)))
    print(f"{label}: {query}")
    print("  내적 점수:", scores)
    print("  크기 조정:", [round(number, 4) for number in scaled])
    print("  softmax 비중:", [round(number, 4) for number in weights])
    print(f"  비중 합: {sum(weights):.4f}")

print("검사 통과: 같은 Key라도 Query에 따라 비중이 달라집니다")
PY
```

### 코드 해설

`keys`는 세 토큰 각각의 2차원 Key입니다. 반복문 한 번은 Query 하나를 Key 세 개와 비교하므로 내적 점수도 세 개, 비중도 세 개입니다. Q1과 Q2 사이에 바꾸는 계산 입력은 Query뿐입니다.

`expected_scores`는 내적 계산 후 확인하는 정답이고, `higher`와 `lower`는 비중 대소를 검사할 위치입니다. 비중 계산에는 사용하지 않습니다.

`exp`는 지수함수입니다. `scaled`의 최댓값을 모든 점수에서 똑같이 빼면 softmax의 수학적 결과는 같지만 큰 양수의 지수 계산에서 발생할 수 있는 오버플로를 피할 수 있습니다. `round`는 표시만 반올림합니다.

Q1과 `status`의 내적은 `1×1 + 0×1 = 1`입니다. Q1과 `gpu`의 내적도 `1×1 + 0×0 = 1`이므로 둘의 비중은 같습니다. **토큰 이름이 아니라 지정한 벡터가 계산 결과를 정합니다.**

### 계산 실행 예시

다음은 코드를 Python으로 실행해 확인한 산술 출력 예시입니다. 모델이 학습한 중요도나 다음 단어 예측 확률이 아닙니다.

```text
Key 순서: ['gpu', 'current', 'status']
Q1: [1.0, 0.0]
  내적 점수: [1.0, 0.0, 1.0]
  크기 조정: [0.7071, 0.0, 0.7071]
  softmax 비중: [0.4011, 0.1978, 0.4011]
  비중 합: 1.0000
Q2: [0.0, 1.0]
  내적 점수: [0.0, 1.0, 1.0]
  크기 조정: [0.0, 0.7071, 0.7071]
  softmax 비중: [0.1978, 0.4011, 0.4011]
  비중 합: 1.0000
검사 통과: 같은 Key라도 Query에 따라 비중이 달라집니다
```

점수 0도 softmax 뒤에는 비중이 0이 아닙니다. softmax는 가장 큰 항목 하나만 고르는 `argmax`가 아닙니다. 이 예시에서는 둘 이상의 토큰이 같은 큰 비중을 받습니다.

### 확인 질문과 다음 단계

- [ ] Key를 고정하고 Query만 바꿨을 때 무엇이 달라졌는가?
- [ ] Q1에서 `gpu`와 `status`의 비중이 같은 이유는 무엇인가?
- [ ] 내적 점수 0이 비중 0을 뜻하지 않는 이유는 무엇인가?
- [ ] Q·K를 사람이 정한 것과, 계산된 비중을 사람이 정한 것을 구분하는가?

3/3에서는 이번에 계산한 비중으로 Value를 가중합하고, 기존 `Flatten → Linear` 모델과의 차이를 연결합니다. 여러 head·전체 위치 계산·causal mask·학습은 아직 구현하지 않습니다.

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)
