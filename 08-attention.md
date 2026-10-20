# 08 · Attention 입문 — 펼치기와 가중합

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)

## 목표와 진행 순서

세 토큰 모델은 `Embedding → Flatten → Linear`로 다음 토큰을 예측했습니다. 이번에는 여러 토큰의 벡터를 **가중합**하는 계산을 먼저 관찰합니다.

| 단계 | 내용 | 범위 |
|---|---|---|
| 1/3 | 같은 입력, 다른 가중치 | 직접 정한 비중의 가중합 |
| 2/3 | Query·Key 점수와 softmax | 내적·크기 조정·비중 계산 |
| 3/3 | Value 가중합과 고정 문맥 비교 | Q·K에서 계산한 비중으로 V 가중합 |

이번 코드는 **Transformer 전체 구현이나 모델 학습이 아닙니다.** 1/3에서는 비중을 직접 정하고, 2/3에서는 직접 정한 Query·Key에서 비중을 계산하며, 3/3에서는 그 비중으로 Value를 가중합합니다. PyTorch·GPU·체크포인트가 필요하지 않으며 파일을 생성하거나 수정하지 않습니다.

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

3/3에서는 이번에 계산한 비중으로 Value를 가중합하고, 기존 `Flatten → Linear` 모델과의 차이를 연결합니다.

## 7. 3/3단계 · 계산된 비중으로 Value 섞기

2/3의 Query·Key와 계산 방식을 그대로 유지하고 Value 가중합을 연결합니다. 이번에는 Key와 Value의 역할이 다름을 드러내기 위해 **Value를 앞 실습과 다른 교육용 숫자**로 정합니다. Key와 Value는 같은 토큰 순서에 대응해야 합니다.

| 토큰 | Key: 비교용 | Value: 섞을 정보 |
|---|---|---|
| gpu | `[1.0, 0.0]` | `[10.0, 0.0]` |
| current | `[0.0, 1.0]` | `[0.0, 10.0]` |
| status | `[1.0, 1.0]` | `[1.0, 1.0]` |

3/3 안에서는 Key·Value를 모두 고정하고 Query만 바꿉니다. Q·K·V는 여전히 학습된 벡터가 아닙니다. 이 코드는 Query 한 개의 scaled dot-product Attention 출력을 계산하며, 두 Query를 각각 독립적으로 실행해 비교합니다. 여러 head·전체 위치 계산·causal mask·학습은 구현하지 않습니다. 계산 근거는 *Attention Is All You Need*, §3.2.1입니다.

### 전체 명령

같은 **Ubuntu VM 터미널**, `~/llm-lab/source`의 기존 가상환경에서 실행합니다. Python 표준 라이브러리만 사용하며 파일 저장·수정이나 모델 복원이 없습니다. 이전 실습을 파일로 저장할 필요도 없습니다.

**소스:** [03-attention-output.py](source/08-attention/03-attention-output.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import exp, isclose, sqrt

# 교육용 Q·K·V입니다. Key는 비교용, Value는 섞을 정보입니다.
tokens = ["gpu", "current", "status"]
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
values = [[10.0, 0.0], [0.0, 10.0], [1.0, 1.0]]
cases = [
    ("Q1", [1.0, 0.0], [4.412233019477645, 2.378870239084068]),
    ("Q2", [0.0, 1.0], [2.378870239084068, 4.412233019477645]),
]
assert len(tokens) == len(keys) == len(values)
assert all(len(vector) == 2 for vector in keys + values)
print("토큰 순서:", tokens)
print("Value:", values)
for label, query, expected in cases:
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert isclose(sum(weights), 1.0)
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert len(output) == len(values[0])
    assert all(isclose(actual, target) for actual, target in zip(output, expected, strict=True))
    print(f"{label}: {query}")
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])

print("검사 통과: 같은 Key·Value도 Query에 따라 출력 벡터가 달라집니다")
PY
```

### 코드와 계산 해설

`scores → scaled → exps → weights`는 2/3과 같습니다. 새로 연결하는 부분은 `output`이며, 계산된 비중과 Value를 토큰 순서대로 묶고 각 좌표를 가중합합니다.

```text
Q1의 출력 첫 좌표 ≈ 0.4011×10 + 0.1978×0  + 0.4011×1 ≈ 4.4122
Q1의 출력 둘째 좌표 ≈ 0.4011×0  + 0.1978×10 + 0.4011×1 ≈ 2.3789
```

위 비중은 표시를 위해 반올림한 값이므로 이를 직접 곱하면 끝자리 차이가 납니다. 코드는 반올림 전 비중으로 계산합니다. `expected`는 별도 고정밀 계산에서 구한 검사 기준이고, 출력 계산에는 사용하지 않습니다.

Key 세 개에서 비중 세 개가 나오지만, Value가 각각 두 숫자이므로 Query 하나의 출력도 두 숫자입니다. **출력 벡터의 좌표 합이 1일 필요는 없습니다.** 비중 합이 1인 것과, 그 비중으로 합친 정보 벡터를 구분합니다. 이 출력은 다음 토큰 예측 확률이 아닙니다.

### 계산 실행 예시

아래는 코드를 Python으로 실행해 확인한 산술 출력 예시입니다. 모델 학습이나 다음 단어 예측 결과가 아닙니다.

```text
토큰 순서: ['gpu', 'current', 'status']
Value: [[10.0, 0.0], [0.0, 10.0], [1.0, 1.0]]
Q1: [1.0, 0.0]
  비중: [0.4011, 0.1978, 0.4011]
  출력 벡터: [4.4122, 2.3789]
Q2: [0.0, 1.0]
  비중: [0.1978, 0.4011, 0.4011]
  출력 벡터: [2.3789, 4.4122]
검사 통과: 같은 Key·Value도 Query에 따라 출력 벡터가 달라집니다
```

### 기존 세 토큰 모델과 비교

| 항목 | 기존 세 토큰 모델 | 이번 산술 예제 |
|---|---|---|
| 입력 정보 | 마지막 세 토큰의 8차원 임베딩 | 직접 정한 Q·K·V |
| 정보 조합 | 임베딩 3개를 길이24로 펼친 뒤 Linear | Q·K에서 비중을 계산해 V를 가중합 |
| 조합 계수 | 추론 중 고정된 Linear 파라미터 | Q·K 입력에 따라 바뀌는 혼합 비중 |
| 출력 | 어휘별 다음 토큰 점수 | Query 하나당 2차원 정보 벡터 |
| 이번 실행에서 학습 | 기존 저장 모델의 학습과 별개 | 없음 |

실제 Attention 모델의 **학습된 파라미터도 추론 중에는 고정**됩니다. 달라지는 것은 입력에서 계산되는 Attention 비중입니다. 기존 모델도 입력 임베딩이 바뀌면 출력이 달라지므로, 단순히 “출력이 변한다”만으로 두 구조의 차이를 설명하지 않습니다.

Attention으로 바꾼다고 입력에서 잘린 과거 토큰이 저절로 보이지는 않습니다. 제공한 토큰 범위와 정보를 조합하는 방식은 별도입니다. 다음 토큰 예측 모델로 만들려면 이 정보 벡터를 어휘 점수로 바꾸는 출력층 등 추가 구성이 필요합니다.

### 복습과 이후 연결

- [ ] Q·K는 비중 계산용, V는 섞을 정보임을 구분한다.
- [ ] 같은 K·V에서 Q만 바꿨는데 출력이 달라진 과정을 설명한다.
- [ ] 비중 합1과 출력 벡터의 좌표 합을 구분한다.
- [ ] 현재 예제가 학습된 Transformer나 문장 생성기가 아님을 설명한다.

이번 3/3의 출력까지 확인하면 입문 산술 흐름을 마칩니다. 다음 절에서는 입력에서 Q·K·V를 만드는 선형 변환 계산을 연결합니다. 변환 계수의 실제 학습은 아직 진행하지 않습니다.

## 8. 후속 실습 · 입력에서 Q·K·V 만들기

### 목표와 개념

앞에서는 Q·K·V를 직접 적었습니다. 이번에는 **입력 X와 변환 계수만 정하고 Q·K·V를 계산**합니다. 새로운 부분은 Attention 계산 앞에 붙습니다.

```text
입력 X → 서로 다른 선형 변환 → Q·K·V
                                  ↓
                       앞서 배운 비중 계산 → V 가중합
```

Self-Attention에서는 같은 입력으로부터 서로 다른 변환을 사용해 Q·K·V를 만듭니다. 행벡터 표기로 `Q = X Wq`, `K = X Wk`, `V = X Wv`입니다. 실제 모델에서는 이 변환 행렬이 학습되는 파라미터입니다. 근거: *Attention Is All You Need*, §3.2.2–3.2.3, arXiv:1706.03762. 여기서는 **직접 정한 작은 행렬**로 계산만 확인합니다. 학습된 파라미터라고 해석하지 않습니다.

이 실습은 Q·K·V 생성까지만 실행합니다. softmax·Value 가중합 재연결, loss·역전파·모델 저장은 포함하지 않습니다. 기존 세 토큰 모델의 `Embedding(…, 8) → Flatten → Linear(24, …)`를 수정하거나 체크포인트를 읽지 않습니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source`의 가상환경에서 실행합니다. Python 3.10 이상과 표준 라이브러리만으로 실행되며 추가 패키지·GPU가 필요하지 않습니다. 파일을 생성하거나 수정하지 않습니다.

`cd`는 실습 폴더로 이동하고, `source`는 기존 가상환경을 활성화합니다. `python -`는 `PY`까지의 코드를 읽어 실행합니다. 이전 코드의 변수는 필요하지 않습니다.

**소스:** [04-qkv-projection.py](source/08-attention/04-qkv-projection.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
# 교육용 입력과 변환 계수입니다. 학습된 값이 아닙니다.
tokens = ["gpu", "current", "status"]
inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
wq = [[1.0, 0.0], [1.0, 1.0]]
wk = [[0.0, 1.0], [1.0, 0.0]]
wv = [[2.0, 0.0], [0.0, 3.0]]


def project(x, matrix):
    # ponytail: 2차원 행벡터 전용; 차원 확장은 PyTorch 행렬곱으로 전환합니다.
    return [x[0] * matrix[0][column] + x[1] * matrix[1][column]
            for column in range(2)]


queries = [project(x, wq) for x in inputs]
keys = [project(x, wk) for x in inputs]
values = [project(x, wv) for x in inputs]
assert queries == [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
assert keys == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
assert values == [[2.0, 0.0], [0.0, 3.0], [2.0, 3.0]]
for token, x, q, k, v in zip(tokens, inputs, queries, keys, values, strict=True):
    print(f"{token}: X={x} -> Q={q}, K={k}, V={v}")
print("검사 통과: 같은 입력에서 서로 다른 선형 변환으로 Q·K·V를 만들었습니다")
PY
```

### 코드와 차원 해설

입력은 토큰 세 개 × 숫자 두 개인 `3×2`입니다. 각 변환 행렬은 `2×2`이므로 Q·K·V도 각각 `3×2`입니다. **토큰마다 다른 행렬을 주지 않습니다.** 모든 토큰에 같은 `wq`를 적용하고, K·V도 각자의 행렬을 모든 토큰에 공유합니다.

`project(x, matrix)`는 행벡터 한 개에 행렬 하나를 곱합니다. 출력의 `column`번째 숫자는 입력의 두 좌표와 행렬의 해당 열을 곱해 더한 값입니다. 행렬곱이지, 같은 위치끼리만 곱하는 연산이 아닙니다.

예를 들어 `status`의 X는 `[1, 1]`이고 `wq`는 `[[1, 0], [1, 1]]`입니다.

```text
Q 첫 좌표 = 1×1 + 1×1 = 2
Q 둘째 좌표 = 1×0 + 1×1 = 1
따라서 Q = [2, 1]
```

- `inputs`: 사람이 정한 교육용 입력입니다. 저장 모델에서 꺼낸 임베딩이 아닙니다.
- `wq`, `wk`, `wv`: 서로 다른 역할의 변환 계수입니다. Attention 비중이나 확률이 아닙니다.
- `queries`, `keys`, `values`: 입력을 변환해서 얻은 결과입니다. 토큰 이름은 계산에 사용하지 않습니다.
- 세 `assert`: 별도로 적은 계산 기준과 결과를 비교합니다. 결과 생성에 정답을 사용하지 않습니다.
- `project`: 이 예제의 고정 2차원 전용 함수입니다. 범용 행렬 연산기로 사용하지 않습니다.

이번 계산은 bias를 더하지 않는 선형 변환입니다. 앞 모델의 `nn.Linear`와 연결해 볼 때는 bias 유무와 가중치 저장 방향까지 구분해야 하며, 이 리스트를 그대로 PyTorch의 weight에 복사하는 실습은 아닙니다.

### 계산 실행 예시

아래는 해당 코드를 Python으로 실행해 확인한 산술 출력입니다. 모델 학습·추론 결과가 아닙니다.

```text
gpu: X=[1.0, 0.0] -> Q=[1.0, 0.0], K=[0.0, 1.0], V=[2.0, 0.0]
current: X=[0.0, 1.0] -> Q=[1.0, 1.0], K=[1.0, 0.0], V=[0.0, 3.0]
status: X=[1.0, 1.0] -> Q=[2.0, 1.0], K=[1.0, 1.0], V=[2.0, 3.0]
검사 통과: 같은 입력에서 서로 다른 선형 변환으로 Q·K·V를 만들었습니다
```

앞 절의 Q·K·V와 숫자가 같을 필요는 없습니다. 이번에는 새로 지정한 변환 행렬로 계산했습니다. **입력 벡터 X, 변환 계수 W, 계산 결과 Q·K·V, 이후 Q·K에서 계산할 Attention 비중은 서로 다릅니다.** 현재 출력에 비중이나 다음 토큰 확률은 없습니다.

### 복습과 다음 연결

- [ ] `status`의 Q `[2, 1]`을 손으로 계산한다.
- [ ] 같은 X에서도 서로 다른 변환으로 Q·K·V가 만들어지는 이유를 설명한다.
- [ ] 토큰별로 따로 행렬을 정하지 않고 같은 변환을 공유함을 구분한다.
- [ ] 이번에 계산한 Q·K·V를 학습된 값이나 Attention 비중으로 부르지 않는다.

다음 절에서는 이렇게 만든 Q·K·V를 앞서 배운 내적·softmax·가중합에 연결합니다. 실제 파라미터 학습·여러 head·causal mask·Transformer 전체 구현은 후속 주제입니다.

## 9. 후속 실습 · 입력 X에서 Attention 출력까지

### 목표와 계산 범위

8절의 입력과 변환 행렬을 그대로 유지하고, 7절의 내적·크기 조정·softmax·Value 가중합을 연결합니다. 같은 입력 X에서 Q·K·V를 만들고 각 토큰의 Q로 같은 입력의 모든 K·V를 참고하는 **Self-Attention의 한 head 계산**입니다. 근거: *Attention Is All You Need*, §3.2.1–3.2.3, arXiv:1706.03762.

```text
X (3×2) → Q·K·V (각 3×2)
Q 하나와 K 세 개의 내적 → sqrt(2)로 나누기 → softmax 비중 세 개
비중 세 개로 V 세 개 가중합 → 출력 벡터 하나 (숫자 두 개)
위 계산을 Q 세 개에 반복 → 전체 출력 (3×2)
```

행벡터 표기와 bias 없는 변환을 유지합니다. 이전 3/3의 직접 지정 Q·K·V가 아니라 **바로 앞 8절에서 계산한 값**을 사용하므로 7절의 출력 숫자와는 다릅니다. 모든 위치를 계산하지만 causal mask는 없으므로 `gpu` 위치도 뒤의 `current`, `status`를 참고합니다. 다음 토큰 학습에 그대로 쓰는 causal 모델이 아니며, 위치 정보·여러 head·출력 투영·잔차 연결·정규화도 구현하지 않습니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**에서 기존 가상환경을 활성화한 뒤 실행합니다. Python 3.10 이상과 표준 라이브러리만 사용합니다. 아래 명령은 이전 실습 변수가 없어도 실행되며 파일·체크포인트를 읽거나 쓰지 않습니다.

**소스:** [05-self-attention.py](source/08-attention/05-self-attention.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import exp, isclose, sqrt

# 앞 실습과 같은 교육용 입력·계수입니다. 학습된 값이 아닙니다.
tokens = ["gpu", "current", "status"]
inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
wq = [[1.0, 0.0], [1.0, 1.0]]
wk = [[0.0, 1.0], [1.0, 0.0]]
wv = [[2.0, 0.0], [0.0, 3.0]]


def project(x, matrix):
    # ponytail: 2차원 행벡터 전용; 차원 확장은 PyTorch 행렬곱으로 전환합니다.
    return [x[0] * matrix[0][column] + x[1] * matrix[1][column]
            for column in range(2)]


queries = [project(x, wq) for x in inputs]
keys = [project(x, wk) for x in inputs]
values = [project(x, wv) for x in inputs]
assert queries == [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
assert keys == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
assert values == [[2.0, 0.0], [0.0, 3.0], [2.0, 3.0]]
# 별도 Decimal 고정밀 계산의 검사 기준이며, 출력 계산에는 사용하지 않습니다.
expected_scores = [[0.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
expected_outputs = [[1.1977758146404282, 2.4066725560787155],
                    [1.5034898434845539, 2.255234765226831],
                    [1.43200918051748, 2.579912264869866]]
print("참고할 토큰 순서:", tokens)
for token, query, score_check, output_check in zip(
        tokens, queries, expected_scores, expected_outputs, strict=True):
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    assert scores == score_check
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert len(weights) == len(tokens) and isclose(sum(weights), 1.0)
    assert all(0.0 < weight < 1.0 for weight in weights)
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert all(isclose(actual, target) for actual, target in zip(output, output_check, strict=True))
    print(f"{token}: Q={query}")
    print("  내적 점수:", scores)
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 입력 X에서 Q·K·V를 만들고 각 토큰의 Attention 출력을 계산했습니다")
PY
```

### 코드와 계산 해설

- `project`와 입력·행렬·QKV 검사는 8절과 같습니다. 독립 복사 실행을 위해 같은 코드를 포함하며 범용 행렬 라이브러리는 만들지 않습니다.
- 바깥 반복문은 `gpu`, `current`, `status`의 Q를 하나씩 선택합니다. 안쪽 `scores` 계산은 선택한 Q를 **모든 K**와 비교합니다. 토큰 자신의 K 하나만 비교하는 것이 아닙니다.
- `weights`의 순서는 언제나 `gpu`, `current`, `status`입니다. 각 Q에 대해 별도로 softmax를 계산하므로 비중 합도 각각 1입니다.
- `output`은 K가 아니라 **V를 가중합**합니다. `expected_scores`, `expected_outputs`는 검사 기준이며 계산 입력이 아닙니다. `round`도 출력 표시만 바꿉니다.

예를 들어 `gpu`의 Q는 `[1, 0]`이고 K는 `[0, 1]`, `[1, 0]`, `[1, 1]`입니다. 내적은 차례로 `[0, 1, 1]`입니다. 크기를 조정하고 softmax를 적용한 비중으로 V를 섞으면 다음과 같습니다. 아래 비중은 반올림되어 끝자리가 다를 수 있습니다.

```text
출력 첫 좌표 ≈ 0.1978×2 + 0.4011×0 + 0.4011×2 ≈ 1.1978
출력 둘째 좌표 ≈ 0.1978×0 + 0.4011×3 + 0.4011×3 ≈ 2.4067
```

### 로컬 산술 실행 예시

다음은 위 소스를 Python으로 실행하고 assert를 통과한 출력입니다. 교육용 상수 계산이며 파라미터 학습이나 다음 단어 예측 결과가 아닙니다.

```text
참고할 토큰 순서: ['gpu', 'current', 'status']
gpu: Q=[1.0, 0.0]
  내적 점수: [0.0, 1.0, 1.0]
  비중: [0.1978, 0.4011, 0.4011]
  출력 벡터: [1.1978, 2.4067]
current: Q=[1.0, 1.0]
  내적 점수: [1.0, 1.0, 2.0]
  비중: [0.2483, 0.2483, 0.5035]
  출력 벡터: [1.5035, 2.2552]
status: Q=[2.0, 1.0]
  내적 점수: [1.0, 2.0, 3.0]
  비중: [0.14, 0.284, 0.576]
  출력 벡터: [1.432, 2.5799]
검사 통과: 입력 X에서 Q·K·V를 만들고 각 토큰의 Attention 출력을 계산했습니다
```

### 복습과 다음 연결

- [ ] 입력·변환 행렬·QKV·비중·출력 벡터를 구분한다.
- [ ] `gpu`의 내적 점수 `[0, 1, 1]`을 계산한다.
- [ ] Q 하나당 비중 세 개와 출력 숫자 두 개가 나오는 이유를 설명한다.
- [ ] 비중은 토큰을 참고하는 비율이며 다음 토큰의 어휘 확률이 아님을 구분한다.
- [ ] mask가 없어서 뒤 토큰도 참고한다는 한계를 설명한다.

다음 연결 주제는 causal mask로 뒤 위치를 참고하지 못하게 제한하는 계산입니다. 현재 실습에서는 구현하지 않습니다.

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)
