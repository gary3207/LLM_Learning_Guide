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

다음 절에서는 causal mask로 뒤 위치를 참고하지 못하게 제한하는 계산을 연결합니다.

## 10. 후속 실습 · causal mask로 뒤 토큰 가리기

### 목표와 개념

9절에서는 모든 위치가 모든 토큰을 참고했습니다. 이번에는 **입력 X·변환 행렬·QKV는 그대로 두고**, 위치별로 참고할 수 있는 범위만 제한합니다. 자신의 위치와 앞 위치는 허용하고, 뒤 위치는 차단합니다.

| Q 위치 | gpu의 K·V | current의 K·V | status의 K·V |
|---|---|---|---|
| gpu | 허용 | 차단 | 차단 |
| current | 허용 | 허용 | 차단 |
| status | 허용 | 허용 | 허용 |

다음 토큰을 예측하는 모델을 여러 위치에서 학습할 때는 뒤에 있는 정답 토큰을 미리 참고하면 안 됩니다. causal mask는 이 방향의 정보 유출을 막는 장치입니다. **현재 위치는 허용**합니다. 현재 입력 토큰을 보고 그 다음 토큰을 예측하는 것이므로, 정답을 입력 위치보다 한 칸 뒤로 맞추는 작업과 mask는 구분해야 합니다. 이번에는 정답·loss·학습을 연결하지 않고 mask의 산술 계산만 확인합니다.

근거: *Attention Is All You Need*, §3.2.3, arXiv:1706.03762. 논문처럼 softmax **전**에 허용되지 않은 점수를 `-∞`로 바꿉니다. 코드의 `float("-inf")`가 음의 무한대입니다. 지수 계산에서 이 항목은 0이 되므로 최종 비중도 0입니다. 점수를 단순히 0으로 바꾸면 `exp(0)=1`이어서 차단되지 않습니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source`의 가상환경에서 실행합니다. `cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 `PY`까지의 코드 실행입니다. Python 3.10 이상과 표준 라이브러리만 사용하며 이전 변수·저장 소스·모델 파일이 필요하지 않습니다. 파일을 읽거나 쓰지 않고 학습도 하지 않습니다.

**소스:** [06-causal-mask.py](source/08-attention/06-causal-mask.py)

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
# 앞쪽 토큰만으로 별도 계산한 검사 기준입니다. 출력 계산에는 사용하지 않습니다.
expected_weights = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                    [0.140029245043378, 0.28399540974126, 0.575975345215362]]
expected_outputs = [[2.0, 0.0], [1.0, 1.5], [1.43200918051748, 2.579912264869866]]
print("참고할 토큰 순서:", tokens)
for position, (token, query) in enumerate(zip(tokens, queries, strict=True)):
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [score / sqrt(len(query)) for score in scores]
    masked = [score if key_position <= position else float("-inf")
              for key_position, score in enumerate(scaled)]
    exps = [exp(score - max(masked)) for score in masked]
    weights = [number / sum(exps) for number in exps]
    assert isclose(sum(weights), 1.0)
    assert all(weight == 0.0 for weight in weights[position + 1:])
    assert all(weight > 0.0 for weight in weights[:position + 1])
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(weights, expected_weights[position], strict=True))
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(output, expected_outputs[position], strict=True))
    print(f"{token}: 허용={tokens[:position + 1]}")
    print("  mask 적용 점수:", [round(number, 4) for number in masked])
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 각 위치는 자신과 앞 토큰만 참고하고 뒤 토큰의 비중은 0입니다")
PY
```

### 코드 해설

- `enumerate`는 0부터 시작하는 위치 번호와 항목을 함께 줍니다. `position`은 Q의 위치이고 `key_position`은 비교하는 K의 위치입니다.
- `key_position <= position`이면 자신 또는 앞 토큰입니다. 이때 기존 점수를 유지하고, 뒤 토큰의 점수만 `-inf`로 바꿉니다. Q·K·V 자체는 바꾸지 않습니다.
- `exps`와 `weights`는 mask가 적용된 점수로 계산합니다. 최댓값을 빼는 안정화 방식은 이전 실습과 같습니다. 이 예제는 자기 위치를 항상 허용하므로 한 행 전체가 차단되지는 않습니다.
- `weights[position + 1:]`는 뒤 토큰들의 비중입니다. 이들이 정확히 0인지 검사합니다. 허용 범위의 양수 비중 검사도 포함하며, 앞 실습의 “모든 비중이 양수” 검사를 그대로 쓰지 않습니다.
- 비중은 여전히 세 개이고 합은 1입니다. 차단 항목은 0이지만 위치를 삭제하지 않습니다. Value 가중합 방식과 출력 차원도 그대로입니다.
- `expected_weights`, `expected_outputs`는 허용된 앞쪽 토큰만으로 별도 계산한 검사 기준입니다. 출력 계산이나 학습 목표로 사용하지 않습니다.

### 로컬 산술 실행 예시와 비교

아래는 정본 코드를 Python으로 실행하여 assert를 통과한 결과입니다. 파라미터 학습이나 모델의 단어 예측 결과가 아닙니다.

```text
참고할 토큰 순서: ['gpu', 'current', 'status']
gpu: 허용=['gpu']
  mask 적용 점수: [0.0, -inf, -inf]
  비중: [1.0, 0.0, 0.0]
  출력 벡터: [2.0, 0.0]
current: 허용=['gpu', 'current']
  mask 적용 점수: [0.7071, 0.7071, -inf]
  비중: [0.5, 0.5, 0.0]
  출력 벡터: [1.0, 1.5]
status: 허용=['gpu', 'current', 'status']
  mask 적용 점수: [0.7071, 1.4142, 2.1213]
  비중: [0.14, 0.284, 0.576]
  출력 벡터: [1.432, 2.5799]
검사 통과: 각 위치는 자신과 앞 토큰만 참고하고 뒤 토큰의 비중은 0입니다
```

| 위치 | mask 없는 9절 출력 | causal mask 적용 출력 | 이유 |
|---|---|---|---|
| gpu | `[1.1978, 2.4067]` | `[2.0, 0.0]` | 자기 V만 참고 |
| current | `[1.5035, 2.2552]` | `[1.0, 1.5]` | gpu·current의 허용 점수가 같아 반씩 혼합 |
| status | `[1.432, 2.5799]` | `[1.432, 2.5799]` | 마지막 위치이므로 가릴 뒤 토큰 없음 |

이 비교에서 바꾼 것은 **참고 허용 범위뿐**입니다. 마지막 위치의 출력이 같은 것은 mask가 동작하지 않아서가 아니라, 그 위치에서 차단할 항목이 없기 때문입니다.

### 한계와 복습

- [ ] 자기 위치까지 허용하는 이유를 설명한다.
- [ ] 차단 점수를 0이 아니라 `-inf`로 바꾸는 이유를 설명한다.
- [ ] 차단된 위치를 포함한 비중 세 개의 합이 1임을 구분한다.
- [ ] `status`의 출력이 앞 실습과 같은 이유를 설명한다.
- [ ] mask만 추가한 계산과 학습된 문장 생성기를 구분한다.

이것은 고정된 교육용 입력의 단일 Attention 계산입니다. 이 절에서는 padding mask·batch·여러 head·출력 투영·위치 표현·학습은 구현하지 않습니다. 토큰 내용이 아니라 위치 번호로 허용 범위를 정하며, 다음 절의 위치 표현을 벡터에 더하는 실습과는 다릅니다.

## 11. 후속 실습 · 토큰 벡터에 위치 정보 더하기

### 목표와 앞 실습 연결

causal mask는 **어디까지 참고할 수 있는가**를 정했습니다. 이번에는 **같은 토큰이 몇 번째 위치에 놓였는가**를 입력 벡터에 반영하는 덧셈만 확인합니다. 두 장치는 역할이 다릅니다. 위치 벡터를 더한다고 미래 토큰 참조가 차단되지는 않습니다.

토큰별 같은 벡터를 조회하면 같은 `gpu`는 어디에 놓여도 같은 벡터를 얻습니다. 이전 QKV 변환도 모든 토큰에 같은 행렬을 적용하므로 같은 입력에서는 같은 QKV가 나옵니다. 다만 causal mask가 허용하는 문맥은 위치마다 다르므로 **위치 표현이 없으면 Attention 출력도 항상 같다**고 단정하면 안 됩니다.

원래 Transformer는 입력 임베딩과 같은 차원의 위치 표현을 더합니다. 근거: *Attention Is All You Need*, §3.5, arXiv:1706.03762. 여기서는 논문의 sin·cos 공식을 구현하지 않고 **직접 정한 위치 벡터 표**로 덧셈 원리만 봅니다. 학습된 임베딩이나 위치를 학습한 모델이 아닙니다.

같은 토큰을 두 위치에서 비교하려고 입력 순서를 `gpu current status`에서 **`gpu current gpu`**로 바꿉니다. `gpu`·`current`의 교육용 토큰 벡터는 그대로입니다. 이 절은 10절과 Attention 출력을 비교하는 실험이 아니라, 반복된 토큰에 위치 정보를 넣는 독립 계산입니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**의 기존 `~/llm-lab/source`에서 실행합니다. Python 3.10 이상만 필요하며 패키지·GPU·모델 파일이 필요하지 않습니다. 파일 읽기·쓰기와 학습은 없습니다. `cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 마지막 `PY`까지의 코드 실행입니다.

**소스:** [07-position-input.py](source/08-attention/07-position-input.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
# 교육용 토큰·위치 벡터입니다. 학습된 임베딩이 아닙니다.
# 같은 토큰을 비교하려고 앞 실습의 문장을 gpu current gpu로 바꿉니다.
tokens = ["gpu", "current", "gpu"]
token_vectors = {"gpu": [1.0, 0.0], "current": [0.0, 1.0]}
# ponytail: 위치 세 개 전용 수동 표; 실제 모델에서는 위치 표현 방식을 별도로 정합니다.
position_vectors = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
inputs = [token_vectors[token][:] for token in tokens]
combined = [[x + p for x, p in zip(vector, position, strict=True)]
            for vector, position in zip(inputs, position_vectors, strict=True)]

assert inputs == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
assert combined == [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]
assert inputs[0] == inputs[2]
assert combined[0] != combined[2]
assert all(len(vector) == 2 for vector in combined)
for index, (token, vector, position, result) in enumerate(
        zip(tokens, inputs, position_vectors, combined, strict=True)):
    print(f"위치 {index}: {token}")
    print(f"  토큰 벡터: {vector}, 위치 벡터: {position}")
    print("  합친 입력:", result)
print("검사 통과: 같은 gpu 벡터도 위치 벡터를 더하면 입력이 달라집니다")
PY
```

### 코드와 차원 해설

- `tokens`는 토큰의 순서이고, `token_vectors`는 토큰 이름으로 벡터를 조회하는 사전입니다. 두 `gpu`는 같은 `[1, 0]`을 사용합니다. `[:]`는 원래 표를 바꾸지 않도록 리스트를 복사합니다.
- `position_vectors`는 토큰 종류가 아니라 **위치 0·1·2**에 대응합니다. 첫 벡터를 `[0, 0]`으로 둔 것은 이 예시의 선택일 뿐 필수 규칙이 아닙니다.
- 바깥 `zip`은 각 토큰 벡터와 해당 위치 벡터를 짝짓고, 안쪽 `zip`은 같은 좌표끼리 짝지어 더합니다. `strict=True`는 짝의 개수가 다르면 오류를 냅니다.
- `combined`는 이어 붙이기가 아닙니다. 토큰·위치 벡터가 각각 숫자 두 개이므로 **합친 결과도 두 개**, 전체 모양은 `3×2`입니다. 숫자 네 개가 되지 않습니다.
- `assert`의 고정 숫자는 직접 계산한 검사 기준이고, `combined` 계산에 정답을 넣지 않습니다. 원래 두 `gpu` 입력이 같고 합친 입력은 다른지도 확인합니다.

```text
위치 0의 gpu: [1, 0] + [0, 0] = [1, 0]
위치 2의 gpu: [1, 0] + [0, 2] = [1, 2]
```

위치 벡터는 단어 의미나 Attention 비중이 아닙니다. 이 단순한 표는 덧셈을 드러내기 위한 예시이며, 일반적인 단어·위치 조합이 언제나 서로 다른 벡터가 된다는 보장도 아닙니다.

### 로컬 산술 실행 예시

아래는 정본 코드를 Python으로 실행해 assert를 통과한 출력입니다. 다음 토큰 예측·학습 결과가 아닙니다.

```text
위치 0: gpu
  토큰 벡터: [1.0, 0.0], 위치 벡터: [0.0, 0.0]
  합친 입력: [1.0, 0.0]
위치 1: current
  토큰 벡터: [0.0, 1.0], 위치 벡터: [0.0, 1.0]
  합친 입력: [0.0, 2.0]
위치 2: gpu
  토큰 벡터: [1.0, 0.0], 위치 벡터: [0.0, 2.0]
  합친 입력: [1.0, 2.0]
검사 통과: 같은 gpu 벡터도 위치 벡터를 더하면 입력이 달라집니다
```

### 한계·복습과 다음 연결

- [ ] 같은 `gpu`의 토큰 벡터는 같지만 합친 입력은 다른 이유를 설명한다.
- [ ] 벡터 덧셈과 이어 붙이기를 구분한다.
- [ ] 위치 표현과 causal mask의 역할을 구분한다.
- [ ] 이 수동 표를 학습된 위치 표현이나 논문의 sin·cos 공식으로 부르지 않는다.

이 절의 코드는 위치 세 개의 덧셈만 계산합니다. QKV·Attention 출력 재계산·sin·cos·위치 파라미터 학습·모델 저장은 포함하지 않습니다. 다음 절에서는 합친 입력에서 QKV와 causal Attention 출력을 계산합니다.

## 12. 후속 실습 · 위치 포함 입력에서 causal Attention 출력까지

### 목표와 계산 흐름

11절의 **`gpu current gpu`·토큰 벡터·위치 벡터를 그대로 유지**하고, 10절에서 사용한 변환 행렬과 causal mask 계산을 연결합니다. 새 위치 표현 방식을 만들거나 계수를 학습하는 실습이 아닙니다.

```text
토큰 벡터 + 위치 벡터 → combined (3×2)
combined → 같은 변환 행렬 Wq·Wk·Wv → Q·K·V (각 3×2)
각 Q와 모든 K의 내적 → sqrt(2)로 나누기 → 미래 위치 -inf
softmax → V 가중합 → 각 위치 출력 (숫자 두 개)
```

Q·K·V는 원래 토큰 벡터 `inputs`가 아니라 **위치 정보를 더한 `combined`**에서 만듭니다. 위치 표현은 입력을 바꾸고, causal mask는 참조 범위를 제한합니다. 둘 중 하나가 다른 하나를 대신하지 않습니다. 계산 근거: *Attention Is All You Need*, §3.2.1–3.2.3·§3.5, arXiv:1706.03762. 수동 위치 표는 논문의 sin·cos 구현이 아닙니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source` 가상환경에서 실행합니다. `cd`는 폴더 이동, `source`는 가상환경 활성화, `python -`는 마지막 `PY`까지의 코드 실행입니다. Python 3.10 이상과 표준 라이브러리만 필요합니다. 앞 실습의 변수가 없어도 실행되며 패키지 설치·GPU·파일 읽기·쓰기·학습은 없습니다.

**소스:** [08-position-attention.py](source/08-attention/08-position-attention.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import exp, isclose, sqrt

# 앞 실습과 같은 교육용 토큰·위치 벡터입니다. 학습된 값이 아닙니다.
tokens = ["gpu", "current", "gpu"]
token_vectors = {"gpu": [1.0, 0.0], "current": [0.0, 1.0]}
position_vectors = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
inputs = [token_vectors[token][:] for token in tokens]
combined = [[x + p for x, p in zip(vector, position, strict=True)]
            for vector, position in zip(inputs, position_vectors, strict=True)]
assert combined == [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]
# 이전 QKV 실습과 같은 변환 행렬입니다.
wq = [[1.0, 0.0], [1.0, 1.0]]
wk = [[0.0, 1.0], [1.0, 0.0]]
wv = [[2.0, 0.0], [0.0, 3.0]]


def project(x, matrix):
    # ponytail: 2차원 행벡터 전용; 차원 확장은 PyTorch 행렬곱으로 전환합니다.
    return [x[0] * matrix[0][column] + x[1] * matrix[1][column]
            for column in range(2)]


queries = [project(x, wq) for x in combined]
keys = [project(x, wk) for x in combined]
values = [project(x, wv) for x in combined]
assert queries == [[1.0, 0.0], [2.0, 2.0], [3.0, 2.0]]
assert keys == [[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
assert values == [[2.0, 0.0], [0.0, 6.0], [2.0, 6.0]]
# 허용된 앞쪽 토큰만의 독립 Decimal 계산 기준이며 출력 계산에는 쓰지 않습니다.
expected_weights = [[1.0, 0.0, 0.0], [0.1955703174930431, 0.8044296825069569, 0.0],
                    [0.011427238405401788, 0.19333548885002997, 0.7952372727445682]]
expected_outputs = [[2.0, 0.0], [0.3911406349860862, 4.826578095041741],
                    [1.61332902229994, 5.931436569567589]]
print("참고할 위치·토큰:", list(enumerate(tokens)))
for position, (token, query) in enumerate(zip(tokens, queries, strict=True)):
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [score / sqrt(len(query)) for score in scores]
    masked = [score if key_position <= position else float("-inf")
              for key_position, score in enumerate(scaled)]
    exps = [exp(score - max(masked)) for score in masked]
    weights = [number / sum(exps) for number in exps]
    assert isclose(sum(weights), 1.0)
    assert all(weight == 0.0 for weight in weights[position + 1:])
    assert all(weight > 0.0 for weight in weights[:position + 1])
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(weights, expected_weights[position], strict=True))
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(output, expected_outputs[position], strict=True))
    print(f"위치 {position}: {token}, 합친 입력={combined[position]}")
    print(f"  Q={query}, K={keys[position]}, V={values[position]}")
    print("  내적 점수:", scores)
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 위치 포함 입력에서 Q·K·V와 causal Attention 출력을 계산했습니다")
PY
```

### 코드와 차원 해설

- `combined`까지는 11절 계산입니다. `project`와 세 행렬, 내적·softmax·mask·가중합은 앞 실습 방식을 재사용합니다. 복사 실행을 위해 코드를 함께 두며 이전 파일을 불러오지 않습니다.
- 같은 토큰도 위치를 더한 입력이 달라져 이번 예제에서는 Q·K·V가 달라집니다. 세 행렬은 토큰·위치마다 새로 정하지 않고 모든 위치에서 공유합니다.
- 출력의 `Q=…, K=…, V=…`는 **해당 위치 자체의** QKV입니다. 하지만 그 아래 내적 점수·비중 세 개는 선택한 Q를 위치 0·1·2의 **모든 K**와 비교한 결과입니다. 두 `gpu`를 같은 항목으로 합치지 않습니다.
- 미래 위치의 내적 점수는 표시되지만 softmax 전 `-inf`로 가립니다. 가려진 위치는 비중 0이라 V 가중합에 참여하지 않습니다.
- 비중 세 개의 합은 반올림 전 1입니다. 출력은 V를 섞은 2차원 정보 벡터이며 좌표 합이 1일 필요가 없습니다. 전체 출력 모양도 `3×2`입니다.
- `expected_weights`, `expected_outputs`는 허용된 앞쪽 토큰만으로 별도 고정밀 계산한 검사 기준입니다. 학습 정답이나 출력 생성용 값이 아닙니다.

예를 들어 위치 2의 `gpu`는 합친 입력 `[1, 2]`에서 `Q=[3, 2]`, `K=[2, 1]`, `V=[2, 6]`을 얻습니다. 그 Q와 전체 K의 내적은 `[2, 6, 8]`입니다. 마지막 위치는 세 위치를 모두 허용하므로, 비중 세 개로 V 세 개를 가중합합니다.

### 로컬 산술 실행 예시

아래는 정본 소스를 로컬 Python으로 실행해 assert를 통과한 출력입니다. 학습·다음 단어 예측 결과가 아닙니다.

```text
참고할 위치·토큰: [(0, 'gpu'), (1, 'current'), (2, 'gpu')]
위치 0: gpu, 합친 입력=[1.0, 0.0]
  Q=[1.0, 0.0], K=[0.0, 1.0], V=[2.0, 0.0]
  내적 점수: [0.0, 2.0, 2.0]
  비중: [1.0, 0.0, 0.0]
  출력 벡터: [2.0, 0.0]
위치 1: current, 합친 입력=[0.0, 2.0]
  Q=[2.0, 2.0], K=[2.0, 0.0], V=[0.0, 6.0]
  내적 점수: [2.0, 4.0, 6.0]
  비중: [0.1956, 0.8044, 0.0]
  출력 벡터: [0.3911, 4.8266]
위치 2: gpu, 합친 입력=[1.0, 2.0]
  Q=[3.0, 2.0], K=[2.0, 1.0], V=[2.0, 6.0]
  내적 점수: [2.0, 6.0, 8.0]
  비중: [0.0114, 0.1933, 0.7952]
  출력 벡터: [1.6133, 5.9314]
검사 통과: 위치 포함 입력에서 Q·K·V와 causal Attention 출력을 계산했습니다
```

두 `gpu`의 출력은 다르지만 **위치 표현 때문만이라고 해석하지 않습니다.** 위치 0과 2는 허용되는 문맥도 다릅니다. 또한 10절과는 마지막 토큰 자체도 다르므로 10절 수치와 비교해 위치 표현의 순수 효과를 측정한 실험이 아닙니다.

### 한계·복습과 다음 연결

- [ ] QKV를 `inputs`가 아니라 `combined`에서 만드는 이유를 설명한다.
- [ ] 위치 2의 Q `[3, 2]`와 전체 내적 `[2, 6, 8]`을 계산한다.
- [ ] 같은 토큰이 두 번 있어도 비중은 위치별로 세 개임을 구분한다.
- [ ] 위치 1의 세 번째 내적 점수는 6인데도 비중은 0인 이유를 설명한다.
- [ ] 비중과 정보 벡터, 다음 토큰 확률을 구분한다.

이 절은 수동 위치 표·단일 head 산술 계산입니다. Attention head의 출력 투영·잔차 연결·정규화·여러 head·loss·역전파·모델 저장은 구현하지 않습니다. 이 절의 코드는 단어를 선택하거나 생성하지 않으며, 다음 절에서 어휘 출력층을 별도로 연결합니다.

## 13. 후속 실습 · 정보 벡터를 어휘 점수와 확률로 바꾸기

### 목표와 개념

앞 절의 마지막 위치 출력은 숫자 두 개의 정보 벡터였습니다. 이번에는 **그 벡터 하나를 고정 입력으로 가져와**, 어휘 후보마다 점수를 만들고 softmax로 확률을 계산합니다. Attention 전체를 재실행하거나 모델 파일에서 벡터를 읽지 않습니다.

```text
앞 절 위치 2의 Attention 정보 벡터 hidden (1×2)
          ↓ 출력 행렬 w_vocab (2×3), bias 없음
어휘 점수 logits (1×3): failed, healthy, <eos>
          ↓ 어휘 축 softmax
세 후보의 확률 → 가장 큰 확률의 후보 확인
```

`logits`는 softmax 전 점수이며 음수일 수도 있고 합이 1일 필요도 없습니다. softmax 뒤에는 세 확률의 합이 1입니다. 원래 Transformer의 선형 변환·softmax 출력 개념은 *Attention Is All You Need*, §3.4, arXiv:1706.03762를 참고합니다. 여기서는 완전한 decoder 출력 대신 앞 절의 Attention 정보 벡터에 **직접 정한 출력 행렬**을 적용하는 작은 산술 예제입니다. 학습된 모델이나 정답을 맞히는 실험이 아닙니다.

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**의 기존 `~/llm-lab/source` 가상환경에서 실행합니다. Python 3.10 이상과 표준 라이브러리만 필요합니다. `cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 마지막 `PY`까지의 코드 실행입니다. 앞 실습의 변수·소스 파일이 없어도 실행되며 GPU·패키지 설치·파일 읽기·쓰기·학습은 없습니다.

**소스:** [09-vocabulary-output.py](source/08-attention/09-vocabulary-output.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import exp, isclose

# 앞 실습 위치 2의 반올림 전 Attention 출력입니다. 여기서는 재계산하지 않습니다.
hidden = [1.61332902229994, 5.931436569567589]
# 출력 후보만 담은 교육용 어휘입니다. 앞 입력 토큰 목록과 다릅니다.
vocabulary = ["failed", "healthy", "<eos>"]
# 행벡터 hidden (1×2) × 직접 정한 출력 행렬 (2×3); bias는 없습니다.
# ponytail: 고정 교육용 계수; 학습된 모델의 예측 품질을 나타내지 않습니다.
w_vocab = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]
assert len(w_vocab) == len(hidden)
assert all(len(row) == len(vocabulary) for row in w_vocab)
logits = [sum(value * row[column] for value, row in zip(hidden, w_vocab, strict=True))
          for column in range(len(vocabulary))]
exps = [exp(score - max(logits)) for score in logits]
probabilities = [number / sum(exps) for number in exps]
expected_logits = [1.61332902229994, 2.9657182847837945, -1.61332902229994]
# 독립 Decimal 계산의 검사 기준이며 확률 계산에는 쓰지 않습니다.
expected_probabilities = [0.20381779795457093, 0.7880926900858558, 0.008089511959573232]
assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(logits, expected_logits, strict=True))
assert all(isclose(a, b, abs_tol=1e-12)
           for a, b in zip(probabilities, expected_probabilities, strict=True))
assert all(0.0 < probability < 1.0 for probability in probabilities)
assert isclose(sum(probabilities), 1.0)
chosen = max(range(len(vocabulary)), key=probabilities.__getitem__)
assert vocabulary[chosen] == "healthy"
print("Attention 정보 벡터:", hidden)
for token, score, probability in zip(vocabulary, logits, probabilities, strict=True):
    print(f"{token}: 점수={score:.4f}, 확률={probability:.4f}")
print(f"확률 합: {sum(probabilities):.4f}")
print("가장 큰 확률의 후보:", vocabulary[chosen])
print("검사 통과: 정보 벡터를 어휘 점수와 확률로 바꿨습니다. 학습·문장 생성은 하지 않았습니다")
PY
```

### 코드와 계산 해설

- `hidden`은 앞 절 위치 2의 반올림 전 출력 `[1.61332902229994, 5.931436569567589]`를 옮긴 값입니다. 이전 표시값 `[1.6133, 5.9314]`로 바꿔 넣지 않습니다. 여기서는 Attention 계산을 반복하지 않고 출력층 계산만 분리합니다.
- `vocabulary`는 출력 후보의 순서입니다. 입력 `gpu current gpu`와 다른 목록이며, 두 `gpu` 위치를 포함한 Attention 비중의 순서와도 다릅니다. 일반 언어 모델은 훨씬 큰 어휘에 점수를 주지만, 여기서는 세 후보만 둡니다.
- `w_vocab`의 **열 하나가 어휘 하나**에 대응합니다. 입력 숫자 두 개로 각 열을 계산해 점수 세 개를 만듭니다. 기존 2×2 QKV 전용 `project`를 2×3 출력 행렬에 그대로 쓰지 않습니다.
- `exps`에서 최댓값을 빼는 안정화와 softmax 방식은 앞 실습과 같습니다. 하지만 이번 softmax의 축은 **참고할 입력 위치가 아니라 출력 어휘**입니다. 둘 다 숫자가 세 개인 것은 이 예시의 우연입니다. 어휘에 causal mask를 적용하지 않습니다.
- `chosen`은 가장 큰 확률의 인덱스를 찾는 argmax 방식입니다. 무작위 추출이 아니며, 후보 하나를 확인할 뿐 반복 생성하지 않습니다.
- `expected_*`는 별도 계산한 검사 기준입니다. softmax 계산에 정답 확률을 넣지 않습니다.

```text
failed 점수 = hidden[0]×1  + hidden[1]×0   = hidden[0]
healthy 점수 = hidden[0]×0 + hidden[1]×0.5 = hidden[1]/2
<eos> 점수 = hidden[0]×(-1) + hidden[1]×0  = -hidden[0]
```

이 어휘 출력 행렬은 Attention 안의 head 출력 투영 행렬 `Wo`와 다른 역할입니다. 이번 코드는 head 출력 투영·Transformer 블록 전체·입출력 임베딩 가중치 공유를 구현하지 않습니다.

### 로컬 산술 실행 예시

아래는 정본 코드를 로컬 Python으로 실행해 assert를 통과한 출력입니다.

```text
Attention 정보 벡터: [1.61332902229994, 5.931436569567589]
failed: 점수=1.6133, 확률=0.2038
healthy: 점수=2.9657, 확률=0.7881
<eos>: 점수=-1.6133, 확률=0.0081
확률 합: 1.0000
가장 큰 확률의 후보: healthy
검사 통과: 정보 벡터를 어휘 점수와 확률로 바꿨습니다. 학습·문장 생성은 하지 않았습니다
```

`healthy`가 선택된 것은 직접 정한 행렬에서 그 점수가 가장 높기 때문입니다. **GPU 상태가 healthy라고 판단했거나 정답을 맞힌 증거가 아닙니다.** 확률 약 0.7881도 임의 계수로 계산한 이 세 후보 안의 분포이지, 검증된 신뢰도가 아닙니다. `<eos>`는 후보로만 포함했으며 반복 생성·종료 규칙은 실행하지 않습니다.

### 복습과 다음 연결

- [ ] 정보 벡터 두 숫자가 어휘 점수 세 숫자로 바뀌는 이유를 설명한다.
- [ ] softmax 전 점수와 softmax 뒤 확률을 구분한다.
- [ ] 입력 위치에 대한 Attention 비중과 출력 어휘 확률을 구분한다.
- [ ] 직접 정한 계수에서 후보가 선택된 것을 학습 성공으로 부르지 않는다.

이번에는 마지막 위치 하나의 출력층만 계산합니다. 정답·loss·역전파·파라미터 수정·모델 저장·반복 생성은 없습니다. 다음에는 정답 토큰과 이 확률을 비교해 loss를 계산하는 과정을 연결할 수 있습니다.

## 14. 후속 실습 · 정답 확률에서 loss 계산하기

### 목표와 개념

13절은 최대 확률 후보가 `healthy`임을 확인했습니다. 하지만 **예측 후보와 정답은 다릅니다.** 정답은 argmax가 아니라 데이터가 제공합니다. 이번에는 앞 절의 반올림 전 확률을 가져와, 설명을 위해 정답을 `failed`로 정합니다. 실제 GPU 상태의 정답을 판정하는 실험은 아닙니다.

하나의 정답 토큰을 갖는 한 위치에서, 클래스 가중치·label smoothing이 없는 cross-entropy는 **`loss = -ln(정답 토큰의 확률)`**입니다. `ln`은 자연로그입니다. 세 확률을 모두 더하거나, 최대 확률을 무조건 골라 loss를 계산하지 않습니다.

```text
고정 어휘 확률 + 데이터의 정답 (여기서는 failed로 가정)
          ↓ 정답 인덱스로 확률 하나 선택
정답 확률 → 음의 자연로그 → 한 위치의 loss (숫자 하나)
```

### 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source`에서 실행합니다. Python 3.10 이상과 표준 라이브러리만 필요합니다. `cd`는 폴더 이동, `source`는 기존 가상환경 활성화, `python -`는 마지막 `PY`까지 코드 실행입니다. 앞 실습 변수·파일은 필요하지 않으며 GPU·설치·파일 읽기·쓰기·학습은 없습니다.

**소스:** [10-target-loss.py](source/08-attention/10-target-loss.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import isclose, isfinite, log

# 앞 출력층의 반올림 전 확률을 고정 입력으로 가져옵니다. 재계산하지 않습니다.
vocabulary = ["failed", "healthy", "<eos>"]
probabilities = [0.20381779795457093, 0.7880926900858558, 0.008089511959573232]
# ponytail: 세 후보·한 위치 전용 산술; 실제 학습에서는 logits와 CrossEntropyLoss를 사용합니다.
assert vocabulary == ["failed", "healthy", "<eos>"]
assert len(probabilities) == len(vocabulary)
assert all(isfinite(p) and 0.0 < p <= 1.0 for p in probabilities)
assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)
# 정답은 argmax가 아니라 데이터가 제공합니다. 여기서는 설명을 위해 failed로 정합니다.
target = "failed"
assert target in vocabulary
target_index = vocabulary.index(target)
target_probability = probabilities[target_index]
loss = -log(target_probability)
assert isclose(loss, 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
assert isclose(-log(1.0), 0.0, abs_tol=1e-12)
assert -log(0.8) < -log(0.2)
print("교육용 정답 토큰:", target)
print(f"정답에 부여한 확률: {target_probability:.4f}")
print(f"한 위치의 loss = -ln(정답 확률): {loss:.4f}")
print("같은 분포에서 정답만 달리 가정한 비교 (학습 전후 아님):")
# 독립 Decimal 계산 기준이며 loss 계산에 사용하지 않습니다.
expected_losses = [1.5905288315131196, 0.23813956902926515, 4.817186876113]
for token, probability, expected in zip(vocabulary, probabilities, expected_losses, strict=True):
    candidate_loss = -log(probability)
    assert isclose(candidate_loss, expected, rel_tol=0.0, abs_tol=1e-12)
    print(f"  정답={token}: 확률={probability:.4f}, loss={candidate_loss:.4f}")
print("검사 통과: 정답 확률에서 loss를 계산했습니다. 역전파·가중치 수정은 하지 않았습니다")
PY
```

### 코드와 계산 해설

- `vocabulary`와 `probabilities`는 같은 순서로 대응합니다. `target_index`는 정답 문자열의 위치, `target_probability`는 그 위치의 확률입니다. 출력은 숫자 하나의 loss입니다.
- `log(x)`는 Python의 자연로그입니다. 정답 확률이 작을수록 `-log(p)`가 커집니다. 확률이 1이면 손실은 0입니다. loss는 확률이 아니므로 1보다 클 수 있습니다.
- 표시용 `0.2038`이 아니라 반올림 전 확률로 계산합니다. `isfinite`·범위·합 검사는 이 고정 분포를 확인하며, 0을 임의의 작은 숫자로 바꿔 오류를 숨기지 않습니다.
- `expected_losses`는 독립 고정밀 Decimal 계산의 검사 기준입니다. loss 입력이나 학습 정답이 아닙니다. 이 예시 전용 assert를 다른 정답 실험에 그대로 적용하지 않습니다.
- 마지막 반복문은 **동일한 분포에서 정답만 바꿔 가정한 세 경우**입니다. 세 학습 샘플이나 학습 전후의 loss 감소가 아닙니다. 실제 데이터의 정답을 손실이 작아지도록 바꾸면 안 됩니다.

앞 모델에서 사용한 PyTorch `nn.CrossEntropyLoss`에는 **softmax 전 logits와 정답 클래스 인덱스**를 넣습니다. 확률을 직접 꺼내 로그를 계산하는 이번 코드는 원리를 보기 위한 분리 실습입니다. 실제 학습에서는 안정적인 log-softmax 기반 계산을 사용하며 softmax를 한 번 더 붙여 넣지 않습니다. 근거: PyTorch 공식 `CrossEntropyLoss` 문서의 클래스 인덱스 정답 수식과 `LogSoftmax`·`NLLLoss` 동치 설명.

### 로컬 산술 실행 예시

정본 소스를 로컬 Python으로 실행해 assert를 통과한 출력입니다. 학습 결과가 아닙니다.

```text
교육용 정답 토큰: failed
정답에 부여한 확률: 0.2038
한 위치의 loss = -ln(정답 확률): 1.5905
같은 분포에서 정답만 달리 가정한 비교 (학습 전후 아님):
  정답=failed: 확률=0.2038, loss=1.5905
  정답=healthy: 확률=0.7881, loss=0.2381
  정답=<eos>: 확률=0.0081, loss=4.8172
검사 통과: 정답 확률에서 loss를 계산했습니다. 역전파·가중치 수정은 하지 않았습니다
```

### 한계·복습과 다음 연결

- [ ] 최대 확률 후보와 데이터의 정답을 구분한다.
- [ ] 같은 분포에서도 정답에 따라 loss가 다른 이유를 설명한다.
- [ ] 정답 확률이 커질수록 loss가 작아짐을 설명한다.
- [ ] loss 계산과 역전파·가중치 갱신을 구분한다.

이번에는 고정 확률에서 한 위치의 손실만 계산합니다. 전체 문장·batch 평균, padding 제외, label smoothing, 기울기·파라미터 갱신·모델 저장·반복 생성은 구현하지 않습니다. 다음에는 loss에서 출력 점수의 기울기로 연결할 수 있습니다.

[← 이전: 학습 점검](07-resume.md) · [목차](README.md)
