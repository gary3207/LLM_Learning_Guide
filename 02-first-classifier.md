# 02. 첫 텍스트 분류기 — 코드를 읽으며 학습의 흐름 이해하기

[← 이전: 환경과 기초](01-foundations.md) · [목차](README.md) · [다음: 추론과 데이터 보완 →](03-inference-and-data.md)

## 1. 왜 작은 분류기로 시작할까?

세 종류의 짧은 문의를 분류하며 **모델이 데이터와 정답으로부터 무엇을 바꾸는지** 확인합니다.

```text
문의 문장과 정답 준비
        ↓
단어 목록과 등장 횟수로 숫자 입력 만들기
        ↓
모델로 GPU·네트워크·스토리지 점수 계산
        ↓
예측과 정답을 비교해 loss 계산
        ↓
기울기 계산 → 가중치 수정 → 반복
        ↓
학습 데이터 정확도와 세 문장의 예측 확인
```

이 모델은 문장을 생성하는 LLM이 아닙니다. 외부 모델이나 데이터를 내려받지 않고, 코드에 넣은 문의를 세 분류 중 하나로 고르는 **작은 선형 분류기**입니다. 핵심은 모델 크기가 아니라 **예측 → 오차 → 기울기 → 수정**의 관계입니다.

**학습 목표**

- 문자열을 어떤 규칙으로 숫자 입력으로 바꾸는지 설명한다.
- 사람이 정한 데이터·규칙과 모델이 학습하는 가중치를 구분한다.
- `zero_grad()`, `backward()`, `step()`의 역할을 설명한다.
- 학습 데이터 정확도와 새로운 데이터에서의 성능을 구분한다.
- 코드와 체크포인트의 차이를 이해하고 가중치·어휘·분류 정보를 저장한다.

**이 장을 따라가는 순서**

1. [3절 첫 실행](#3-첫-분류기-전체-코드와-실행-예시): 학습 흐름을 체험합니다. 코드 파일이나 체크포인트는 자동 저장되지 않습니다.
2. [4절부터 코드 읽기](#4-실행-래퍼와-python-시작-부분) → [12절 검사 범위](#12-마지막-assert의-성공-조건은-좁다): 입력·학습·출력과 loss 감소 검사의 의미를 확인합니다.
3. [14절 파일 배치·저장](#14-학습-코드를-파일로-보관하고-모델-저장하기): 새 모델을 만들어 100회 학습한 뒤 `classifier.pt`를 저장합니다. 3절의 모델을 이어 학습하는 과정이 아닙니다.
4. [15절 복원 준비·경로 점검](#15-체크포인트와-복원-준비) → [03장 첫 복원 실습](03-inference-and-data.md): 별도 Python 프로세스에서 저장 모델을 CPU로 읽습니다. 이 복원에는 재학습이 없습니다.

## 2. 실행 환경 준비

**실행 위치:** OCI VM에 SSH로 접속한 Linux Bash 터미널입니다. 앞 장에서 만든 실습 폴더 `~/llm-lab/source`와 가상환경 `.venv`를 사용합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
```

`(.venv)`는 현재 터미널에서 가상환경이 활성화됐다는 표시입니다. 환경을 다시 설치하는 명령이 아닙니다. 앞 장의 GPU 계산과 NumPy 변환 검사가 통과하는 환경에서 진행합니다. 검사가 실패하면 드라이버·Python 환경·패키지 상태를 먼저 점검합니다.

## 3. 첫 분류기 전체 코드와 실행 예시

[소스 파일: first_classifier.py](source/02-first-classifier/first_classifier.py)

다음 블록을 `python - <<'PY'`부터 마지막 `PY`까지 Bash에 입력합니다. 링크한 `.py` 파일에는 Bash 래퍼를 제외한 Python 본문이 들어 있습니다. 이 버전은 학습 흐름을 확인하며, 체크포인트 저장은 14절에서 추가합니다.

```bash
python - <<'PY'
import torch
from torch import nn

torch.manual_seed(42)
assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
device = "cuda"

# 1. 학습용 문의와 정답: GPU=0, 네트워크=1, 스토리지=2
labels = ["GPU", "네트워크", "스토리지"]
data = [
    ("gpu memory error", 0),
    ("gpu device unavailable", 0),
    ("cuda memory unavailable", 0),
    ("network connection error", 1),
    ("network timeout", 1),
    ("connection timeout", 1),
    ("disk storage full", 2),
    ("disk space error", 2),
    ("storage space full", 2),
]

# 2. 문장을 단어별 등장 횟수로 변환
vocab = sorted({word for text, _ in data for word in text.split()})

def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]

x = torch.tensor(
    [encode(text) for text, _ in data],
    dtype=torch.float32, device=device
)
y = torch.tensor(
    [label for _, label in data],
    dtype=torch.long, device=device
)

# 3. 모델과 학습 방법 준비
model = nn.Linear(len(vocab), len(labels)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

initial_loss = loss_fn(model(x), y).item()
print(f"학습 전 loss: {initial_loss:.4f}")

# 4. 전체 학습 데이터를 100번 반복
model.train()
for epoch in range(1, 101):
    scores = model(x)
    loss = loss_fn(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            current_loss = loss_fn(model(x), y).item()
        print(f"epoch {epoch:3d} | loss: {current_loss:.4f}")

# 5. 학습 결과 확인
model.eval()
with torch.no_grad():
    final_scores = model(x)
    final_loss = loss_fn(final_scores, y).item()
    accuracy = (final_scores.argmax(dim=1) == y).float().mean().item()
    print(f"\n학습 데이터 정확도: {accuracy:.1%}")

    for text in ["gpu memory", "network connection", "disk full"]:
        sample = torch.tensor(
            [encode(text)], dtype=torch.float32, device=device
        )
        prediction = model(sample).argmax(dim=1).item()
        print(f"{text} -> {labels[prediction]}")

assert final_loss < initial_loss, "학습 후 오차가 줄지 않았습니다."
print("\n검증 통과: 학습 후 오차 감소")
PY
```

### 실행 예시

아래 수치는 결과를 읽는 방법을 보여주는 예입니다. `torch.manual_seed(42)`를 사용해도 장치·PyTorch 버전·실행 환경에 따라 수치가 달라질 수 있으며, 동일한 loss나 정확도를 보장하지 않습니다.

```text
학습 전 loss: 1.2353
epoch  20 | loss: 0.6780
epoch  40 | loss: 0.4176
epoch  60 | loss: 0.2877
epoch  80 | loss: 0.2148
epoch 100 | loss: 0.1695

학습 데이터 정확도: 100.0%
gpu memory -> GPU
network connection -> 네트워크
disk full -> 스토리지

검증 통과: 학습 후 오차 감소
```

예시에서는 loss가 `1.2353`에서 `0.1695`로 감소하고 학습 문의를 모두 맞힙니다. 세 예측도 의도한 분류와 같습니다. 그러나 이는 **새 문의 전반의 성능이나 별도 검증 데이터의 성능이 아닙니다.** 아래 절에서 각 계산과 검사의 범위를 읽어봅니다.

## 4. 실행 래퍼와 Python 시작 부분

### 4.1 `python - <<'PY'`는 Python 코드가 아니다

이 부분은 **Bash의 here-document**입니다.

- `python`: 활성화된 환경의 Python을 실행합니다.
- `-`: 파일 대신 표준 입력에서 코드를 받습니다.
- `<<'PY'`: 마지막 `PY` 전까지의 여러 줄을 입력으로 보냅니다.
- 따옴표로 감싼 구분자: 내부의 `$` 등을 Bash가 먼저 해석하지 않고 전달합니다.
- 마지막 `PY`: 입력 끝 표시이며, Python 본문에 넣지 않습니다.

따라서 `.py` 파일을 읽을 때는 `import torch`부터 보면 됩니다. 표준 입력으로 실행하는 것만으로 코드 파일이나 체크포인트가 자동 저장되지는 않습니다.

### 4.2 도구 불러오기

| 코드 | 의미와 이유 |
|---|---|
| `import torch` | 텐서 등 PyTorch 기능을 사용합니다. 학습된 모델을 불러오는 것은 아닙니다. |
| `from torch import nn` | 모델 구성 요소와 손실함수를 제공하는 `nn`을 사용합니다. |

이후 `torch.tensor`, `nn.Linear`, `nn.CrossEntropyLoss`라는 이름이 여기서 연결됩니다.

### 4.3 난수와 장치 설정

| 코드 | 의미와 이유 |
|---|---|
| `torch.manual_seed(42)` | 초기 무작위 값을 비교하기 쉽도록 난수의 시작점을 정합니다. `42`가 성능을 높이는 값은 아니며, 다른 장치·버전까지 동일 결과를 보장하지 않습니다. |
| `assert torch.cuda.is_available(), …` | 이 가이드가 사용하는 GPU 경로를 사용할 수 없으면 오류를 발생시킵니다. 학습 자체에 GPU가 절대 필요하다는 뜻이 아닙니다. |
| `device = "cuda"` | 사용할 장치 이름을 변수에 저장합니다. 이것만으로 모든 Python 계산이 GPU로 바뀌지는 않습니다. |

앞 장의 작은 곱셈과 이 장의 분류기는 CPU로도 원리를 배울 수 있는 규모입니다. 이 가이드의 코드는 CUDA 환경을 전제로 하므로 GPU 검사를 포함합니다. CPU 실행을 선택한다면 검사 조건과 `device`를 함께 조정하고, 모델과 입력을 같은 장치에 배치해야 합니다.


## 5. 문장과 정답: 사람이 준비한 학습 자료

### 5.1 `labels`: 분류 번호와 이름의 대응표

```python
labels = ["GPU", "네트워크", "스토리지"]
```

리스트의 위치가 분류 번호입니다. `0`은 GPU, `1`은 네트워크, `2`는 스토리지입니다. 모델 출력의 세 점수도 이 순서로 해석합니다. 나중에 `labels[prediction]`으로 번호를 읽을 수 있는 이름으로 바꿉니다.

### 5.2 `data`: 문장·정답 쌍의 목록

`("gpu memory error", 0)`은 문의 문장과 정답 번호를 한 쌍으로 묶은 것입니다. 이 데이터에서는 사람이 GPU라고 정답을 붙였습니다. 모델이 스스로 정답을 알아낸 것이 아닙니다.

전체 코드의 `data`에는 분류마다 세 개씩, 총 아홉 개의 문장이 들어 있습니다. 외부 데이터 다운로드는 없습니다. 학습이란 이 문장·정답 쌍을 기준으로 모델의 가중치를 조정하는 작업입니다.

`for text, _ in data`는 각 쌍에서 문장을 `text`로 꺼냅니다. `_`도 변수 이름이지만 그 값은 사용하지 않겠다는 의도를 나타냅니다. 특별한 삭제 기능이 아닙니다. 반대로 `for _, label in data`는 정답만 사용합니다.


## 6. 문장을 숫자로 바꾸기: 이 부분은 학습이 아니다

### 6.1 `vocab` 한 줄 풀어 읽기

```python
vocab = sorted({word for text, _ in data for word in text.split()})
```

오른쪽에서 하는 일은 **전체 학습 문장의 단어를 모아 중복을 제거하고 정렬하기**입니다.

| 부분 | 하는 일 |
|---|---|
| `for text, _ in data` | 문장을 차례로 꺼냄 |
| `text.split()` | 문장을 공백 기준 단어들로 나눔 |
| `for word in …` | 단어를 하나씩 꺼냄 |
| `{ … }` | 집합으로 모아 중복 제거 |
| `sorted(…)` | 정렬된 리스트로 바꿈 |

정렬을 하는 이유는 **숫자 벡터의 각 위치가 어떤 단어인지 고정**하기 위해서입니다. 학습 때와 예측 때 같은 목록·순서를 써야 합니다.

### 6.2 `encode` 함수의 세 줄

```python
def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]
```

1. `def encode(text):`는 문장 하나를 받아 숫자 목록을 반환할 함수를 **정의**합니다. 이 줄에서 모든 문장을 변환하는 것은 아닙니다.
2. `lower()`로 소문자로 맞추고 `split()`으로 공백 기준 단어를 나눕니다. 따라서 입력의 `GPU`는 `gpu`로 처리됩니다.
3. `vocab`의 각 단어가 그 문장에서 몇 번 나왔는지 셉니다. `return`으로 그 리스트를 돌려줍니다.

이 방식은 단어 **등장 횟수**를 사용합니다. 단어의 순서나 의미를 모델에 그대로 넣는 방법은 아닙니다. 이 함수 자체도 데이터에 맞춰 학습되는 부분이 아니라 **사람이 정한 변환 규칙**입니다.

예를 들어 `gpu memory`와 `memory gpu`는 같은 횟수 벡터가 됩니다. 공백으로만 나누므로 `gpu`와 `gpu,`는 다른 문자열이며, 뒤의 문자열이 `vocab`에 없으면 반영되지 않습니다. 이는 변환 규칙 설명이지 모델의 실행·예측 결과가 아닙니다.

### 6.3 단어 목록과 숫자 입력 확인하기

아래는 데이터와 코드 구조로 정해지는 **설명용 예시**입니다. 단어 목록, 문장 하나의 벡터, 정답 순서와 크기를 차례로 살펴봅니다. 학습 loss를 나타내는 표가 아닙니다.

```text
vocab: ['connection', 'cuda', 'device', 'disk', 'error', 'full', 'gpu', 'memory', 'network', 'space', 'storage', 'timeout', 'unavailable']
vocab length: 13
gpu memory error: [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
labels: [0, 0, 0, 1, 1, 1, 2, 2, 2]
shapes: (9, 13) (9, 3)
parameters: 42
```

`"gpu memory error"`의 벡터에서 `error`, `gpu`, `memory`의 위치만 1입니다. 나머지 단어는 이 문장에 없으므로 0입니다. 단어가 몇 번 나왔는지를 고정된 열 위치에 넣은 것입니다.

여기서 `shapes`는 입력과 모델 출력의 크기입니다. 첫 번째는 문장 수 × 어휘 수, 두 번째는 문장 수 × 분류 수입니다. 전체 코드가 이 요약을 자동 출력하는 것은 아닙니다.


## 7. 입력 `x`와 정답 `y`: 역할·모양·자료형

| 코드의 부분 | 뜻 |
|---|---|
| `[encode(text) for text, _ in data]` | 모든 문장을 같은 단어 순서의 숫자 목록으로 바꿈 |
| `torch.tensor(…)` | Python 목록을 PyTorch 텐서로 만듦 |
| `dtype=torch.float32` | 모델 입력을 32비트 실수로 만듦 |
| `[label for _, label in data]` | 정답 번호만 차례대로 모음 |
| `dtype=torch.long` | 이 손실함수가 분류 번호로 받는 정답을 정수형으로 만듦 |
| `device=device` | 왼쪽은 인자 이름, 오른쪽은 앞에서 정한 `"cuda"` 변수 값 |

코드의 입력·출력 구조는 다음과 같습니다. 아래는 구조 설명이며 학습 실행 로그가 아닙니다.

```text
x: [9, 13]  문의 9개 × 단어별 숫자 13개
y: [9]      각 문의의 정답 번호

model(x): [9, 3]  각 문의의 GPU·네트워크·스토리지 점수
```

단어 횟수는 원래 정수이지만 `Linear`의 실수 가중치와 계산하므로 입력은 실수형입니다. 정답 `y`는 “분류 번호”이므로 `long`입니다. 첫 번째 입력 행은 첫 번째 정답과 대응합니다. 두 목록의 순서가 다르면 잘못된 정답으로 학습하게 됩니다.

또한 텐서뿐 아니라 뒤에서 **모델도 같은 GPU로 옮깁니다.** 장치 이름을 변수에 저장한 것과 실제 텐서·모델 배치는 다른 일입니다.


## 8. 모델·손실함수·옵티마이저 준비

### 8.1 `nn.Linear`: 학습되는 조절값이 들어 있는 곳

```python
model = nn.Linear(len(vocab), len(labels)).to(device)
```

이 데이터에서는 입력 크기 13, 출력 크기 3인 선형 모델입니다.

- `len(vocab)`: 입력 숫자 개수입니다.
- `len(labels)`: 출력할 분류별 점수 개수입니다.
- `Linear`: 입력에 가중치를 곱해 합산하고 편향을 더해 점수를 만듭니다.
- `.to(device)`: 모델 파라미터를 선택한 장치로 옮깁니다.

가중치는 입력별·분류별로 있고, 각 분류의 편향도 있습니다. 가중치 13 × 3개와 편향 3개를 합쳐 **42개 파라미터**입니다. 이 가중치·편향이 학습으로 바뀝니다. loss만으로 특정 단어의 가중치 변화량을 알 수는 없으며, 그 값은 파라미터를 따로 비교해야 합니다.

### 8.2 `CrossEntropyLoss`: 오차를 숫자로 만드는 도구

```python
loss_fn = nn.CrossEntropyLoss()
```

세 분류 중 정답 하나를 고르는 예측과 정답 번호를 비교합니다. 모델이 내는 세 값은 확률이 아니라 **원시 점수(logits)**입니다. 이 손실함수는 그 점수를 직접 받으므로 앞에 별도의 `softmax`를 붙이지 않았습니다. 기본 설정은 예제별 손실의 평균입니다.

정확도는 맞았는지만 세지만, 이 loss는 출력 점수에 따라 달라지고 기울기를 구할 수 있으므로 학습 기준으로 사용합니다.

가장 큰 점수의 분류가 같아도 정답과 다른 분류의 상대 점수가 바뀌면 loss는 달라질 수 있습니다. 그래서 정확도가 같은 두 모델도 loss가 다를 수 있으며, [04장](04-evaluation.md)에서 이를 모델 선택의 보조 기준으로 사용합니다.

### 8.3 `SGD`: 실제 수정 방법 준비

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

| 부분 | 뜻 |
|---|---|
| `torch.optim.SGD` | 가중치 수정에 사용할 옵티마이저 |
| `model.parameters()` | 수정할 가중치·편향 목록 |
| `lr=0.1` | 수정 크기를 조절하는 학습률 |

이번 설정의 개념적 수정 규칙은 **새 가중치 = 기존 가중치 − 학습률 × 기울기**입니다. `0.1`은 정확도나 학습 데이터 비율이 아닙니다. 이 생성 줄은 수정 방법을 준비할 뿐 **아직 수정하지 않습니다.**


## 9. 학습 전 기준값과 출력 형식

```python
initial_loss = loss_fn(model(x), y).item()
print(f"학습 전 loss: {initial_loss:.4f}")
```

안쪽부터 읽으면 `model(x)`로 점수를 계산하고, `loss_fn(…, y)`로 정답과 비교한 뒤 `.item()`으로 숫자 하나가 든 텐서에서 일반 Python 숫자를 꺼냅니다. 마지막 loss와 비교할 **시작 기준값**입니다.

`f"…"`는 문자열 안에 변수 값을 넣는 형식입니다. `:.4f`는 소수점 아래 네 자리로 표시하라는 뜻입니다. 저장된 숫자를 바꾸는 것이 아니라 출력 모양을 바꿉니다.

3절 실행 예시의 시작값은 `1.2353`입니다. 자신의 출력에서는 시작값과 최종값을 함께 비교합니다. 다른 환경에서 이 숫자와 정확히 일치하는지를 성공 조건으로 삼지 않습니다.

## 10. 학습 루프: 무엇이 언제 바뀌는가?

### 10.1 모드 설정과 반복

| 코드 | 해설 |
|---|---|
| `model.train()` | 모델을 학습 모드로 설정. 이것만으로 학습하거나 가중치를 수정하지 않음 |
| `for epoch in range(1, 101):` | 1부터 100까지 반복. `range`의 끝값 101은 포함하지 않음 |

`epoch`는 전체 학습 데이터를 한 번 사용하는 단위입니다. 이 코드는 매번 아홉 문의 전체를 한꺼번에 사용하므로 한 epoch에 가중치 수정도 한 번입니다.

`train()`과 뒤의 `eval()`은 학습·평가 때 동작이 다른 구성 요소의 모드를 정합니다. 지금 모델은 `Linear` 하나라 이 모드만 바꾼다고 계산 방식이 달라지지 않습니다. 이 장에서는 별도의 층을 추가하지 않습니다.

### 10.2 가장 중요한 다섯 줄

```python
scores = model(x)
loss = loss_fn(scores, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

| 줄 | 하는 일 | 하지 않는 일 |
|---|---|---|
| `scores = model(x)` | 현재 가중치로 점수 계산 | 가중치 수정은 아직 아님 |
| `loss = loss_fn(scores, y)` | 현재 점수와 정답을 비교해 손실 텐서 생성 | 학습 완료 판정이 아님 |
| `optimizer.zero_grad()` | 앞에서 남아 있는 기울기 초기화 | 배운 가중치를 지우는 명령이 아님 |
| `loss.backward()` | 계산 경로를 거꾸로 따라가며 파라미터의 기울기 계산 | 가중치를 직접 수정하지 않음 |
| `optimizer.step()` | 기울기를 사용해 가중치·편향 수정 | 새 데이터에서의 성능 보장은 아님 |

기울기는 “이 파라미터를 조금 바꾸면 loss가 어느 방향으로 얼마나 달라지는가”에 관한 정보입니다. PyTorch의 자동 미분이 그 계산을 도와줍니다.

PyTorch의 기울기는 기본적으로 누적되므로 이번 반복의 정보만 쓰려면 초기화합니다. 학습용 `loss`에 `.item()`을 붙이지 않는 이유도 중요합니다. 다음 `backward()`에 미분할 수 있는 **텐서와 계산 경로**가 필요하기 때문입니다. `.item()`으로 꺼낸 일반 숫자는 그 역할을 할 수 없습니다.

### 10.3 진행 상황은 수정 후의 loss로 확인

```python
if epoch % 20 == 0:
    with torch.no_grad():
        current_loss = loss_fn(model(x), y).item()
    print(f"epoch {epoch:3d} | loss: {current_loss:.4f}")
```

- `% 20 == 0`: 20으로 나눈 나머지가 0일 때만 출력합니다.
- `torch.no_grad()`: 이 블록에서는 미분용 계산 기록을 만들지 않습니다. 확인용 계산이므로 가중치 수정을 위한 기록이 필요 없습니다.
- `current_loss`: **방금 `step()`으로 수정한 모델**로 다시 계산한 값입니다.
- 이미 있던 `loss`는 수정 **이전**의 값이므로 그대로 출력하는 것과 시점이 다릅니다.
- `:3d`: 정수 표시 폭을 세 칸으로 맞추고, `:.4f`는 소수점 네 자리로 표시합니다.

이 코드는 매 반복의 loss 감소를 검사하지 않습니다. 중간 기록을 보는 것과 매번 감소한다고 증명하는 것은 다릅니다.


## 11. 평가와 예측: 여기서는 더 학습하지 않는다

### 11.1 `eval()`과 `no_grad()`는 역할이 다르다

```text
model.eval()
with torch.no_grad():
```

`eval()`은 모델의 평가 모드를 설정하고, `no_grad()`는 해당 블록의 미분용 기록을 끕니다. `eval()` 자체가 자동으로 정확도를 계산하거나 기울기 추적을 끄는 것은 아닙니다.

그 안에서 `final_scores = model(x)`로 학습 문의의 점수를 구하고, `final_loss = loss_fn(final_scores, y).item()`으로 마지막 오차를 꺼냅니다.

### 11.2 정확도 한 줄 읽기

```python
accuracy = (final_scores.argmax(dim=1) == y).float().mean().item()
```

| 순서 | 의미 |
|---|---|
| `argmax(dim=1)` | 각 문의 행에서 세 분류 점수 중 가장 큰 값의 **위치**, 즉 분류 번호 선택 |
| `== y` | 예측 번호가 정답 번호와 같은지 비교 |
| `.float()` | 맞음·틀림을 실수형으로 바꿈 |
| `.mean()` | 맞힌 비율 계산 |
| `.item()` | 비율을 Python 숫자로 꺼냄 |

`argmax`는 가장 큰 점수 자체를 돌려주는 것이 아니라 그 **위치**를 돌려줍니다. `dim=1`은 이 텐서에서 분류 점수 열 방향입니다. 출력문의 `\n`은 줄바꿈, `:.1%`는 소수점 한 자리의 백분율 표시입니다.

**가장 중요한 한계:** 여기에 넣는 `x`는 이미 학습에 사용한 데이터입니다. 따라서 이 비율은 “본 문제를 얼마나 맞혔는가”이며 새로운 문의에 대한 검증 정확도가 아닙니다. 설령 높은 숫자가 나와도 일반화 성능이 입증되지는 않습니다.

### 11.3 세 문장 예측

코드는 `"gpu memory"`, `"network connection"`, `"disk full"`을 순회합니다. 학습 문장과 완전히 같지는 않지만, 포함된 단어는 모두 기존 단어 목록에 있습니다.

```python
sample = torch.tensor(
    [encode(text)], dtype=torch.float32, device=device
)
prediction = model(sample).argmax(dim=1).item()
print(f"{text} -> {labels[prediction]}")
```

- 같은 `encode`와 `vocab`을 사용하므로 학습 때와 같은 열 순서입니다.
- `[encode(text)]`로 한 번 감싸 문장 하나인 **묶음 형태**를 유지합니다. 구조는 `[1, 13]`입니다.
- `model(sample)`의 세 점수에서 번호를 고르고, `labels`에서 이름을 찾습니다.
- 여기에 `backward()`와 `step()`은 없습니다. **예측만 하고 추가 학습하지 않습니다.**

3절 실행 예시처럼 문장 옆에 분류 이름이 표시되는지 확인합니다. 세 문장이 맞는 것만으로 폭넓은 새 문의에 대한 성능이 입증되지는 않습니다.


## 12. 마지막 `assert`의 성공 조건은 좁다

```python
assert final_loss < initial_loss, "학습 후 오차가 줄지 않았습니다."
print("\n검증 통과: 학습 후 오차 감소")
```

첫 줄은 **최종 학습 loss가 처음보다 작은지** 확인합니다. 거짓이면 오류를 발생시키고, 참이면 다음 성공 문구로 넘어갑니다.

이 검사는 다음을 확인하지 않습니다.

- 모든 epoch에서 loss가 감소했는가?
- 정확도가 어떤 기준 이상인가?
- 세 새 문장의 분류가 맞는가?
- 처음 보는 실제 고객 문의에서도 잘 작동하는가?
- 모델을 저장하고 다시 읽을 수 있는가?

문구의 “검증 통과”는 **코드 안의 loss 감소 검사**를 뜻합니다. 별도 validation 데이터로 성능 평가를 끝냈다는 뜻이 아닙니다.

## 13. 무엇을 학습하고, 무엇은 학습하지 않을까?

| 요소 | 정한 주체 | 이 코드의 학습으로 바뀌나? |
|---|---|---|
| 문의 문장·정답 | 사람이 작성 | 아니요 |
| `labels` | 사람이 작성 | 아니요 |
| `vocab` | 코드가 데이터에서 추출 | 아니요 |
| `encode` | 사람이 정한 횟수 세기 규칙 | 아니요 |
| 학습률·반복 횟수 | 사람이 지정 | 아니요 |
| `Linear`의 가중치·편향 | 초기 생성 후 옵티마이저가 수정 | **예** |

현재 변환 방식은 단어 순서를 사용하지 않고, `vocab`에 없는 단어는 반영하지 않습니다. 모르는 문의에 “모르겠다”고 답하는 로직도 없습니다. 모든 단어가 어휘 밖이라면 입력 벡터는 모두 0이 되지만, 모델은 편향으로 점수를 계산하고 여전히 한 분류를 선택합니다. 그 출력을 이해도나 확신으로 해석하면 안 됩니다.

3절 코드에는 **모델 저장 명령이 없습니다.** 실행을 끝내면 학습된 파라미터를 다음 Python 실행에서 사용할 수 없으므로, 다음 절에서 새 모델을 학습하고 저장하는 파일을 만듭니다. [03장](03-inference-and-data.md)에서는 저장 모델 복원·새 입력의 오류 분석·데이터 보완을, [04장](04-evaluation.md)에서는 검증·테스트의 역할 분리와 모델 비교를 다룹니다. 04장도 이미 관찰한 검증 문장과 공개된 테스트 문장을 사용하므로 엄격한 미관측·블라인드 평가는 아닙니다.


## 14. 학습 코드를 파일로 보관하고 모델 저장하기

### 14.1 파일 위치 정하기

표준 입력 실행은 코드를 바로 시험하기 편하지만 학습된 가중치를 자동 저장하지 않습니다. 저장 버전은 학습을 수행한 뒤 체크포인트를 기록합니다. 아래 코드를 `~/llm-lab/source/train_classifier.py`에 저장합니다. 같은 이름의 기존 파일이 있다면 내용을 먼저 확인하고 임의로 덮어쓰지 않습니다.

[저장 기능이 포함된 전체 소스: train_classifier.py](source/02-first-classifier/train_classifier.py)

**배치 위치가 중요합니다.** 이 코드는 `__file__`을 기준으로 모델 경로를 정합니다. 아래 위치에 두면 모델 파일은 `~/llm-lab/classifier.pt`입니다. 저장소의 장별 하위 폴더에서 바로 실행하면 저장 위치가 달라지므로, 이 가이드의 VM 실습 폴더에 배치합니다. `__file__`이 필요한 코드이므로 here-document로 실행하지 않습니다.

| VM에서의 위치 | 역할 |
|---|---|
| `~/llm-lab/source/train_classifier.py` | 저장 버전 코드를 배치하고 파일로 실행 |
| `~/llm-lab/classifier.pt` | 새 학습 결과로 생성할 체크포인트; 기존 파일은 보호 |

이미 필요한 체크포인트가 있다면 재실행을 위해 삭제하지 말고 [15절의 경로 점검](#모델-파일이-목록에-보이지-않을-때)과 복원 준비를 확인합니다.

```python
from pathlib import Path

import torch
from torch import nn

torch.manual_seed(42)
assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
device = "cuda"

# 1. 이전 실습과 동일한 학습 데이터
labels = ["GPU", "네트워크", "스토리지"]
data = [
    ("gpu memory error", 0),
    ("gpu device unavailable", 0),
    ("cuda memory unavailable", 0),
    ("network connection error", 1),
    ("network timeout", 1),
    ("connection timeout", 1),
    ("disk storage full", 2),
    ("disk space error", 2),
    ("storage space full", 2),
]

vocab = sorted({word for text, _ in data for word in text.split()})


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


# 2. 학습 데이터를 GPU 텐서로 변환
x = torch.tensor(
    [encode(text) for text, _ in data],
    dtype=torch.float32,
    device=device,
)
y = torch.tensor(
    [label for _, label in data],
    dtype=torch.long,
    device=device,
)

# 기존 모델 파일이 있으면 덮어쓰지 않음
save_path = Path(__file__).resolve().parent.parent / "classifier.pt"
if save_path.exists():
    raise FileExistsError(f"기존 모델을 보호하기 위해 중단합니다: {save_path}")

# 3. 모델과 학습 방법 준비
model = nn.Linear(len(vocab), len(labels)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

initial_loss = loss_fn(model(x), y).item()
print(f"학습 전 loss: {initial_loss:.4f}")

# 4. 학습
model.train()
for epoch in range(1, 101):
    scores = model(x)
    loss = loss_fn(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | loss: {loss.item():.4f}")

# 5. 학습 결과 확인
model.eval()
with torch.no_grad():
    final_loss = loss_fn(model(x), y).item()
    predictions = model(x).argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    print(f"\n학습 데이터 정확도: {accuracy:.1%}")

    for text in ["gpu memory", "network connection", "disk full"]:
        inputs = torch.tensor(
            [encode(text)], dtype=torch.float32, device=device
        )
        prediction = model(inputs).argmax(dim=1).item()
        print(f"{text} -> {labels[prediction]}")

assert final_loss < initial_loss, "학습 후 오차가 줄지 않았습니다."
print("\n검증 통과: 학습 후 오차 감소")

# 6. 가중치와 문장 해석에 필요한 정보를 함께 저장
checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "labels": labels,
}
with save_path.open("xb") as file:
    torch.save(checkpoint, file)

assert save_path.stat().st_size > 0, "모델 파일이 비어 있습니다."
print(f"모델 저장 완료: {save_path}")
```

### 14.2 학습 파일 실행하기

파일을 저장한 뒤 다음 명령을 실행합니다. 이 명령은 새로 학습하고 체크포인트를 생성합니다. 기존 체크포인트가 있으면 보호 검사에서 오류를 발생시킵니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python train_classifier.py
```

### 14.3 저장을 위해 추가된 부분

| 코드 | 의미 |
|---|---|
| `Path(__file__).resolve().parent.parent` | 코드 파일의 부모 폴더에서 다시 한 단계 위, 즉 안내한 배치의 `~/llm-lab` |
| `save_path.exists()` | 이미 모델이 있으면 재학습 전에 오류를 발생시켜 보호 |
| `model.state_dict()` | 학습한 가중치·편향을 이름과 함께 가져오기 |
| `vocab` | 추론 때도 숫자 벡터의 열 순서를 똑같이 유지 |
| `labels` | 출력 번호와 분류 이름의 대응 유지 |
| `save_path.open("xb")` | 파일이 없을 때만 새 바이너리 파일 생성. 있으면 실패 |
| `torch.save(checkpoint, file)` | 가중치·어휘·분류 정보를 디스크에 기록 |
| `stat().st_size > 0` | 저장 파일이 비어 있지 않은지 검사. 복원 성공까지 증명하는 검사는 아님 |

### 14.4 저장 버전 실행 예시

아래는 **실행 예시**이며 수치와 홈 경로는 환경에 따라 달라집니다.

**표시 시점 주의:** 여기의 epoch loss는 가중치 수정 전 값입니다. 3절은 수정 후 다시 계산한 값을 표시하므로 같은 epoch 줄끼리 바로 비교하지 않습니다.

```text
학습 전 loss: 1.2353
epoch  20 | loss: 0.6970
epoch  40 | loss: 0.4266
epoch  60 | loss: 0.2924
epoch  80 | loss: 0.2176
epoch 100 | loss: 0.1714

학습 데이터 정확도: 100.0%
gpu memory -> GPU
network connection -> 네트워크
disk full -> 스토리지

검증 통과: 학습 후 오차 감소
모델 저장 완료: /home/<계정>/llm-lab/classifier.pt
```

**왜 3절의 마지막 표시값 `0.1695`와 여기의 `0.1714`가 다를까요?** 첫 코드는 `step()` 이후 다시 계산한 `current_loss`를 표시하고, 저장 버전은 해당 반복에서 수정하기 전에 계산한 `loss.item()`을 표시합니다. 같은 epoch 표기라도 계산 시점이 다릅니다. 저장 버전도 학습 루프 뒤에 `final_loss`를 새로 계산해 처음 값과 비교하지만, 그 값을 epoch 줄에 출력하지는 않습니다. 표시값의 차이를 곧바로 오류나 성능 악화로 해석하지 않습니다.

파일 크기가 0보다 크다는 검사는 저장 결과가 비어 있지 않다는 조건만 확인합니다. 가중치가 올바르게 복원되고 같은 전처리로 예측되는지는 별도 Python 실행에서 검사해야 합니다.

## 15. 체크포인트와 복원 준비

### `.py`와 `.pt`, `.pth`의 차이

- `train_classifier.py`는 **학습하는 방법을 적은 코드**입니다.
- `classifier.pt`는 **학습된 가중치와 해석에 필요한 정보를 저장한 데이터**입니다.
- 이 체크포인트에는 모델 구조 전체나 학습 코드가 들어 있지 않습니다. 복원할 때 같은 `nn.Linear` 구조를 먼저 만듭니다.
- `.pt`와 `.pth`는 파일 이름 관례입니다. `.pt`는 모델 전체, `.pth`는 가중치 전용이라는 구분은 맞지 않습니다. 실제 내용은 저장 코드를 확인합니다.
- `weights_only=True`여도 출처를 신뢰할 수 없는 체크포인트를 읽지 않습니다.

이 파일은 추론 복원에 필요한 정보를 담지만, optimizer·난수·학습 진행 위치까지 포함한 완전한 학습 재개 상태는 아닙니다.

전체 복원 코드는 [다음 장 첫 실습](03-inference-and-data.md)에 있습니다. 저장된 어휘와 분류 목록으로 같은 구조를 만들고, 가중치를 넣은 뒤 CPU에서 예측합니다. **저장된 모델을 읽는 것은 다시 학습하는 과정이 아닙니다.** 다음은 복원 성공 시 읽게 될 **실행 예시**입니다. 경로와 결과는 실행 환경과 저장한 모델에 따라 달라질 수 있습니다.

```text
모델 불러오기 완료: /home/<계정>/llm-lab/classifier.pt
gpu memory -> GPU
network connection -> 네트워크
disk full -> 스토리지

검증 통과: 저장한 모델을 재학습 없이 불러와 예측 성공
```

### 모델 파일이 목록에 보이지 않을 때

안내한 위치에 학습 코드를 저장했다면 폴더 구조는 다음과 같습니다.

```text
~/llm-lab/
├── classifier.pt
└── source/
    ├── .venv/
    └── train_classifier.py
```

`~/llm-lab/source`에서 `ls`를 실행하면 상위 폴더의 모델 파일은 표시되지 않습니다. 현재 위치와 저장 경로를 비교합니다.

```bash
pwd
ls -lh ../classifier.pt
```

`..`는 상위 폴더, `-l`은 상세 목록, `-h`는 읽기 쉬운 크기 표시입니다. 파일이 없다면 학습 출력의 저장 경로, 스크립트 배치 위치, 저장 전 발생한 오류를 확인합니다. 다른 경로에 있는 파일을 찾기 전에 무조건 재학습하지 않습니다.

## 16. 문제 해결과 실습 완료 조건

| 증상 | 먼저 확인할 것 |
|---|---|
| `No module named 'torch'` | `.venv` 활성화와 `python -m pip --version` 경로 |
| GPU 사용 검사 오류 | 앞 장의 GPU 검사와 현재 Python의 CUDA 지원 |
| 텐서와 모델의 장치 불일치 | `x`, `y`, 모델, 예측 입력이 같은 장치인지 |
| `CrossEntropyLoss` 자료형 오류 | 입력 `float32`, 분류 번호 정답 `long`인지 |
| `FileExistsError` | 기존 체크포인트 보호 동작. 필요한 파일인지 확인하고 별도 보관·새 경로 사용을 결정 |
| loss 감소 검사 오류 | 데이터·정답 순서, 학습률, `zero_grad()`·`backward()`·`step()` 흐름 |
| 단어를 바꾸자 엉뚱한 분류 출력 | 어휘 밖 단어가 무시되는지, 단어 순서가 사라지는지 |

기존 체크포인트를 지워 오류를 숨기지 않습니다. 저장 파일이 존재하는 것, 비어 있지 않은 것, 복원해서 예측할 수 있는 것, 새 문의에서 성능이 좋은 것은 서로 다른 조건입니다.

**단계별 확인 범위**

- 이 장 첫 실행: 최종 학습 loss가 처음보다 작은지 검사합니다. 정확도나 새 문장 정답을 보장하는 검사는 아닙니다.
- 이 장 저장 버전: 위 loss 검사 후 파일을 새로 저장하고 비어 있지 않은지 검사합니다. 복원 검사는 아직 아닙니다.
- 03장: 별도 프로세스 복원·추론·오류 분석과 데이터 보완으로 이어집니다.
- 04장: 검증 기준으로 후보를 고르고 테스트하되, 작은 공개 예제의 평가 한계를 함께 다룹니다.

## 17. 복습 질문과 체크리스트

**스스로 답해 보기**

1. `encode` 규칙과 `Linear` 가중치 중 학습으로 바뀌는 것은 무엇인가?
2. `vocab` 순서가 학습과 예측 사이에 달라지면 왜 문제가 되는가?
3. 입력 `x`는 `float32`, 정답 `y`는 `long`인 이유는 무엇인가?
4. `backward()`만 호출하고 `step()`을 하지 않으면 무엇이 빠지는가?
5. `zero_grad()`는 가중치를 초기화하는가, 기울기를 초기화하는가?
6. `model.eval()`과 `torch.no_grad()`는 무엇이 다른가?
7. 학습 데이터 정확도가 높다고 새 문의 성능도 높다고 말할 수 있는가?
8. 마지막 “검증 통과”가 뜻하는 범위는 정확히 어디까지인가?
9. 가중치뿐 아니라 `vocab`과 `labels`도 저장하는 이유는 무엇인가?
10. 저장 버전의 epoch 100 표시값과 최종 검사에 쓰는 loss는 왜 시점이 다른가?

**정리 체크리스트**

- [ ] Bash 래퍼와 Python 본문을 구분한다.
- [ ] 문장→단어 횟수→텐서→세 점수의 흐름을 설명한다.
- [ ] 예측·loss·기울기·가중치 수정 시점을 각각 짚을 수 있다.
- [ ] 이 모델이 분류기이며 LLM은 아님을 안다.
- [ ] 단어 순서·모르는 단어·보류 응답의 한계를 설명한다.
- [ ] 학습 전후 loss를 비교하고 학습 데이터 정확도의 한계를 설명한다.
- [ ] 코드 파일과 체크포인트를 구분하고 저장 경로를 확인한다.
- [ ] 별도 복원 검사와 독립 데이터 평가가 추가로 필요함을 안다.

[← 이전 장: 환경과 기초](01-foundations.md) · [목차](README.md) · [다음 장: 추론과 데이터 보완 →](03-inference-and-data.md)
