# 03 · 추론과 데이터 보완 — 오답을 설명하고 비교하기

[← 이전: 첫 분류기와 저장·복원](02-first-classifier.md) · [목차](README.md) · [다음: 평가 데이터 분리 →](04-evaluation.md)

> [!NOTE]
> 저장된 모델로 추론하고, 데이터 한 건을 보완한 모델과 비교합니다.
> 아래 수치와 출력은 실행 예시이며 동일한 결과를 보장하지 않습니다.

## 🎯 이 장의 목표

- 저장된 체크포인트를 CPU로 복원하고, **학습 없이** 새 입력을 분류한다.
- 미등록 단어와 단어 순서가 입력 숫자에서 어떻게 사라지는지 확인한다.
- `network unavailable`의 오분류를 실제 가중치·편향으로 설명한다.
- 예제를 하나 추가했을 때 **학습 예제 교정**과 **새 입력의 개선**을 구분한다.

### 실습 흐름과 파일

**절 바로가기:** [복원](#1-저장한-모델을-불러오면-무엇이-복원되는가) · [새 입력 시험](#2-새로운-조합-모르는-단어-단어-순서-시험) · [기여도 분석](#3-오답의-이유를-추측에서-실제-수치로-확인하기) · [데이터 보완](#4-데이터-보완--구조를-바꾸지-않고-한-예제만-추가) · [두 모델 비교](#5-같은-문장을-두-모델에-넣어-비교하기) · [한계](#6-한계와-다음-장으로-넘길-기준)

```text
classifier.pt → 새 조합·미등록 단어·순서 시험 → 단어별 기여도 분석
             → 예제 한 개 추가·별도 학습 → classifier_augmented.pt
             → 같은 문장으로 두 모델 비교
```

| 위치 | 역할 |
|---|---|
| `/home/ubuntu/llm-lab/source/` | Python 코드 실행 위치 |
| `/home/ubuntu/llm-lab/source/.venv/` | 이전 단계에서 만든 가상환경 |
| `/home/ubuntu/llm-lab/source/train_classifier.py` | 기존 학습 코드; 이전 장에서 준비 |
| `/home/ubuntu/llm-lab/classifier.pt` | 원본 모델·어휘·라벨; 보존 대상 |
| `/home/ubuntu/llm-lab/classifier_augmented.pt` | 보완 데이터를 처음부터 학습한 별도 모델 |

이 문서의 경로 예시는 홈 디렉터리가 `/home/ubuntu`인 Linux 환경을 기준으로 합니다. 코드의 `Path.home()`은 실행 계정의 홈을 사용하므로, 다른 계정에서는 경로도 달라집니다. `source`는 코드 폴더 이름이고, `source .venv/bin/activate`의 `source`는 셸에서 파일을 읽는 명령입니다. 같은 글자지만 역할이 다릅니다.

```bash
cd /home/ubuntu/llm-lab/source
# 가상환경이 꺼져 있을 때만, 기존 환경을 활성화하는 명령
source .venv/bin/activate
```

> [!NOTE]
> 아래 Python 코드 전문은 링크된 `.py` 파일과 같습니다. 실행 명령은 해당 파일을 `/home/ubuntu/llm-lab/source/`에 배치한 경우를 기준으로 합니다. 체크포인트는 이전 장에서 준비해야 합니다.

## 1. 저장한 모델을 불러오면 무엇이 복원되는가?

| 개념·코드 | 이 실습에서의 의미 |
|---|---|
| `checkpoint` | `model_state_dict`, `vocab`, `labels`를 담은 딕셔너리 |
| `vocab` | 단어를 입력 벡터의 어느 칸에 넣을지 정하는 순서 있는 목록 |
| `labels` | 출력 번호를 `GPU`, `네트워크`, `스토리지`로 읽는 순서 |
| `nn.Linear(len(vocab), len(labels))` | 어휘 크기만큼 입력받아 세 점수를 내는 동일한 구조를 생성 |
| `load_state_dict(...)` | 임의 초기값 대신 저장된 가중치·편향을 넣음 |
| `map_location="cpu"` | GPU에서 저장한 모델도 이 실습에서는 CPU로 읽음 |
| `model.eval()` | 평가 모드로 전환; **기울기 계산 중단과는 별개** |
| `torch.no_grad()` | 이 블록에서는 학습용 기울기 기록을 하지 않음 |
| `argmax` | 점수가 가장 큰 출력 번호 선택; 확률을 출력하는 명령이 아님 |

체크포인트에는 모델 구조 코드 전체가 없으므로 같은 구조를 먼저 만들어야 합니다. `vocab` 순서를 바꾸면 같은 단어가 다른 입력 칸으로 들어가므로 가중치만 맞아도 해석이 틀어집니다. `.pt`와 `.pth`는 이 실습에서는 저장 내용의 종류를 구분하는 규칙이 아니라 파일명 관례입니다.

### 1-1. 복원 결과를 검사하는 전체 코드

파일: [`restore_classifier.py`](source/03-inference-and-data/restore_classifier.py)

실행 위치는 `/home/ubuntu/llm-lab/source`이며, 명령은 `python restore_classifier.py`입니다. 이 코드는 체크포인트를 읽기만 하고 모델 파일을 저장하거나 덮어쓰지 않습니다.

```python
from pathlib import Path

import torch
from torch import nn

# 1. 저장된 모델 파일 읽기
save_path = Path.home() / "llm-lab" / "classifier.pt"
checkpoint = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True,
)

# 2. 학습에 사용한 단어 목록과 분류 이름 복원
vocab = checkpoint["vocab"]
labels = checkpoint["labels"]

# 3. 같은 구조의 모델을 만들고 학습된 가중치 넣기
model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"모델 불러오기 완료: {save_path}")


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


# 4. 재학습 없이 이전과 같은 문의를 분류
examples = [
    ("gpu memory", "GPU"),
    ("network connection", "네트워크"),
    ("disk full", "스토리지"),
]

with torch.no_grad():
    for text, expected in examples:
        inputs = torch.tensor([encode(text)], dtype=torch.float32)
        prediction = model(inputs).argmax(dim=1).item()
        result = labels[prediction]

        print(f"{text} -> {result}")
        assert result == expected, f"예상과 다른 결과: {text} -> {result}"

print("\n검증 통과: 저장한 모델을 재학습 없이 불러와 예측 성공")
```

**코드 읽기:** `Path.home()`으로 홈 아래 체크포인트를 찾고 → 어휘·라벨을 읽고 → 같은 선형층을 만들고 → 가중치를 복원합니다. `examples`의 두 번째 값은 검사할 의도이며 모델 입력에 섞이지 않습니다. `assert result == expected`는 이 세 예시의 복원 결과를 확인할 뿐, 전체 일반화 성능 검사가 아닙니다.

### 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
모델 불러오기 완료: /home/ubuntu/llm-lab/classifier.pt
gpu memory -> GPU
network connection -> 네트워크
disk full -> 스토리지

검증 통과: 저장한 모델을 재학습 없이 불러와 예측 성공
```

예시에서는 세 문장이 기대한 분류와 일치합니다. 이 검사가 확인하는 범위는 **저장 → 별도 실행에서 복원 → 예측** 흐름의 성공입니다. `loss.backward()`, `optimizer.step()`과 학습 반복문이 없으므로 재학습하지 않습니다. `weights_only=True`가 있다고 해서 출처 불명의 체크포인트를 안전하다고 간주하지 않습니다. 직접 저장했거나 신뢰할 수 있는 체크포인트만 읽으세요.

## 2. 새로운 조합, 모르는 단어, 단어 순서 시험

### 전체 코드

파일: [`probe_classifier.py`](source/03-inference-and-data/probe_classifier.py)

파일을 실행 위치에 배치한 뒤 사용하는 명령:

```bash
cd /home/ubuntu/llm-lab/source
python probe_classifier.py
```

```python
from pathlib import Path

import torch
from torch import nn

# 1. 이전 실습에서 직접 저장한 모델 복원
save_path = Path.home() / "llm-lab" / "classifier.pt"
checkpoint = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True,
)

vocab = checkpoint["vocab"]
labels = checkpoint["labels"]

model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


print("모델 불러오기 완료")
print("모델이 아는 단어:", ", ".join(vocab))

# 2. 학습 문장과 다른 조합으로 예측
examples = [
    ("cuda error", "GPU"),
    ("network unavailable", "네트워크"),
    ("disk full error", "스토리지"),
]

print("\n[1] 새로운 문장 예측")

with torch.no_grad():
    for text, expected in examples:
        inputs = torch.tensor([encode(text)], dtype=torch.float32)
        result = labels[model(inputs).argmax(dim=1).item()]
        mark = "일치" if result == expected else "불일치"
        print(f"{text} -> {result} | 의도: {expected} | {mark}")

    # 3. 단어 목록에 없는 문장
    text = "internet disconnected"
    vector = encode(text)
    unknown = [word for word in text.split() if word not in vocab]
    inputs = torch.tensor([vector], dtype=torch.float32)
    result = labels[model(inputs).argmax(dim=1).item()]

    print("\n[2] 모르는 단어 시험")
    print("입력:", text)
    print("모르는 단어:", unknown)
    print("변환된 숫자:", vector)
    print("모델의 예측:", result)
    assert not any(vector), "예상과 달리 등록된 단어가 있습니다."

    # 4. 단어 순서만 바꿔 비교
    first = encode("gpu memory")
    second = encode("memory gpu")

    print("\n[3] 단어 순서 시험")
    print("gpu memory:", first)
    print("memory gpu:", second)
    print("두 입력의 숫자가 같은가?:", first == second)
    assert first == second, "예상과 달리 입력 표현이 다릅니다."

print("\n검사 완료: 미등록 단어와 단어 순서의 처리 방식 확인")
```

### 코드 구간별 설명

| 구간 | Python을 처음 읽을 때 볼 점 |
|---|---|
| `encode(text)` | `lower()`로 소문자화, `split()`으로 공백 분리, `count()`로 어휘별 개수 계산. 리스트의 순서는 `vocab` 순서 |
| `examples` 반복문 | `(문장, 의도)` 쌍을 하나씩 꺼내 예측과 비교; 틀려도 출력하고 계속 진행 |
| `[word for ... if word not in vocab]` | 어휘 밖 단어만 새 리스트로 모음 |
| `not any(vector)` | 하나라도 0이 아닌 값이 있는지 확인한 뒤 부정; 전부 0이라는 검사 |
| `first == second` | 두 Python 리스트의 길이·각 원소가 같은지 검사 |

이 표현을 **단어 등장 횟수 기반 bag-of-words**라고 읽으면 됩니다. 문장의 뜻·순서를 그대로 전달하는 것이 아닙니다. 기존 단어의 새 조합과 어휘 밖 단어를 구분해 시험합니다.

### 실행 예시 — 성공 사례와 오답

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
모델 불러오기 완료
모델이 아는 단어: connection, cuda, device, disk, error, full, gpu, memory, network, space, storage, timeout, unavailable

[1] 새로운 문장 예측
cuda error -> GPU | 의도: GPU | 일치
network unavailable -> GPU | 의도: 네트워크 | 불일치
disk full error -> 스토리지 | 의도: 스토리지 | 일치

[2] 모르는 단어 시험
입력: internet disconnected
모르는 단어: ['internet', 'disconnected']
변환된 숫자: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
모델의 예측: 네트워크

[3] 단어 순서 시험
gpu memory: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
memory gpu: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
두 입력의 숫자가 같은가?: True

검사 완료: 미등록 단어와 단어 순서의 처리 방식 확인
```

| 관찰 | 해석 | 해석하면 안 되는 것 |
|---|---|---|
| `network unavailable → GPU` | 학습 데이터 100% 정확도와 새 조합의 정확도는 다름 | 모델이 모든 네트워크 문장을 이해했다 |
| `internet disconnected → 영벡터 → 네트워크` | 아는 단어가 전혀 없어 입력 기여가 사라지고 편향으로 선택 | `internet`의 의미를 알고 정답을 맞혔다 |
| `gpu memory == memory gpu` | 순서가 다른 두 문장이 같은 숫자가 됨 | 학습 횟수만 늘리면 이 표현에서 순서를 구분할 수 있다 |

선형층은 **분류별 점수 = 단어별 가중치 × 등장 횟수의 합 + 편향**으로 계산합니다. 입력이 전부 0이면 가중치 항이 없어져 편향만 남습니다. 같은 영벡터가 되는 다른 입력도 이 모델에서는 같은 결과가 됩니다.

## 3. 오답의 이유를 추측에서 실제 수치로 확인하기

`unavailable`이 GPU 쪽 학습 예제에 등장한다는 사실만으로 그 가중치를 단정할 수는 없습니다. 다음 코드는 저장된 가중치와 편향을 읽고, 직접 계산한 점수와 모델 출력을 비교합니다.

### 전체 코드

파일: [`explain_classifier.py`](source/03-inference-and-data/explain_classifier.py)

```bash
cd /home/ubuntu/llm-lab/source
python explain_classifier.py
```

```python
from pathlib import Path

import torch
from torch import nn

# 1. 저장된 모델 복원
save_path = Path.home() / "llm-lab" / "classifier.pt"
checkpoint = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True,
)

vocab = checkpoint["vocab"]
labels = checkpoint["labels"]

model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. 이전에 오분류한 문장을 같은 방식으로 숫자 변환
text = "network unavailable"
words = text.lower().split()
inputs = torch.tensor(
    [words.count(word) for word in vocab],
    dtype=torch.float32,
)

with torch.no_grad():
    # 분류별·단어별 기여도 = 학습된 가중치 × 단어 등장 횟수
    contributions = model.weight * inputs

    # 분류별 최종 점수 = 모든 단어의 기여도 합 + 편향
    calculated_scores = contributions.sum(dim=1) + model.bias
    model_scores = model(inputs)

    print("분석할 문장:", text)
    print("분류 순서:", " | ".join(labels))
    print("\n[1] 각 단어가 분류별 점수에 더하는 값")

    for index, word in enumerate(vocab):
        if inputs[index].item() > 0:
            values = contributions[:, index]
            formatted = " | ".join(f"{value.item():+.4f}" for value in values)
            print(f"{word}: {formatted}")

    print("\n[2] 편향과 최종 점수")

    for index, label in enumerate(labels):
        word_total = contributions[index].sum().item()
        bias = model.bias[index].item()
        total = calculated_scores[index].item()
        print(
            f"{label}: 단어 합 {word_total:+.4f}"
            f" + 편향 {bias:+.4f}"
            f" = 최종 {total:+.4f}"
        )

    prediction = model_scores.argmax().item()
    print("\n최종 예측:", labels[prediction])

    # 반올림해 출력하기 전의 실제 숫자로 검사
    assert torch.allclose(
        calculated_scores,
        model_scores,
        rtol=1e-5,
        atol=1e-6,
    ), "직접 합산한 점수와 모델 출력이 다릅니다."

print("\n검증 통과: 단어별 기여도 + 편향 = 모델 출력")
```

### 코드 구간별 설명

- `inputs`는 문장 하나의 1차원 벡터입니다. 앞선 코드의 `[encode(text)]`는 문장 하나를 배치로 감싼 2차원 입력이었습니다. 이 선형층은 두 형태로 계산할 수 있습니다.
- `model.weight`는 **분류별 행 × 단어별 열**입니다. `model.weight * inputs`는 각 열의 가중치에 단어 개수를 곱해 기여도를 구합니다.
- `sum(dim=1)`은 각 분류 행에서 모든 단어 기여도를 더합니다. `model.bias`를 더하면 세 분류의 최종 점수가 됩니다.
- `contributions[:, index]`는 그 단어의 열에서 모든 분류 값을 가져옵니다. 출력하지 않은 단어는 등장 횟수가 0이라 기여도도 0입니다.
- `enumerate()`는 순서 번호와 원소를 함께 꺼냅니다. `:+.4f`는 부호를 표시하고 소수 넷째 자리까지 출력하는 형식입니다.
- `torch.allclose()`는 반올림 전 텐서가 허용 오차 안에서 같은지 검사합니다. 부동소수점 계산은 화면에 보이는 소수 네 자리보다 더 많은 자릿수를 씁니다.

### 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
분석할 문장: network unavailable
분류 순서: GPU | 네트워크 | 스토리지

[1] 각 단어가 분류별 점수에 더하는 값
network: -0.1798 | +0.7552 | -0.4535
unavailable: +0.8278 | -0.4399 | -0.2806

[2] 편향과 최종 점수
GPU: 단어 합 +0.6480 + 편향 -0.1527 = 최종 +0.4953
네트워크: 단어 합 +0.3154 + 편향 +0.1693 = 최종 +0.4847
스토리지: 단어 합 -0.7341 + 편향 -0.1048 = 최종 -0.8388

최종 예측: GPU

검증 통과: 단어별 기여도 + 편향 = 모델 출력
```

이 예시에서 `network`는 네트워크 점수를 올리고 GPU 점수를 낮춥니다. 반대로 `unavailable`은 GPU 점수를 올리고 네트워크 점수를 낮춥니다. 편향은 네트워크에 유리하지만 최종 점수는 GPU가 더 높습니다.

> [!TIP]
> **가장 높은 점수**와 **의미를 이해함**은 다릅니다. `+0.4953`은 GPU일 확률 49.53%가 아닙니다. 출력은 확률 변환 전의 점수입니다. 단어별 반올림 값의 합계가 마지막 자리에서 다르게 보일 수 있지만, 코드의 검사는 반올림 전 정밀도의 숫자로 수행합니다.

## 4. 데이터 보완 — 구조를 바꾸지 않고 한 예제만 추가

이 장의 “데이터 보완/augmentation”은 자동으로 대량의 변형 문장을 만드는 작업이 아닙니다. **직접 라벨을 붙인 `("network unavailable", 1)` 한 건을 기존 9건에 추가**합니다. 라벨 번호 `1`은 `labels[1]`, 즉 네트워크입니다.

| 유지한 조건 | 바꾼 조건 |
|---|---|
| 선형층 구조·기존 예제·어휘 순서 | 네트워크 예제 1건 추가: 총 10건 |
| seed 42·SGD·학습률 0.1·100 epoch | 저장명 `classifier.pt → classifier_augmented.pt` |
| 기존 코드·기존 모델 보존 | 새 코드·새 모델 파일 생성 |

**기존 가중치를 이어서 학습한 것이 아닙니다.** 같은 조건으로 새 모델을 만들고 처음부터 학습합니다. `network`와 `unavailable`은 이미 어휘에 있으므로 새 단어를 추가하는 실습도 아닙니다.

### 4-1. 보완 학습 파일을 만드는 전체 코드

이전 장의 `train_classifier.py`가 실행 위치에 있어야 합니다. 아래 코드는 원본의 두 문자열을 검사하고, 예제와 저장 파일명만 변경한 새 학습 파일을 만듭니다.

파일: [`make_augmented.py`](source/03-inference-and-data/make_augmented.py)

```bash
cd /home/ubuntu/llm-lab/source
python make_augmented.py
```

```python
from pathlib import Path

original = Path("train_classifier.py")
target = Path("train_classifier_augmented.py")
source = original.read_text(encoding="utf-8")

changes = [
    (
        '    ("connection timeout", 1),',
        '    ("connection timeout", 1),\n'
        '    ("network unavailable", 1),',
    ),
    (
        '/ "classifier.pt"',
        '/ "classifier_augmented.pt"',
    ),
]

for old, new in changes:
    assert source.count(old) == 1, f"변경 위치를 확인하세요: {old}"
    source = source.replace(old, new, 1)

# 파일을 만들기 전에 Python 문법 검사
compile(source, str(target), "exec")

# 기존 파일이 있으면 덮어쓰지 않음
with target.open("x", encoding="utf-8") as file:
    file.write(source)

print(f"생성 완료: {target.resolve()}")
print("원본 코드와 classifier.pt는 변경하지 않았습니다.")
```

**왜 문자열을 검사하는가?** `source.count(old) == 1`은 변경할 위치가 정확히 하나인지 확인합니다. 원본이 달라졌는데 엉뚱한 부분을 바꾸지 않도록 한 장치입니다. `replace(..., 1)`은 한 곳만 바꿉니다. `compile()`은 생성할 코드의 문법만 검사하며 학습을 실행하지 않습니다. `open("x")`는 새 파일만 생성하고 기존 파일이 있으면 중단합니다.

### 4-2. 변환 결과의 전체 학습 코드

파일: [`train_classifier_augmented.py`](source/03-inference-and-data/train_classifier_augmented.py). 앞의 변환 코드가 만드는 전체 학습 코드입니다.

> [!WARNING]
> **아래 코드는 CUDA GPU로 새 모델을 학습하고 파일을 생성합니다.**
>
> - **배치·실행:** `/home/ubuntu/llm-lab/source/train_classifier_augmented.py`에 배치한 뒤 `python train_classifier_augmented.py`로 실행합니다.
> - **저장 경로:** `__file__`의 상위 두 단계에 있는 `classifier_augmented.pt`입니다. 예시 배치에서는 `/home/ubuntu/llm-lab/classifier_augmented.pt`가 생성됩니다.
> - **중첩 경로 주의:** 저장소의 중첩된 `source/03-inference-and-data/`에서 직접 실행하면 저장 경로가 달라집니다.
> - **기존 파일 보호:** 같은 모델이 있으면 덮어쓰지 않고 중단하며, `classifier.pt`는 변경하지 않습니다. 기존 파일을 삭제하거나 보호 장치를 제거하기 전에 경로와 보존 필요성을 확인하세요.

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
    ("network unavailable", 1),
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
save_path = Path(__file__).resolve().parent.parent / "classifier_augmented.pt"
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

### 학습 코드 구간별 설명

| 구간 | 하는 일 | 주의점 |
|---|---|---|
| seed·CUDA 검사 | 초기화 난수 기준 고정, GPU 사용 가능 여부 검사 | CPU 추론과 달리 이 학습 코드는 CUDA가 필요 |
| `labels`, `data`, `vocab` | 문장·정답을 정의하고 정렬된 어휘 생성 | 숫자 라벨과 이름 순서가 대응해야 함 |
| `x`, `y` 텐서 | 입력은 실수, 정답은 정수 번호로 GPU에 배치 | `x`는 10개 문장 행, `y`는 10개 정답 |
| 저장 경로 검사 | 학습 전에 같은 모델 파일이 있으면 중단 | 기존 모델 보호 |
| `nn.Linear`, loss, SGD | 분류 점수 계산 방식·오차·가중치 수정 방법 준비 | 문장을 생성하는 LLM이 아님 |
| 100 epoch 반복 | 예측 → loss → 기존 기울기 초기화 → 역전파 → 가중치 갱신 | `zero_grad()` 없이 기울기를 계속 더하지 않도록 함 |
| `eval()`·`no_grad()` | 학습 데이터 loss·정확도와 세 예시 확인 | 별도 검증 세트 평가가 아님 |
| `assert final_loss < initial_loss` | 학습 전보다 loss 감소했는지 확인 | 처음 보는 문장의 품질을 보장하지 않음 |
| `checkpoint`·`open("xb")` | 가중치·어휘·라벨을 새 바이너리 파일에 저장 | `x`는 배타 생성, `b`는 바이너리 모드 |

`epoch`는 학습 데이터 전체를 한 번 사용하는 반복 단위입니다. 여기서는 작은 전체 데이터를 한 번에 처리합니다. 20회마다 표시하는 `loss`는 해당 반복의 가중치 갱신 **전** 계산값이며, 반복 뒤의 `final_loss`와 같은 시점의 값은 아닙니다.

### 학습 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
학습 전 loss: 1.2376
epoch  20 | loss: 0.7375
epoch  40 | loss: 0.4717
epoch  60 | loss: 0.3326
epoch  80 | loss: 0.2521
epoch 100 | loss: 0.2011

학습 데이터 정확도: 100.0%
gpu memory -> GPU
network connection -> 네트워크
disk full -> 스토리지

검증 통과: 학습 후 오차 감소
모델 저장 완료: /home/ubuntu/llm-lab/classifier_augmented.pt
```

예시에는 학습 데이터 정확도 100%, loss 감소와 별도 경로 저장이 나타납니다. 기존 원본 모델의 loss와 이 학습 loss는 **학습 예제 수가 다르므로** 수치만 놓고 우열을 정하는 평가가 아닙니다. 다음 단계에서는 학습에 추가한 문장과 학습에 없는 문장의 결과를 구분합니다.

## 5. 같은 문장을 두 모델에 넣어 비교하기

### 전체 코드

파일: [`compare_classifiers.py`](source/03-inference-and-data/compare_classifiers.py)

```bash
cd /home/ubuntu/llm-lab/source
python compare_classifiers.py
```

```python
from pathlib import Path

import torch
from torch import nn

root = Path.home() / "llm-lab"
models = []

for filename in ["classifier.pt", "classifier_augmented.pt"]:
    checkpoint = torch.load(
        root / filename, map_location="cpu", weights_only=True
    )
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]

    model = nn.Linear(len(vocab), len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append((model, vocab, labels))

assert models[0][1:] == models[1][1:], (
    "두 모델의 어휘 또는 분류 순서가 다릅니다."
)

groups = {
    "기존 학습 문장": [
        ("gpu memory error", "GPU"),
        ("gpu device unavailable", "GPU"),
        ("cuda memory unavailable", "GPU"),
        ("network connection error", "네트워크"),
        ("network timeout", "네트워크"),
        ("connection timeout", "네트워크"),
        ("disk storage full", "스토리지"),
        ("disk space error", "스토리지"),
        ("storage space full", "스토리지"),
    ],
    "추가한 학습 문장": [
        ("network unavailable", "네트워크"),
    ],
    "두 모델의 학습 데이터에 없는 문장": [
        ("network error", "네트워크"),
        ("connection unavailable", "네트워크"),
        ("disk unavailable", "스토리지"),
        ("cuda error", "GPU"),
    ],
}

with torch.no_grad():
    for group, examples in groups.items():
        print(f"\n[{group}]")
        correct = [0, 0]

        for text, expected in examples:
            predictions = []

            for index, (model, vocab, labels) in enumerate(models):
                words = text.lower().split()
                inputs = torch.tensor(
                    [words.count(word) for word in vocab],
                    dtype=torch.float32,
                )
                prediction = labels[model(inputs).argmax().item()]
                predictions.append(prediction)
                correct[index] += int(prediction == expected)

            print(
                f"{text}: {predictions[0]} -> {predictions[1]}"
                f" | 의도: {expected}"
            )

        print(
            f"의도와 일치: 보완 전 {correct[0]}/{len(examples)}"
            f" | 보완 후 {correct[1]}/{len(examples)}"
        )

print("\n비교 완료: 두 모델 파일은 읽기만 했습니다.")
```

### 코드 구간별 설명

1. 두 체크포인트를 CPU로 읽어 `(모델, 어휘, 라벨)` 튜플을 리스트에 넣습니다. `models[0][1:] == models[1][1:]`는 어휘와 분류 순서가 같은지 검사합니다.
2. `groups` 딕셔너리에 **기존 학습 문장 / 추가 학습 문장 / 두 모델의 학습에 없는 문장**을 나눠 넣습니다. 학습에 넣은 문장의 교정을 새 문장 일반화라고 부르지 않기 위한 구분입니다.
3. 모델별로 같은 문장을 숫자로 바꿔 예측합니다. `int(prediction == expected)`는 일치하면 1, 아니면 0을 더합니다.
4. `predictions[0] -> predictions[1]`은 **원본 예측 → 보완 예측**입니다. 출력의 화살표는 학습 중 변화나 문장 생성 과정이 아닙니다.
5. 이 코드는 최종 분류와 일치 개수만 비교합니다. 예측이 같아도 가중치나 점수가 같다는 뜻이 아닙니다.

### 비교 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
[기존 학습 문장]
gpu memory error: GPU -> GPU | 의도: GPU
gpu device unavailable: GPU -> GPU | 의도: GPU
cuda memory unavailable: GPU -> GPU | 의도: GPU
network connection error: 네트워크 -> 네트워크 | 의도: 네트워크
network timeout: 네트워크 -> 네트워크 | 의도: 네트워크
connection timeout: 네트워크 -> 네트워크 | 의도: 네트워크
disk storage full: 스토리지 -> 스토리지 | 의도: 스토리지
disk space error: 스토리지 -> 스토리지 | 의도: 스토리지
storage space full: 스토리지 -> 스토리지 | 의도: 스토리지
의도와 일치: 보완 전 9/9 | 보완 후 9/9

[추가한 학습 문장]
network unavailable: GPU -> 네트워크 | 의도: 네트워크
의도와 일치: 보완 전 0/1 | 보완 후 1/1

[두 모델의 학습 데이터에 없는 문장]
network error: 네트워크 -> 네트워크 | 의도: 네트워크
connection unavailable: 네트워크 -> 네트워크 | 의도: 네트워크
disk unavailable: GPU -> GPU | 의도: 스토리지
cuda error: GPU -> GPU | 의도: GPU
의도와 일치: 보완 전 3/4 | 보완 후 3/4

비교 완료: 두 모델 파일은 읽기만 했습니다.
```

| 비교 대상 | 원본 → 보완 | 내릴 수 있는 결론 |
|---|---|---|
| 기존 학습 9문장 | 9/9 → 9/9 | 확인한 기존 예제에서는 정답 유지 |
| 추가한 `network unavailable` | GPU → 네트워크 | 학습에 추가한 오답 한 건 교정 |
| 학습에 없는 관찰 4문장 | 3/4 → 3/4 | 이 관찰에서는 최종 정답 수 변화 없음 |
| `disk unavailable` | GPU → GPU | 스토리지로 분류하지 못하는 오류가 그대로 남음 |

## 6. 한계와 다음 장으로 넘길 기준

> [!CAUTION]
> **“학습에 없는 문장”이라고 해서 자동으로 깨끗한 테스트 세트가 되지 않습니다.** `cuda error`는 이전 시험에서 이미 봤고, 비교 결과와 오답은 개선 방향에 영향을 줍니다. 이 네 문장은 작은 **관찰용 예제**이며, 이후 장에서는 검증용으로 지정합니다. 반복해서 본 예제를 마지막 독립 평가의 근거처럼 재사용하지 않습니다.

- 단어 빈도 표현은 순서를 버립니다. epoch 증가만으로 이미 사라진 순서 정보를 되살릴 수 없습니다.
- 모르는 단어는 입력에서 무시됩니다. 모든 단어가 모르면 편향에 따른 강제 선택입니다. “모름”을 내는 기능은 없습니다.
- 예제 하나를 추가해 한 오답이 고쳐져도 비슷한 오답 전체가 해결된 것은 아닙니다.
- `disk unavailable`을 학습에 추가하면 그 순간부터 해당 문장을 새 입력 평가라고 부를 수 없습니다. 위 코드에서는 이 문장을 학습 데이터에 추가하지 않습니다.
- 다음 장은 새 학습이 아니라 **이미 있는 두 모델을 검증 데이터로 비교하고 선택 후 테스트하는 과정**입니다.

## 🧠 복습 질문

1. 저장 가중치는 같은데 `vocab` 순서를 바꾸면 왜 안 될까요?
2. `internet disconnected`를 네트워크로 예측했어도 의미를 이해했다고 할 수 없는 이유는?
3. `model.eval()`과 `torch.no_grad()`는 각각 무엇을 담당하나요?
4. `network unavailable`을 교정한 결과는 왜 일반화 성능 향상 증거로 충분하지 않을까요?
5. 예측 라벨이 같으면 점수와 loss도 반드시 같을까요?

<details>
<summary>답을 확인하기</summary>

1. 입력 열과 가중치 열의 단어 대응이 달라집니다.
2. 모든 단어가 어휘 밖이라 영벡터가 됐고, 편향으로 선택했습니다.
3. 전자는 평가 모드, 후자는 기울기 기록 중단입니다. 이 단순 선형층에는 dropout 같은 모드별 층이 없지만 두 역할은 구분합니다.
4. 그 문장 자체를 학습에 넣었습니다. 별도의 미사용 입력으로 평가해야 합니다.
5. 아닙니다. 가장 큰 항목의 번호는 같아도 점수의 분포는 달라질 수 있습니다.

</details>

## 선택 사항: 모델을 실행하지 않는 코드 검사

아래 검사는 표준 라이브러리만 사용합니다. 두 장에 실린 Python 코드 전문 9개와 연결된 소스 파일의 일치, Python 문법, 보완 파일 변환 결과, train 10 / validation 4 / test 3 구성, 덮어쓰기 방지를 확인합니다. 모델을 로드하거나 PyTorch를 실행하지 않습니다. 임시 폴더의 보호용 파일은 파일 보존 검사에만 쓰며 모델 출력이 아닙니다.

파일: [`verify_examples.py`](source/03-inference-and-data/verify_examples.py). 저장소 루트에서 실행하는 선택적 오프라인 검사입니다.

```bash
python source/03-inference-and-data/verify_examples.py
```

검사 출력 예시:

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
PASS: 9 examples scripts compile and match the full chapter code blocks.
PASS: augmentation round-trip, train 10 / validation 4 / test 3, exclusive writes.
NOT RUN: PyTorch model loading, inference, training, or VM commands.
```

이 검사는 저장 모델의 예측이나 위 실행 예시의 수치를 검증하지 않습니다.

[← 이전](02-first-classifier.md) · [목차](README.md) · [다음: 평가 데이터 분리 →](04-evaluation.md)
