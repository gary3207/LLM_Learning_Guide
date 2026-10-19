# 04 · 평가 분리 — 학습·검증·테스트의 역할을 고정하기

[← 이전: 추론과 데이터 보완](03-inference-and-data.md) · [목차](README.md) · [다음: 다음 토큰 예측 →](05-next-token.md)

> [!NOTE]
> 학습이 끝난 두 모델을 비교하고, 검증 결과로 선택한 모델만 테스트합니다. 평가 코드에는 재학습 과정이 없습니다.
> 아래 수치와 출력은 실행 예시이며 동일한 결과를 보장하지 않습니다.

## 🎯 이 장의 목표

- 모델을 바꾸는 데이터와 모델을 고르는 데이터, 마지막 평가 데이터를 구분한다.
- 기존 학습 코드를 **실행하지 않고** 데이터 목록만 읽어 분리 파일을 만든다.
- 정확도가 같을 때 loss가 달라질 수 있음을 실제 두 모델로 확인한다.
- 검증 결과로 모델을 선택한 뒤 **선택한 한 모델만** 마지막 테스트한다.
- 테스트 100%가 왜 모든 오류의 해결이나 실전 100% 정확도를 뜻하지 않는지 설명한다.

## 1. 실습 순서: 무작위 분할 대신 역할 지정

```text
이미 학습한 원본·보완 모델
    ↓ 기존 학습 예제는 유지
train 10 / validation 4 / test 3 파일 생성
    ↓ 검증 정확도 우선, 동률이면 검증 loss
보완 모델 선택
    ↓ 테스트에서는 후보를 다시 고르지 않음
test 3문장 최종 평가
```

| 구분 | 구성 | 사용 목적 | 평가 단계에서 가중치 변경? |
|---|---|---|---|
| `train` 10개 | 보완 모델의 기존 학습 데이터 | 어떤 예제로 학습됐는지 보존 | 아니요; 학습은 앞 장에서 끝남 |
| `validation` 4개 | 앞 장에서 예측을 관찰한 네 문장 | 후보 비교·모델 선택 | 아니요 |
| `test` 3개 | 앞선 비교에 넣지 않은 알려진 단어의 새 조합 | 선택을 마친 모델의 마지막 평가 | 아니요 |

원본 모델은 9건, 보완 모델은 10건으로 학습된 모델입니다. JSON의 `train` 10건이 원본 모델도 10건으로 학습했다는 뜻은 아닙니다. 이 실습은 작은 기존 예제를 무작위 비율로 나누고 재학습하는 과정이 아닙니다.

### 누수와 독립성: 평가 설계의 한계

> [!CAUTION]
> 검증용 네 문장은 분리 파일 생성 전에 앞 장의 비교 코드에서 두 모델의 예측을 관찰하는 데 사용됩니다. `validation`이라는 이름을 붙여도 이미 관찰한 결과가 사라지지는 않습니다. 이들은 개선·선택 과정에 참고하는 **검증 데이터**이며, 엄격한 미관측 검증 세트나 독립적인 최종 테스트가 아닙니다.

- 학습·검증·테스트 사이에 **정규화한 문장 문자열이 중복되지 않는지** 검사합니다. 이것은 정확히 같은 문장의 중복을 막는 검사이지 의미상 유사성이나 정보 누수를 모두 검출하는 검사가 아닙니다.
- 모든 평가 문장은 학습 어휘만 사용합니다. 새 단어 처리나 실제 긴 장애 문장 성능을 보장하지 않습니다.
- 테스트 문장은 분리 생성 코드에서 미리 공개됩니다. 따라서 엄격한 블라인드 평가라고 부르지 않습니다. 검증 결과로 모델을 선택한 뒤 보완 모델만 테스트하며, 테스트 예측으로 후보를 다시 고르지 않습니다.
- 테스트 결과를 본 뒤 마음에 드는 다른 모델로 바꾸면 테스트까지 선택에 사용한 셈입니다. 이후 개선에 이 테스트 결과를 참고한다면 더 이상 완전히 손대지 않은 마지막 평가로 반복 주장할 수 없습니다.

## 2. 실행 위치와 전제

실행 위치 예시는 `/home/ubuntu/llm-lab/source`입니다. 홈 디렉터리가 `/home/ubuntu`인 Linux 환경을 기준으로 하며, `Path.home()`은 실행 계정의 홈을 사용합니다. 다른 계정에서는 모델과 데이터 경로도 그 홈 아래에 맞추세요.

```bash
cd /home/ubuntu/llm-lab/source
# 기존 가상환경이 비활성 상태일 때만
source .venv/bin/activate
```

| 필요 파일 | 읽는 이유 |
|---|---|
| `source/train_classifier_augmented.py` | 기존 `data` 값을 읽음; 파일을 실행하지 않음 |
| `classifier.pt` | 원본 모델 검증 |
| `classifier_augmented.pt` | 보완 모델 검증 및 선택 후 테스트 |
| `source/classifier_splits.json` | 이 단계에서 생성하는 역할별 데이터 |

아래 Python 코드 전문과 링크된 `.py` 파일의 내용은 같습니다. `python 파일명.py` 명령은 해당 파일을 실행 위치에 배치한 경우를 기준으로 합니다. 이전 장의 두 체크포인트와 `train_classifier_augmented.py`를 먼저 준비해야 합니다. 저장소를 받는 것만으로 체크포인트가 생성되지는 않습니다.

`create_splits.py`는 현재 작업 디렉터리의 `train_classifier_augmented.py`를 읽고 `classifier_splits.json`을 생성합니다. 예시 경로는 `/home/ubuntu/llm-lab/source/classifier_splits.json`이며, 파일이 이미 있으면 덮어쓰지 않고 중단합니다. 나머지 평가 코드는 모델과 JSON을 읽기만 합니다.

## 3. 코드를 실행하지 않고 학습 데이터 읽기·분리 파일 생성

### 전체 코드

파일: [`create_splits.py`](source/04-evaluation/create_splits.py)

```bash
cd /home/ubuntu/llm-lab/source
python create_splits.py
```

```python
import ast
import json
from pathlib import Path

# 학습 파일을 실행하지 않고 data 값만 읽습니다.
tree = ast.parse(Path("train_classifier_augmented.py").read_text(encoding="utf-8"))
assignments = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "data" for t in node.targets)]
assert len(assignments) == 1, "data 정의를 확인하세요."
train = ast.literal_eval(assignments[0].value)

splits = {
    "train": train,
    "validation": [
        ("network error", 1),
        ("connection unavailable", 1),
        ("disk unavailable", 2),
        ("cuda error", 0),
    ],
    "test": [
        ("cuda device error", 0),
        ("network connection unavailable", 1),
        ("disk space full", 2),
    ],
}

seen = set()
train_vocab = {word for text, _ in train for word in text.split()}
for name, examples in splits.items():
    assert examples, f"빈 데이터: {name}"
    for text, label in examples:
        key = " ".join(text.lower().split())
        assert key and key not in seen, f"빈 문장 또는 중복 문장: {text}"
        assert type(label) is int and label in (0, 1, 2), f"잘못된 라벨: {label}"
        assert set(key.split()) <= train_vocab, f"학습 어휘에 없는 단어: {text}"
        seen.add(key)

payload = {"labels": ["GPU", "네트워크", "스토리지"], **splits}
with Path("classifier_splits.json").open("x", encoding="utf-8") as file:
    json.dump(payload, file, ensure_ascii=False, indent=2)

for name, examples in splits.items():
    print(f"{name}: {len(examples)}개")
print("검사 통과: 문장 중복 없음, 라벨 유효, 평가 문장은 학습 어휘만 사용")
print("저장 완료:", Path("classifier_splits.json").resolve())
```

### 코드 구간별 설명

| 코드 구간 | 의미 |
|---|---|
| `ast.parse(...)` | Python 파일을 문법 구조인 AST로 읽음. `import`하거나 학습 파일을 실행하지 않음 |
| `tree.body`·`ast.Assign` | 파일 최상위의 대입문 중 이름이 `data`인 대상을 찾음 |
| `assert len(assignments) == 1` | 데이터 정의가 정확히 한 곳인지 확인 |
| `ast.literal_eval(...)` | 목록·튜플·문자열·정수 같은 리터럴 값만 읽음. 임의 코드를 실행하는 `eval()`과 구분 |
| `splits` | 이미 학습한 10건과 명시적으로 정한 검증·테스트 문장을 역할별로 묶음 |
| `" ".join(text.lower().split())` | 대소문자와 여러 공백을 정규화한 비교 키 |
| `seen` 집합 | 한 분리 안뿐 아니라 모든 분리를 통틀어 동일한 키가 두 번 나오면 중단 |
| `type(label) is int` | 숫자 라벨이 정확히 정수인지 확인; 허용 값은 0·1·2 |
| `set(key.split()) <= train_vocab` | 평가 단어가 학습 어휘의 부분집합인지 확인 |
| `{"labels": ..., **splits}` | `splits`의 키·값을 같은 딕셔너리 안으로 펼쳐 저장 |
| `ensure_ascii=False`, `indent=2` | 한글 라벨을 그대로 쓰고 들여쓰기로 JSON을 읽기 쉽게 함 |
| `open("x")` | 파일이 이미 있으면 덮어쓰지 않고 중단 |

`create_splits.py`는 표준 라이브러리만 쓰므로 데이터 분리 단계 자체는 GPU나 PyTorch 연산이 필요하지 않습니다. 모델을 실행하지 않고 데이터 정의만 읽는 것이 목적입니다.

### 데이터 생성 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
train: 10개
validation: 4개
test: 3개
검사 통과: 문장 중복 없음, 라벨 유효, 평가 문장은 학습 어휘만 사용
저장 완료: /home/ubuntu/llm-lab/source/classifier_splits.json
```

### 고정한 평가 문장

| 역할 | 문장 | 정답 |
|---|---|---|
| 검증 | `network error` | 네트워크 |
| 검증 | `connection unavailable` | 네트워크 |
| 검증 | `disk unavailable` | 스토리지 |
| 검증 | `cuda error` | GPU |
| 테스트 | `cuda device error` | GPU |
| 테스트 | `network connection unavailable` | 네트워크 |
| 테스트 | `disk space full` | 스토리지 |

새로운 “조합”이지만 새 단어는 없습니다. 검증에는 앞 장의 실행 예시에서 오분류되는 `disk unavailable`도 포함합니다. 평가에서 오답을 제외하면 모델의 한계를 파악하기 어렵습니다.

## 4. 두 후보를 검증 데이터로만 비교

### 검증 결과에 적용할 선택 기준

1. 검증 정확도가 더 높은 모델을 우선한다.
2. 정확도가 같으면 **같은 검증 데이터·같은 계산 방식의 loss**가 더 낮은 모델을 선택한다.
3. 테스트 예측은 이 비교 단계에서 실행하지 않는다.

### 전체 코드

파일: [`validate_classifiers.py`](source/04-evaluation/validate_classifiers.py)

```bash
cd /home/ubuntu/llm-lab/source
python validate_classifiers.py
```

```python
import json
from pathlib import Path

import torch
from torch import nn

root = Path.home() / "llm-lab"
dataset = json.loads(
    (root / "source/classifier_splits.json").read_text(encoding="utf-8")
)
examples = dataset["validation"]
assert examples, "검증 데이터가 비어 있습니다."

for filename in ["classifier.pt", "classifier_augmented.pt"]:
    checkpoint = torch.load(
        root / filename, map_location="cpu", weights_only=True
    )
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]
    assert labels == dataset["labels"], "라벨 순서가 다릅니다."

    model = nn.Linear(len(vocab), len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor(
        [[text.lower().split().count(word) for word in vocab]
         for text, _ in examples],
        dtype=torch.float32,
    )
    y = torch.tensor([label for _, label in examples], dtype=torch.long)

    with torch.no_grad():
        scores = model(x)
        loss = nn.functional.cross_entropy(scores, y).item()
        predictions = scores.argmax(dim=1)
        correct = (predictions == y).sum().item()

    print(f"\n[{filename} / validation]")
    for (text, expected), prediction in zip(examples, predictions.tolist()):
        print(f"{text}: {labels[prediction]} | 의도: {labels[expected]}")

    print(
        f"검증 정확도: {correct}/{len(examples)} "
        f"({correct / len(examples):.1%})"
    )
    print(f"검증 loss: {loss:.4f}")

print("\n검증 완료: 재학습·모델 저장·테스트 예측은 실행하지 않았습니다.")
```

### 코드 구간별 설명

- `json.loads(...read_text(...))`는 분리 파일을 Python 자료형으로 읽습니다. `dataset["validation"]`만 평가 대상으로 고릅니다. JSON 파일에는 테스트 문장도 있지만 이 코드는 테스트를 추론하지 않습니다.
- `for filename in [...]`으로 두 후보를 각각 복원합니다. `labels == dataset["labels"]`는 숫자 정답을 같은 라벨 순서로 해석하는지 확인합니다.
- 중첩 리스트 내포의 바깥 반복은 문장마다 한 행을 만들고, 안쪽 반복은 `vocab`의 각 단어 개수를 셉니다. `x`는 실수 텐서, `y`는 정답 번호 정수 텐서입니다.
- `scores = model(x)`는 각 문장에 대해 세 점수를 만듭니다. `cross_entropy(scores, y)`는 그 점수와 정답으로 기본 평균 loss를 구합니다. 이 코드에서는 점수에 softmax를 먼저 적용하지 않습니다.
- `argmax(dim=1)`은 각 문장 행에서 가장 큰 분류 번호를 택합니다. `(predictions == y).sum().item()`은 정답 개수를 Python 값으로 가져옵니다.
- `zip(...)`은 문장·정답과 예측을 같은 순서로 묶어 출력합니다. `predictions.tolist()`는 텐서를 Python 리스트로 바꿉니다.
- 학습용 optimizer·역전파·저장이 없고, 두 모델 파일과 JSON은 읽기만 합니다.

### 검증 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
[classifier.pt / validation]
network error: 네트워크 | 의도: 네트워크
connection unavailable: 네트워크 | 의도: 네트워크
disk unavailable: GPU | 의도: 스토리지
cuda error: GPU | 의도: GPU
검증 정확도: 3/4 (75.0%)
검증 loss: 0.8041

[classifier_augmented.pt / validation]
network error: 네트워크 | 의도: 네트워크
connection unavailable: 네트워크 | 의도: 네트워크
disk unavailable: GPU | 의도: 스토리지
cuda error: GPU | 의도: GPU
검증 정확도: 3/4 (75.0%)
검증 loss: 0.7280

검증 완료: 재학습·모델 저장·테스트 예측은 실행하지 않았습니다.
```

### 선택: 보완 모델

| 모델 | 검증 정확도 | 검증 loss | 결정 |
|---|---:|---:|---|
| `classifier.pt` | 3/4 · 75.0% | 0.8041 | 비교 후보 |
| `classifier_augmented.pt` | 3/4 · 75.0% | **0.7280** | 사전 동률 기준에 따라 선택 |

**정확도는 최종 번호만 보지만 loss는 정답에 부여된 상대적인 점수도 반영**하므로, 네 문장의 예측이 모두 같아도 평균 loss는 달라질 수 있습니다. 이 예시에서 보완 모델의 평균 검증 loss가 더 낮다는 뜻이지, 각 문장의 정답 점수가 모두 개선됐다는 뜻은 아닙니다.

예시에서 `disk unavailable`은 두 모델 모두 GPU로 오분류합니다. 모델 선택은 이 오류의 해결을 의미하지 않습니다. 검증 문장 네 개는 작고 이미 관찰한 예제이므로, 이 차이를 실제 서비스 전반의 유의미한 개선으로 확대 해석하지 않습니다.

## 5. 선택을 끝낸 모델만 마지막 테스트

### 전체 코드

파일: [`test_classifier.py`](source/04-evaluation/test_classifier.py)

```bash
cd /home/ubuntu/llm-lab/source
python test_classifier.py
```

```python
import json
from pathlib import Path

import torch
from torch import nn

root = Path.home() / "llm-lab"
dataset = json.loads(
    (root / "source/classifier_splits.json").read_text(encoding="utf-8")
)
examples = dataset["test"]
assert examples, "테스트 데이터가 비어 있습니다."

filename = "classifier_augmented.pt"
checkpoint = torch.load(
    root / filename, map_location="cpu", weights_only=True
)
vocab = checkpoint["vocab"]
labels = checkpoint["labels"]
assert labels == dataset["labels"], "라벨 순서가 다릅니다."

model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

x = torch.tensor(
    [[text.lower().split().count(word) for word in vocab]
     for text, _ in examples],
    dtype=torch.float32,
)
y = torch.tensor([label for _, label in examples], dtype=torch.long)

with torch.no_grad():
    scores = model(x)
    loss = nn.functional.cross_entropy(scores, y).item()
    predictions = scores.argmax(dim=1)
    correct = (predictions == y).sum().item()

print(f"\n[{filename} / test]")
for (text, expected), prediction in zip(examples, predictions.tolist()):
    print(f"{text}: {labels[prediction]} | 의도: {labels[expected]}")

print(
    f"테스트 정확도: {correct}/{len(examples)} "
    f"({correct / len(examples):.1%})"
)
print(f"테스트 loss: {loss:.4f}")
print("\n최종 평가 완료: 재학습하거나 모델 파일을 변경하지 않았습니다.")
```

### 앞 코드와 달라진 부분

| 변경 | 이유 |
|---|---|
| `dataset["test"]` | 검증 대신 남겨 둔 세 문장을 평가 |
| `filename = "classifier_augmented.pt"` | 선택한 모델 하나로 고정; 테스트로 후보 재선택하지 않음 |
| 후보 비교 반복문 없음 | 테스트 결과를 보고 원본으로 바꾸는 경로를 넣지 않음 |
| 출력 제목 `test`, 테스트 정확도·loss | 모델 선택용 검증 수치와 마지막 평가 수치를 구분 |

나머지는 동일합니다. 체크포인트 CPU 복원 → 어휘 순서대로 숫자화 → `eval()`·`no_grad()` 평가 → 정확도·평균 loss 출력입니다. **그 사이 재학습이나 가중치 수정은 없습니다.**

### 테스트 실행 예시

다음은 실행 예시입니다. 환경/버전에 따라 달라질 수 있습니다.

```text
[classifier_augmented.pt / test]
cuda device error: GPU | 의도: GPU
network connection unavailable: 네트워크 | 의도: 네트워크
disk space full: 스토리지 | 의도: 스토리지
테스트 정확도: 3/3 (100.0%)
테스트 loss: 0.2959

최종 평가 완료: 재학습하거나 모델 파일을 변경하지 않았습니다.
```

## 6. 최종 결과를 과장 없이 읽기

| 구분 | 정확도 | loss | 실행 예시의 해석 |
|---|---:|---:|---|
| 원본 모델 검증 | 3/4 · 75.0% | 0.8041 | 네 검증 문장 중 세 개 일치 |
| 보완 모델 검증 | 3/4 · 75.0% | 0.7280 | 동률 기준인 검증 loss에서 선택 |
| 선택한 보완 모델 테스트 | 3/3 · 100.0% | 0.2959 | 남겨 둔 세 조합 모두 일치 |

> [!WARNING]
> **테스트 3/3은 “이 세 문장을 맞혔다”는 결과입니다. 모든 장애 문장에서 정확도 100%라는 뜻이 아닙니다.** 검증의 `disk unavailable` 오분류는 여전히 미해결입니다. 테스트 통과가 이 미해결 오류를 덮어쓰지 않습니다.

### 혼동하기 쉬운 해석

| 잘못 읽기 | 올바른 해석 |
|---|---|
| `0.7280 → 0.2959`로 loss가 줄었으니 추가 학습 성공 | 서로 **다른 문장 집합**의 loss이며 사이에 학습이 없었음 |
| 테스트가 100%니까 검증보다 더 좋은 모델이 됨 | 모델은 같은 보완 모델. 평가 문장이 달라 결과가 달랐음 |
| 원본보다 보완의 테스트 성능이 높음 | 원본 테스트 결과는 이 단계에서 측정하지 않음; 비교 근거 없음 |
| 정확도가 같으니 학습 보완은 아무 변화가 없음 | 최종 예측은 같아도 평균 검증 loss는 다름 |
| JSON에 train 10개가 있으니 두 모델 모두 10개 학습 | 원본 9개·보완 10개; JSON은 보완 학습 예제를 보존 |
| 문장 중복이 없으니 모든 누수 문제 해결 | 문자열 중복만 검사함. 사전 관찰·유사 예제·작은 규모 한계는 남음 |

### 다음 단계와 구분하기

- 이 장의 흐름은 학습이 끝난 모델의 검증 비교 → 모델 선택 → 선택한 모델의 테스트입니다. 평가 사이에 가중치를 변경하지 않습니다.
- 다음 장에서는 다음 토큰 예측을 다룹니다. 이 장의 분류 결과는 문장 생성 능력을 입증하지 않습니다.

## 🧠 복습 질문

1. 학습·검증·테스트는 각각 어느 결정에 사용하나요?
2. 관찰했던 네 문장을 왜 테스트가 아닌 검증으로 두었나요?
3. 예측이 전부 같아도 평균 loss가 다를 수 있는 이유는?
4. `ast.literal_eval()`로 데이터를 읽는 것이 학습 파일 실행과 어떻게 다른가요?
5. `disk unavailable`이 틀렸는데 테스트 100%가 나오는 것이 모순인가요?
6. 테스트 결과를 보고 원본 모델로 바꾼 뒤 그 결과를 최종 평가라고 부르면 무엇이 달라지나요?

<details>
<summary>답을 확인하기</summary>

1. 학습은 가중치 수정, 검증은 개선·후보 선택, 테스트는 선택 후 마지막 평가입니다.
2. 이미 결과를 봤고 개선에 영향을 주는 예제라서 독립적인 마지막 평가처럼 쓰지 않기 위해서입니다.
3. 최댓값의 번호는 같아도 정답과 다른 분류 사이의 점수 관계가 달라질 수 있습니다.
4. AST를 읽고 리터럴 값만 추출하며, GPU 검사·학습 반복·모델 저장 코드는 실행하지 않습니다.
5. 아닙니다. 그 문장은 검증에 있고 테스트 세 문장에는 없습니다. 테스트 성공은 모든 입력의 정답을 보장하지 않습니다.
6. 테스트가 모델 선택에 참여하므로 더 이상 선택 과정에서 손대지 않은 평가가 아닙니다.

</details>

[← 이전](03-inference-and-data.md) · [목차](README.md) · [다음: 다음 토큰 예측 →](05-next-token.md)
