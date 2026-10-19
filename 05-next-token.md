# 05 · 다음 토큰 예측 — 한 번의 예측을 문장 생성으로

[← 이전: 평가 실습](04-evaluation.md) · [학습 목차](README.md) · [다음: 문맥 확장 →](06-context-window.md)

현재 토큰 하나로 다음 토큰을 예측하는 작은 모델을 만들고, 그 예측을 연결해 문장을 생성합니다. 앞 장에서 사용한 Python·PyTorch 가상환경이 활성화된 터미널에서 진행합니다. 예제의 작업 폴더는 `~/llm-lab/source`입니다.

## 이 장의 목표

- 문장 분류와 다음 토큰 예측의 **정답이 어떻게 다른지** 설명한다.
- 공백 단위 토큰 → 정수 ID → 점수 → 정답 비교 → 가중치 갱신을 따라간다.
- 저장된 가중치를 복원하고, 예측을 다음 입력에 넣어 문장을 만드는 원리를 이해한다.
- **학습 데이터 재현**과 **새 데이터 일반화**를 구분한다.

| 순서 | 실행할 내용 | 확인할 내용 |
|---|---|---|
| 1 | 다음 토큰 입력·정답 9개 준비 | 문장 경계를 넘는 쌍이 없는지 확인 |
| 2 | CPU에서 학습하고 `next_token.pt` 저장 | 학습 쌍 예측과 loss 확인 |
| 3 | 복원 후 gpu·network·disk로 생성 | 문장 재현, 종료 토큰·길이 제한 확인 |

아래 출력은 **실행 예시**입니다. 학습 loss와 생성 결과는 실행 환경·난수 초기화·저장된 가중치에 따라 달라질 수 있습니다. 같은 seed를 사용해도 모든 환경에서 숫자가 완전히 같다고 보장하지 않습니다. 저장 경로의 홈 디렉터리도 환경에 맞게 달라집니다.

## 먼저, 무엇을 예측하나요?

분류기는 `gpu memory error`를 보고 `GPU`처럼 **문장 전체의 종류 하나**를 골랐습니다. 이번에는 현재 토큰에서 **다음 토큰의 ID 하나**를 고릅니다.

```text
문장 분류:       gpu memory error → GPU
다음 토큰 예측:  gpu → memory → error → <eos>
```

이번 토큰화는 `split()`으로 공백을 나누는 입문용 방식입니다. `gpu`, `memory`, `error`가 각각 토큰이고, `<eos>`는 **문장이 끝났음**을 뜻하는 특별 토큰입니다. 일반 서비스의 토크나이저를 구현한 것은 아닙니다.

첫 모델은 문장 전체를 읽지 않습니다. `gpu memory`까지 생성했더라도 **마지막 `memory` 하나만** 다음 입력으로 사용합니다. 먼저 이 단순한 흐름을 완성한 다음, 다음 장에서 한계를 확인합니다.

## 1단계 · 문장별로 입력·정답 쌍 만들기

### 전체 코드 실행하기

먼저 데이터 준비만 실행합니다. 이 코드는 라이브러리를 설치하거나 모델을 학습하지 않으며, 파일도 저장하지 않습니다. 아래 heredoc은 `PY` 사이의 Python 본문을 실행합니다.

**소스:** [01-pairs.py](source/05-next-token/01-pairs.py)

```bash
cd ~/llm-lab/source

python - <<'PY'
sentences = [
    "gpu memory error",
    "network connection error",
    "disk storage full",
]

pairs = []
for sentence in sentences:
    tokens = sentence.split() + ["<eos>"]
    pairs.extend(zip(tokens[:-1], tokens[1:]))

assert len(pairs) == 9
assert pairs[0] == ("gpu", "memory")
assert sum(target == "<eos>" for _, target in pairs) == len(sentences)
assert ("<eos>", "network") not in pairs

for current, target in pairs:
    print(f"{current:>10} -> {target}")
print(f"학습 쌍: {len(pairs)}개")
print("검사 통과: 문장별 종료 토큰 포함, 문장 사이 연결 없음")
PY
```

### 코드 읽기

| 블록 | 실제 하는 일 |
|---|---|
| `sentences` | 학습에 사용할 세 문장을 직접 지정합니다. |
| `split() + ["<eos>"]` | 단어를 나누고 문장 끝을 붙입니다. |
| `zip(tokens[:-1], tokens[1:])` | 같은 문장의 현재 위치와 한 칸 뒤를 짝짓습니다. |
| `pairs.extend(...)` | 세 문장에서 만든 쌍을 하나의 목록에 모읍니다. |
| `assert ...` | 쌍 수·첫 쌍·종료 수·문장 경계를 확인합니다. |

예를 들어 `['gpu', 'memory', 'error', '<eos>']`에서 앞쪽 목록은 `['gpu', 'memory', 'error']`, 뒤쪽 목록은 `['memory', 'error', '<eos>']`입니다. `zip`은 같은 위치끼리 묶습니다. **문장마다 따로 만들기 때문에 `<eos> → network` 같은 가짜 연결을 넣지 않습니다.**

### 실행 예시

```text
       gpu -> memory
    memory -> error
     error -> <eos>
   network -> connection
connection -> error
     error -> <eos>
      disk -> storage
   storage -> full
      full -> <eos>
학습 쌍: 9개
검사 통과: 문장별 종료 토큰 포함, 문장 사이 연결 없음
```

## 2단계 · 정수 ID로 바꾸고 점수표 학습하기

### 전체 코드 실행하기

**소스:** [02-train.py](source/05-next-token/02-train.py)

> [!WARNING]
> 실행하면 CPU에서 학습하고 `~/llm-lab/next_token.pt`를 저장합니다. 기존 파일이 있으면 덮어쓰지 않고 중단합니다. 같은 경로의 체크포인트가 있다면 먼저 다음 단계에서 복원해 확인합니다. 새로 학습할 필요가 있다면 기존 파일을 별도로 백업하고 새 저장 경로를 정한 뒤, 생성 코드의 로드 경로도 맞춥니다. 보호 검사를 제거하지 않습니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
from pathlib import Path
import torch
from torch import nn

output = Path.home() / "llm-lab/next_token.pt"
if output.exists():
    raise FileExistsError(f"기존 파일을 보호하기 위해 중단: {output}")

torch.manual_seed(42)
sentences = [
    "gpu memory error",
    "network connection error",
    "disk storage full",
]

pairs = []
for sentence in sentences:
    tokens = sentence.split() + ["<eos>"]
    pairs.extend(zip(tokens[:-1], tokens[1:]))

vocab = sorted({token for pair in pairs for token in pair})
to_id = {token: index for index, token in enumerate(vocab)}
assert len(pairs) == 9 and len(vocab) == 9
assert all(to_id[vocab[i]] == i for i in range(len(vocab)))
print("토큰 ID:", to_id)

x = torch.tensor([to_id[current] for current, _ in pairs], dtype=torch.long)
y = torch.tensor([to_id[target] for _, target in pairs], dtype=torch.long)

# ponytail: 앞 토큰 하나만 사용; 긴 문맥 학습은 이후 별도 실습.
model = nn.Embedding(len(vocab), len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(201):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    scores = model(x)
    predictions = scores.argmax(dim=1)
    final_loss = loss_fn(scores, y).item()
    correct = (predictions == y).sum().item()
    assert torch.isfinite(scores).all().item(), "예측 점수에 NaN 또는 무한대 발생"

for (current, target), prediction in zip(pairs, predictions.tolist()):
    print(f"{current:>10} -> {vocab[prediction]:<10} | 정답: {target}")
print(f"학습 쌍 정확도: {correct}/{len(pairs)}")
print(f"최종 학습 loss: {final_loss:.4f}")

with output.open("xb") as file:
    torch.save({"vocab": vocab, "model_state_dict": model.state_dict()}, file)
print("저장 완료:", output)
PY
```

### 블록 A · 파일 보호와 어휘표

`output.exists()`는 기존 결과가 있으면 학습 전에 멈춥니다. 마지막의 `open("xb")`도 **이미 있는 파일을 덮어쓰지 않는** 쓰기 방식입니다. 이전 분류기 파일은 저장 대상이 아닙니다.

`vocab`은 모든 입력·정답 토큰의 집합을 정렬한 목록이고, `to_id`는 그 목록의 위치를 ID로 붙입니다. ID의 대소는 중요도나 의미의 크기가 아닙니다. 어휘를 저장하는 이유는 **학습할 때의 5번과 복원할 때의 5번이 같은 토큰이어야 하기 때문**입니다.

### 블록 B · 텐서가 흘러가는 모양

`shape`는 텐서 각 축의 길이입니다. 여기서 배치는 학습 쌍 9개를 한꺼번에 처리한다는 뜻입니다.

```text
현재 토큰 ID 9개 x : [9]       정답 ID 9개 y : [9]
          │
          ▼ nn.Embedding(9, 9)
후보별 점수 scores : [9, 9]
                    └─ 뒤의 9 = 어휘 후보 수
          │ CrossEntropyLoss(scores, y)
          ▼
스칼라 loss → backward() → Adam.step()
```

`dtype=torch.long`은 ID를 정수로 전달합니다. **ID 5를 실수 의미값 5.0으로 해석하는 것이 아니라, 표의 5번 행을 고르는 인덱스로 사용**합니다.

이번 `nn.Embedding(9, 9)`은 학습 가능한 9행·9열 표입니다. 입력 토큰 ID가 행을 선택하고, 그 행의 9개 값이 곧 다음 토큰 후보의 점수입니다. 별도의 `Linear`나 은닉층을 붙이지 않았습니다. 다음 장의 Embedding은 중간 벡터를 만들지만, **이 장에서는 꺼낸 행 자체가 출력 점수**라는 차이가 있습니다.

`model(x)`는 각 입력 ID에 해당하는 행을 가져옵니다. `scores.argmax(dim=1)`은 각 행의 후보 축에서 가장 큰 점수의 **인덱스**를 고릅니다. 그 인덱스를 `vocab[...]`으로 다시 단어로 바꿉니다.

### 블록 C · 학습 루프의 네 동작

| 순서 | 코드 | 왜 필요한가요? |
|---|---|---|
| 1 | `optimizer.zero_grad()` | 지난번 기울기를 지웁니다. |
| 2 | `loss_fn(model(x), y)` | 현재 점수와 정답 ID를 비교합니다. |
| 3 | `loss.backward()` | loss를 줄이는 방향을 알기 위해 가중치에 대한 기울기를 계산합니다. |
| 4 | `optimizer.step()` | Adam이 그 기울기를 사용해 실제 가중치를 바꿉니다. |

`CrossEntropyLoss`에 넣는 것은 확률이 아니라 **원시 점수(logits)**와 정답 ID입니다. 따라서 앞에 `softmax`를 추가하지 않았습니다. `loss.item()`은 숫자를 출력하기 위해 스칼라 텐서에서 Python 값을 꺼냅니다.

`range(201)`은 0부터 200까지 갱신합니다. 중간 로그는 해당 회차의 `optimizer.step()` **전**, `final_loss`는 반복문을 끝낸 **후** 계산합니다. `torch.manual_seed(42)`는 난수 초기화를 고정하는 설정이지, 모든 환경에서 모든 출력 숫자가 같다는 보장은 아닙니다.

### 블록 D · 평가 모드와 저장

- `model.train()` / `model.eval()`은 모듈의 학습·평가 모드를 지정합니다. 이 단일 Embedding에는 dropout 같은 모드별 동작 차이는 없습니다.
- `torch.no_grad()`는 평가 때 기울기 기록을 하지 않도록 합니다. `eval()`과 같은 기능이 아닙니다.
- `torch.isfinite`는 NaN·무한대 점수를 검사합니다.
- `state_dict()`는 학습된 가중치를 담습니다. 이번 저장은 **어휘와 가중치**를 묶은 체크포인트이며, 이후 코드가 같은 구조의 모델을 다시 만듭니다.

### 실행 예시

```text
토큰 ID: {'<eos>': 0, 'connection': 1, 'disk': 2, 'error': 3, 'full': 4, 'gpu': 5, 'memory': 6, 'network': 7, 'storage': 8}
epoch   0 | loss 2.3318
epoch  50 | loss 0.0283
epoch 100 | loss 0.0126
epoch 150 | loss 0.0080
epoch 200 | loss 0.0056
       gpu -> memory     | 정답: memory
    memory -> error      | 정답: error
     error -> <eos>      | 정답: <eos>
   network -> connection | 정답: connection
connection -> error      | 정답: error
     error -> <eos>      | 정답: <eos>
      disk -> storage    | 정답: storage
   storage -> full       | 정답: full
      full -> <eos>      | 정답: <eos>
학습 쌍 정확도: 9/9
최종 학습 loss: 0.0055
저장 완료: /home/ubuntu/llm-lab/next_token.pt
```

`epoch 200`의 `0.0056`과 최종 `0.0055`는 마지막 가중치 갱신 전·후에 측정한 값입니다. **9/9는 학습에 사용한 같은 9쌍을 다시 평가한 결과**입니다. 이 장의 코드는 별도 validation/test 문장으로 일반화를 검증하지 않습니다.

## 3단계 · 복원하고 예측을 연결해 생성하기

### 최고 점수의 토큰 고르기

한 번에 입력 하나를 넣으면 `scores`의 모양은 `[1, 9]`입니다. `argmax(dim=1)`으로 후보 축에서 최고 점수의 ID를 고르고, `.item()`으로 그 ID를 Python 정수 하나로 꺼냅니다. 마지막으로 `vocab`에서 해당 토큰 문자열을 찾습니다.

```python
next_token = vocab[scores.argmax(dim=1).item()]
```

### 전체 코드 실행하기

**소스:** [03-generate.py](source/05-next-token/03-generate.py)

앞 단계에서 저장한 체크포인트를 읽어 생성합니다. 이 코드는 재학습하거나 체크포인트를 변경하지 않습니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
from pathlib import Path
import torch
from torch import nn

checkpoint = torch.load(
    Path.home() / "llm-lab/next_token.pt",
    map_location="cpu",
    weights_only=True,
)
vocab = checkpoint["vocab"]
assert isinstance(vocab, list) and vocab
assert all(isinstance(token, str) for token in vocab)
assert len(vocab) == len(set(vocab)) and "<eos>" in vocab

to_id = {token: index for index, token in enumerate(vocab)}
model = nn.Embedding(len(vocab), len(vocab))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(start, max_new_tokens=10):
    if start not in to_id or start == "<eos>":
        raise ValueError(f"시작에 사용할 수 없는 토큰: {start}")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens는 양의 정수여야 합니다.")

    words = [start]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            current = torch.tensor([to_id[words[-1]]], dtype=torch.long)
            scores = model(current)
            assert torch.isfinite(scores).all().item(), "잘못된 예측 점수"
            next_token = vocab[scores.argmax(dim=1).item()]

            if next_token == "<eos>":
                return " ".join(words), "<eos>"
            words.append(next_token)

    return " ".join(words), "길이 제한"

for start, expected in [
    ("gpu", "gpu memory error"),
    ("network", "network connection error"),
    ("disk", "disk storage full"),
]:
    text, reason = generate(start)
    print(f"{start} -> {text} | 종료: {reason}")
    assert text == expected and reason == "<eos>", "학습 문장 재현 확인 실패"

assert generate("gpu", max_new_tokens=1)[1] == "길이 제한"
print("검사 통과: 학습 문장 재현, 종료 토큰, 생성 길이 제한")
print("재학습·모델 파일 변경 없음")
PY
```

### 생성 코드의 흐름

| 블록 | 설명 |
|---|---|
| `torch.load(..., map_location="cpu", weights_only=True)` | 기존 체크포인트를 CPU로 읽습니다. 학습을 시작하지 않습니다. 출처를 신뢰하는 실습 파일만 사용합니다. |
| 어휘 `assert` | 문자열 목록·중복 없음·종료 토큰을 검사합니다. |
| 새 Embedding + `load_state_dict` | 학습 때와 같은 크기의 표를 만든 뒤 저장된 값을 채웁니다. |
| `generate` 입력 검사 | 모르는 단어·종료 토큰 시작·잘못된 길이를 거절합니다. |
| `words[-1]` | 만든 문장의 마지막 토큰 하나만 읽습니다. |
| `argmax(...).item()` | `[1, 9]` 점수에서 후보 ID 하나를 선택합니다. |
| 종료 토큰 검사 | `<eos>`면 결과에 붙이지 않고 종료 이유로 반환합니다. |
| `words.append` | 선택한 토큰을 붙여 다음 반복의 입력으로 씁니다. |
| `max_new_tokens` | `<eos>`가 안 나와도 생성이 끝나도록 상한을 둡니다. |
| `expected`와 `assert` | 생성이 끝난 뒤 정답 문장과 비교할 뿐, 정답을 모델에 주입하지 않습니다. |

```text
gpu 입력 [1] → 점수 [1, 9] → memory
memory 입력  → 점수        → error
error 입력   → 점수        → <eos> → 종료
```

매번 최고 점수만 고르므로 **탐욕적 선택(greedy)**입니다. 샘플링·temperature는 사용하지 않습니다. 길이 제한 검사는 `gpu`에서 한 토큰만 생성하게 하여 아직 종료되지 않은 상태를 확인합니다.

### 실행 예시

```text
gpu -> gpu memory error | 종료: <eos>
network -> network connection error | 종료: <eos>
disk -> disk storage full | 종료: <eos>
검사 통과: 학습 문장 재현, 종료 토큰, 생성 길이 제한
재학습·모델 파일 변경 없음
```

## 이 결과가 말해 주는 것과 말해 주지 않는 것

**확인할 것:** 다음 토큰 예측 학습 → 가중치 저장 → 같은 구조로 복원 → 예측 토큰을 다음 입력으로 재사용 → 종료의 흐름입니다.

**이 실습으로 검증하지 않는 것:** 새로운 문장 이해, 일반적인 대화 능력, 모르는 단어 처리, 장거리 문맥 이해입니다. `gpu memory error`가 나왔다고 해서 실제 GPU 장애를 진단한 것도 아닙니다. 교육용 문자열을 재현하는 실습입니다.

같은 마지막 토큰에는 같은 점수 행이 대응합니다. 두 다른 앞 문맥이 모두 `status`로 끝나면서 다른 정답을 요구하면, 이 구조는 앞의 차이를 볼 수 없습니다. 그 문제를 다음 장에서 데이터로 확인합니다.

## 복습 체크리스트

- [ ] `<eos>`를 정답에 넣는 이유와 생성 때 멈추는 위치를 설명할 수 있다.
- [ ] ID·Embedding 점수·예측 ID·토큰 문자열을 구분할 수 있다.
- [ ] `backward()`와 `step()`이 각각 무엇을 바꾸는지 말할 수 있다.
- [ ] `eval()`과 `no_grad()`를 같은 기능이라고 말하지 않는다.
- [ ] 9/9를 새 데이터 정확도로 표현하지 않는다.

<details>
<summary>짧은 확인 질문과 답</summary>

1. **9개 쌍인데 왜 어휘 수도 9인가요?** 이 데이터에서 우연히 둘 다 9입니다. 하나는 사례 수, 다른 하나는 서로 다른 토큰 수입니다.
2. **정답 `expected`가 생성 결과를 만드는가요?** 아닙니다. `generate()`가 먼저 끝나고 그 결과를 검사합니다.
3. **왜 저장할 때 어휘도 넣나요?** 가중치 행·열과 문자열의 대응을 그대로 복원해야 하기 때문입니다.
4. **왜 생성 길이 제한이 있나요?** 모델이 종료 토큰을 내지 않는 경우에도 무한 반복하지 않게 하기 위해서입니다.

</details>

---

[← 이전: 평가 실습](04-evaluation.md) · [학습 목차](README.md) · [다음: 문맥 확장 →](06-context-window.md)
