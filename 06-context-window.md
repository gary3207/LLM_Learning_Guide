# 06 · 문맥 확장 — 같은 마지막 단어, 다른 다음 토큰

[← 이전: 다음 토큰 예측](05-next-token.md) · [학습 목차](README.md) · [다음: 학습 점검과 다음 단계 →](07-resume.md)

마지막 토큰 하나만으로 구별할 수 없는 예문을 만들고, 입력을 두 토큰으로 늘려 작은 신경망을 학습합니다. 앞 장과 같은 Python·PyTorch 가상환경에서 진행하며, 작업 폴더는 `~/llm-lab/source`입니다. 확장 연습에서는 두 토큰과 세 토큰의 정보 차이를 비교한 뒤, 세 토큰 모델의 학습·저장까지 진행합니다.

## 이 장의 목표

- 모델 입력에서 필요한 정보가 사라지면 학습 횟수만 늘려도 해결되지 않음을 설명한다.
- 한 토큰·두 토큰 입력의 **정답 충돌**을 빈도로 확인한다.
- `Embedding → Flatten → Linear`의 텐서 흐름을 이해한다.
- 두 토큰 모델의 저장·복원·문맥 이동을 따라간다.
- 빈도 비교와 신경망 학습 결과를 구분한다.

| 단계 | 실행할 내용 | 확인할 내용 |
|---|---|---|
| 1 | 한 토큰·두 토큰별 정답 빈도 비교 | 같은 입력에서 정답이 충돌하는지 확인 |
| 2 | 두 토큰 모델 학습·저장 | 학습 쌍 예측과 loss 확인 |
| 3 | 두 토큰 모델 복원·생성 | 문맥 이동과 종료·길이 제한 확인 |
| 확장 연습 | 두 토큰·세 토큰별 정답 빈도 비교 | 입력 범위를 늘렸을 때 충돌이 사라지는지 확인 |
| 확장 2/3 | 세 토큰 모델 학습·별도 저장 | 학습 쌍 예측·loss·저장 확인, 복원·생성은 후속 단계 |

아래 출력은 **실행 예시**입니다. 학습 loss와 생성 결과는 환경·난수 초기화·저장된 가중치에 따라 달라질 수 있습니다. seed를 고정해도 모든 환경에서 숫자가 완전히 같다고 보장하지 않습니다. 빈도 비교는 신경망 결과가 아니라 주어진 문자열의 개수를 계산한 것입니다.

## 1단계 · 마지막 토큰이 같으면 생기는 충돌

다음은 실제 자원 상태가 아니라 **교육용으로 정한 두 문장**입니다.

```text
gpu status failed
network status healthy
```

`status`만 입력하면 두 사례가 완전히 같아집니다. 같은 입력에 항상 같은 정답 하나를 내는 방식에서는 `failed`와 `healthy`를 동시에 맞힐 수 없습니다. 앞의 토큰을 함께 넣으면 `gpu status`와 `network status`가 서로 다른 입력이 됩니다.

```text
입력 한 토큰    status            → failed / healthy 충돌
입력 두 토큰    gpu status        → failed
                network status   → healthy
```

### 전체 코드 실행하기

**소스:** [01-observe.py](source/06-context-window/01-observe.py)

이 코드는 PyTorch를 쓰지 않고 문자열·빈도만 비교합니다. 모델을 만들거나 학습하지 않으며, 파일도 저장하지 않습니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
from collections import Counter

sentences = ["gpu status failed", "network status healthy"]
one_token = {}
two_tokens = {}

for sentence in sentences:
    first, current, target = sentence.split()
    one_token.setdefault(current, Counter())[target] += 1
    two_tokens.setdefault((first, current), Counter())[target] += 1

print("[마지막 한 토큰만 사용]")
for context, counts in one_token.items():
    print(f"{context} -> {dict(counts)}")

print("\n[앞의 두 토큰 사용]")
for context, counts in two_tokens.items():
    print(f"{' '.join(context)} -> {dict(counts)}")

one_correct = sum(max(counts.values()) for counts in one_token.values())
two_correct = sum(max(counts.values()) for counts in two_tokens.values())

assert one_token["status"] == Counter({"failed": 1, "healthy": 1})
assert one_correct == 1 and two_correct == 2

print(f"\n한 토큰 입력의 최대 정답 수: {one_correct}/2")
print(f"두 토큰 입력의 최대 정답 수: {two_correct}/2")
print("검사 통과: 한 토큰 입력 충돌, 두 토큰 입력 구분")
PY
```

### 코드 블록별 설명

- **`Counter`와 두 사전:** 문맥마다 정답이 몇 번 나왔는지 모읍니다. 한 토큰 사전의 키는 `status`, 두 토큰 사전의 키는 `('gpu', 'status')` 같은 튜플입니다.
- **`first, current, target`:** 세 단어를 앞 토큰·현재 토큰·예측할 정답으로 나눕니다. 이번 비교는 각 예문의 **최종 단어 예측만** 다룹니다. 뒤의 학습 데이터 전체와 혼동하지 않습니다.
- **`setdefault(..., Counter())`:** 그 입력이 처음 등장하면 빈 카운터를 만들고, 해당 정답의 횟수를 하나 늘립니다.
- **`max(counts.values())`:** 하나의 입력에서 가장 많이 등장한 정답 하나를 고를 때 맞힐 수 있는 사례 수입니다. 같은 입력 그룹들의 최댓값을 더합니다.
- **`assert`:** `status`의 정답이 한 번씩 갈리는지, 입력을 늘리면 두 사례가 구분되는지를 확인합니다.

### 실행 예시

```text
[마지막 한 토큰만 사용]
status -> {'failed': 1, 'healthy': 1}

[앞의 두 토큰 사용]
gpu status -> {'failed': 1}
network status -> {'healthy': 1}

한 토큰 입력의 최대 정답 수: 1/2
두 토큰 입력의 최대 정답 수: 2/2
검사 통과: 한 토큰 입력 충돌, 두 토큰 입력 구분
```

여기서 `1/2`, `2/2`는 **빈도에서 계산한 고정 최빈값 예측의 최대 정답 수**입니다. 신경망을 훈련해 측정한 정확도나 새로운 데이터 성능이 아닙니다. 특히 `2/2`는 구별에 필요한 정보가 입력에 들어왔다는 뜻이지, 아직 어떤 모델이 학습에 성공했다는 뜻은 아닙니다.

## 2단계 · 두 토큰을 함께 사용하는 모델 학습

### 데이터에 시작 표시를 추가한 이유

두 칸의 입력이 필요하므로 처음에는 `<bos>`를 사용합니다. `<bos>`는 시작 표시, `<eos>`는 종료 표시입니다. 다음 데이터 준비 코드는 **6개 학습 쌍**을 만듭니다.

```text
<bos> gpu       → status
gpu status      → failed
status failed   → <eos>

<bos> network   → status
network status  → healthy
status healthy  → <eos>
```

이 코드는 `<bos>`에서 `gpu`와 `network` 중 하나를 고르는 실습이 아닙니다. 시작 단어를 지정하고, **그 뒤를 이어 생성**합니다. 쌍 생성이 `range(2, len(seq))`부터 시작하는 이유도 입력 두 칸을 먼저 채우기 위해서입니다.

### 전체 코드 실행하기

**소스:** [02-train.py](source/06-context-window/02-train.py)

> [!WARNING]
> 실행하면 CPU에서 학습하고 `~/llm-lab/next_token_context2.pt`를 저장합니다. 앞 장의 `next_token.pt`는 변경하지 않습니다. 저장 경로에 파일이 있으면 덮어쓰지 않고 중단합니다. 기존 파일은 먼저 다음 단계에서 복원해 확인합니다. 새로 학습하려면 기존 파일을 별도로 백업하고 새 저장 경로를 정한 뒤, 생성 코드의 로드 경로도 맞춥니다. 보호 검사를 제거하지 않습니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
from pathlib import Path
import torch
from torch import nn

output = Path.home() / "llm-lab/next_token_context2.pt"
if output.exists():
    raise FileExistsError(f"기존 파일을 보호합니다: {output}")

sentences = ["gpu status failed", "network status healthy"]
sequences = [["<bos>"] + s.split() + ["<eos>"] for s in sentences]
vocab = sorted({token for seq in sequences for token in seq})
to_id = {token: i for i, token in enumerate(vocab)}

pairs = []
for seq in sequences:
    for i in range(2, len(seq)):
        pairs.append((seq[i-2:i], seq[i]))

assert len(pairs) == 6
assert (["gpu", "status"], "failed") in pairs
assert (["network", "status"], "healthy") in pairs

x = torch.tensor([[to_id[t] for t in ctx] for ctx, _ in pairs])
y = torch.tensor([to_id[target] for _, target in pairs])
assert x.shape == (6, 2) and y.shape == (6,)

torch.manual_seed(42)
torch.set_num_threads(1)

model = nn.Sequential(
    nn.Embedding(len(vocab), 8),
    nn.Flatten(start_dim=1),
    nn.Linear(16, len(vocab)),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

with torch.no_grad():
    initial_loss = loss_fn(model(x), y).item()

model.train()
for _ in range(300):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    assert torch.isfinite(loss).item(), "학습 loss가 유한하지 않습니다"
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    scores = model(x)
    final_loss = loss_fn(scores, y).item()
    predicted = scores.argmax(dim=1)

print(f"학습 전 loss: {initial_loss:.4f}")
print(f"학습 후 loss: {final_loss:.4f}")

for (context, target), guess in zip(pairs, predicted.tolist()):
    print(f"{' '.join(context)} -> {vocab[guess]} | 정답: {target}")

correct = (predicted == y).sum().item()
print(f"학습 쌍 정확도: {correct}/{len(pairs)}")
assert correct == len(pairs), "학습 쌍을 모두 맞히지 못했습니다"

with output.open("xb") as file:
    torch.save({
        "vocab": vocab,
        "context_size": 2,
        "embedding_dim": 8,
        "model_state_dict": model.state_dict(),
    }, file)

print(f"저장 완료: {output}")
PY
```

### 블록 A · 어휘와 슬라이딩 입력

`sequences`는 문장 앞뒤에 경계 토큰을 붙인 목록입니다. `seq[i-2:i]`는 정답 위치 `i`의 직전 두 토큰을 가져옵니다. 문장마다 따로 반복하므로 서로 다른 문장을 이어 붙인 학습 쌍은 없습니다.

`vocab`에는 `<bos>`, `<eos>`, `failed`, `gpu`, `healthy`, `network`, `status`의 7개 토큰이 들어갑니다. `x`의 각 행은 두 개의 ID이고, `y`의 각 원소는 정답 ID 하나입니다. 이번에는 정수 목록으로부터 정수 텐서를 만들며, `assert`가 데이터 형태를 확인합니다.

### 블록 B · 신경망 모듈의 역할

```text
6개 학습 쌍, 한 쌍에 ID 2개
x                       [6, 2]
 │ nn.Embedding(7, 8)
 ▼
토큰마다 학습된 벡터 8개  [6, 2, 8]
 │ nn.Flatten(start_dim=1)
 ▼
두 벡터를 순서대로 연결   [6, 16]
 │ nn.Linear(16, 7)
 ▼
다음 토큰 후보별 점수     [6, 7]
 │ CrossEntropyLoss(scores, y), y [6]
 ▼
loss 스칼라 → backward → Adam.step
```

| 모듈 | 이 코드에서 하는 일 | 하지 않는 일 |
|---|---|---|
| `nn.Sequential` | 나열된 모듈에 앞 출력 → 뒤 입력을 순서대로 전달 | 모델 구조를 자동으로 찾아 주지 않음 |
| `nn.Embedding(7, 8)` | ID 하나에서 학습 가능한 8개 숫자 벡터를 조회 | 완성된 한국어·영어 의미를 사전 탑재하지 않음 |
| `nn.Flatten(start_dim=1)` | 배치 축은 그대로 두고 각 쌍의 두 벡터를 16개 값으로 펼침 | 평균을 내거나 새 가중치를 학습하지 않음 |
| `nn.Linear(16, 7)` | 16개 값에 학습한 가중치·편향을 적용해 후보 7개의 점수 생성 | 그 자체로 확률을 출력하지 않음 |

두 토큰을 펼칠 때 **첫 토큰 벡터의 자리와 둘째 토큰 벡터의 자리가 다릅니다.** 단순히 토큰을 집합으로 합치는 방식이 아닙니다. `Linear`의 가중치는 그 위치별 값을 따로 사용할 수 있습니다.

이 모델은 `Sequential`이라고 해서 RNN이 아니며, Attention·Transformer·별도의 위치 임베딩도 사용하지 않습니다. 중간에 ReLU 같은 비선형 활성화도 없습니다. 실습 결과를 더 복잡한 구조의 성능으로 해석하지 않습니다.

### 블록 C · 학습·검사·저장

- `manual_seed(42)`로 초기화 난수를 설정하고 `set_num_threads(1)`로 CPU 연산 스레드 수를 설정합니다. GPU를 사용하도록 이동시키는 코드는 없습니다.
- 처음의 `no_grad()` 구간은 **학습 전 loss를 측정**합니다. 가중치를 바꾸지 않습니다.
- 반복문은 300번, `zero_grad → loss → backward → step` 순서입니다. `isfinite(loss)`는 수치 오류가 생기면 멈춥니다.
- 반복문 뒤에는 `eval()`과 `no_grad()`에서 학습 쌍의 점수·loss·예측을 확인합니다. **학습·평가에 같은 쌍을 사용합니다.**
- `assert correct == len(pairs)`는 학습 쌍을 모두 맞힌 뒤에만 저장하게 합니다.
- 체크포인트에는 `vocab`, `context_size=2`, `embedding_dim=8`, `model_state_dict`를 저장합니다. 구조 정보는 다음 코드가 복원 대상이 맞는지 확인할 때 씁니다.

### 실행 예시

```text
학습 전 loss: 1.6644
학습 후 loss: 0.0000
<bos> gpu -> status | 정답: status
gpu status -> failed | 정답: failed
status failed -> <eos> | 정답: <eos>
<bos> network -> status | 정답: status
network status -> healthy | 정답: healthy
status healthy -> <eos> | 정답: <eos>
학습 쌍 정확도: 6/6
저장 완료: /home/ubuntu/llm-lab/next_token_context2.pt
```

**`0.0000`은 소수 넷째 자리 출력값이지, 수학적으로 정확히 0이라는 증거가 아닙니다.** 또한 6/6은 같은 학습 쌍의 적합 결과입니다. 별도 검증·테스트 자료에 대한 정확도는 이 결과에 포함되어 있지 않습니다.

## 3단계 · 마지막 두 토큰을 옮기며 생성하기

### 전체 코드 실행하기

**소스:** [03-generate.py](source/06-context-window/03-generate.py)

앞 단계에서 저장한 가중치를 같은 구조의 모델에 불러옵니다. `scores.argmax(dim=1)`은 `[1, 7]` 점수에서 후보 축의 최고 점수 ID를 고릅니다. `.item()`으로 정수 하나를 꺼내 `vocab`에서 다음 토큰 문자열을 찾습니다. 복원·생성만 수행하며 재학습하거나 파일을 덮어쓰지 않습니다. 신뢰하는 실습 체크포인트만 불러옵니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
from pathlib import Path
import torch
from torch import nn

path = Path.home() / "llm-lab/next_token_context2.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=True)

vocab = checkpoint["vocab"]
assert isinstance(vocab, list) and vocab
assert all(isinstance(token, str) for token in vocab)
assert len(vocab) == len(set(vocab))
assert "<bos>" in vocab and "<eos>" in vocab
assert checkpoint["context_size"] == 2
assert checkpoint["embedding_dim"] == 8

to_id = {token: i for i, token in enumerate(vocab)}
model = nn.Sequential(
    nn.Embedding(len(vocab), 8),
    nn.Flatten(start_dim=1),
    nn.Linear(16, len(vocab)),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(start, max_new_tokens=10):
    if start not in to_id or start in ("<bos>", "<eos>"):
        raise ValueError(f"시작에 사용할 수 없는 토큰: {start}")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens는 양의 정수여야 합니다")

    history = ["<bos>", start]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = history[-2:]
            x = torch.tensor([[to_id[t] for t in context]])
            scores = model(x)
            assert torch.isfinite(scores).all().item()

            next_token = vocab[scores.argmax(dim=1).item()]

            if next_token == "<eos>":
                return " ".join(history[1:]), "<eos>"
            if next_token == "<bos>":
                raise RuntimeError("생성 도중 시작 토큰이 나왔습니다")

            history.append(next_token)

    return " ".join(history[1:]), "길이 제한"

for start, expected in [
    ("gpu", "gpu status failed"),
    ("network", "network status healthy"),
]:
    text, reason = generate(start)
    print(f"{start} -> {text} | 종료: {reason}")
    assert text == expected and reason == "<eos>"

assert generate("gpu", max_new_tokens=1) == ("gpu status", "길이 제한")

print("검사 통과: 문맥별 문장 재현, 종료 토큰, 생성 길이 제한")
print("재학습·모델 파일 변경 없음")
PY
```

### 무엇이 바뀌었나요?

| 블록 | 한 토큰 모델과의 차이 |
|---|---|
| 체크포인트 검사 | 문맥 2칸·임베딩 8차원을 확인한 뒤 같은 Sequential을 만듭니다. |
| `history = ["<bos>", start]` | 시작부터 두 칸을 채웁니다. |
| `context = history[-2:]` | 전체 생성 기록에서 마지막 두 토큰만 고릅니다. |
| `torch.tensor([[...]])` | 한 문맥도 배치 한 행으로 만들어 `[1, 2]`를 유지합니다. |
| `model(x)` | `[1, 2] → [1, 2, 8] → [1, 16] → [1, 7]`로 점수를 만듭니다. |
| `history.append(next_token)` | 선택한 토큰을 붙이고 다음 회차의 입력 범위를 옮깁니다. |
| `history[1:]` | 표시용 결과에서 맨 앞 `<bos>`만 제외합니다. |
| `<bos>`가 중간에 나오면 오류 | 시작 토큰이 일반 단어처럼 생성되는 비정상 결과를 알립니다. |

생성의 기록은 누적되지만 **모델이 보는 입력은 여전히 두 토큰뿐**입니다.

```text
전체 기록                    모델 입력           다음 선택
<bos> gpu                    <bos> gpu           status
<bos> gpu status             gpu status          failed
<bos> gpu status failed      status failed       <eos> → 종료
```

`expected`는 사후 검사값입니다. 생성 중에는 `model(x)`의 점수와 `argmax`로만 다음 토큰을 고릅니다. 길이를 1로 제한한 검사는 `("gpu status", "길이 제한")`을 확인합니다. `<eos>`를 결과 문장에 붙이거나 무한 생성을 허용하지 않습니다.

### 실행 예시

```text
gpu -> gpu status failed | 종료: <eos>
network -> network status healthy | 종료: <eos>
검사 통과: 문맥별 문장 재현, 종료 토큰, 생성 길이 제한
재학습·모델 파일 변경 없음
```

이 예시에서는 앞 단어가 `gpu`인지 `network`인지 입력에 포함했을 때 학습한 다른 후속 단어를 선택합니다. 실제 장비 상태를 읽거나 알려지지 않은 문장을 이해하는 기능을 검증하는 것은 아닙니다.

## 이 장의 핵심과 한계

1. 입력에 구분 정보가 없으면 같은 입력에 다른 정답이 충돌합니다. 단순히 더 오래 학습해도 빠진 앞 단어를 볼 수는 없습니다.
2. 이번 두 토큰 모델은 그 예문에 필요한 정보를 넣고 작은 신경망이 학습 문장을 재현하도록 한 실험입니다.
3. `history`에 문장이 남아 있는 것과 모델이 그 전체를 읽는 것은 다릅니다. 실제 입력은 `history[-2:]`입니다.
4. 입력 범위를 늘릴 때는 데이터 준비·모델 입력 크기·복원 구조가 함께 맞아야 합니다. 빈도 비교만으로 새 모델의 학습 성공을 결론 내릴 수 없습니다.
5. 여기서 사용한 “문맥 창”은 작은 고정 길이 입력의 비유입니다. 실제 대형 모델의 긴 문맥 처리·Attention을 이 실습에서 구현한 것은 아닙니다.

## 복습 체크리스트

- [ ] 1/2 대 2/2 빈도 비교와 6/6 신경망 학습 결과를 구별한다.
- [ ] `Embedding`의 벡터 차원 8과 문맥 길이 2를 구별한다.
- [ ] `Flatten(start_dim=1)`이 배치 축을 보존하는 이유를 설명한다.
- [ ] 후보 점수의 마지막 차원이 7인 이유를 말할 수 있다.
- [ ] `0.0000`을 정확한 0이나 일반화 성공으로 단정하지 않는다.
- [ ] 세 토큰 빈도 비교와 세 토큰 신경망 학습 결과를 구별한다.

<details>
<summary>확인 질문과 답</summary>

- **두 토큰을 평균내도 같은 코드인가요?** 아닙니다. 이 코드는 순서대로 펼쳐 위치별 자리를 유지합니다.
- **생성 기록이 길어지면 자동으로 더 긴 문맥을 보나요?** 아닙니다. 입력은 여전히 마지막 두 토큰입니다.
- **`current status`로 끝나는 두 예문을 이 두 토큰 모델에 오래 학습시키면 둘 다 구별할 수 있나요?** 그 입력만으로는 두 경우를 구분할 정보가 없습니다.
- **세 토큰 입력의 2/2는 모델 학습 성공인가요?** 아닙니다. 빈도표에서 충돌이 사라졌다는 뜻입니다. 뒤의 확장 2/3단계에서 실제 모델을 학습해 별도로 검사합니다.

</details>

## 확장 연습 · 두 토큰도 모자라는 예문

### 전체 흐름과 이번 실행 범위

세 토큰 문맥 확장은 다음 세 단계로 나누어 진행합니다. **이 절은 1/3단계만 실행합니다.**

| 단계 | 학습할 내용 | 이 절에서 실행하는가? |
|---|---|---|
| 1/3 · 입력 비교 | 두 토큰 입력의 정답 충돌과 세 토큰 입력의 차이 관찰 | 실행: 아래 빈도 비교 코드 |
| 2/3 · 모델 학습 | 세 토큰 학습 쌍 구성, 모델 학습·별도 저장 | 실행하지 않음: 후속 과제 |
| 3/3 · 복원·생성 | 저장 모델 복원, 문맥 이동·종료·길이 제한 검사 | 실행하지 않음: 후속 과제 |

**실행 위치:** OCI A10 GPU VM의 기존 터미널에서 진행합니다. 아래 코드는 Python 표준 라이브러리만 사용하므로 PyTorch·GPU·기존 체크포인트가 필요하지 않습니다. 기존 `(.venv)`가 켜져 있다면 그대로 사용해도 됩니다. 파일 생성·모델 학습·저장은 없습니다.

중간에 `current`를 넣어 구별에 필요한 단어가 입력 범위 밖으로 밀려나는 상황을 만듭니다.

```text
gpu current status     → failed
network current status → healthy

마지막 두 토큰: 둘 다 current status → 충돌
마지막 세 토큰: gpu/network가 포함됨 → 구분 가능
```

### 전체 코드 실행하기

**소스:** [04-observe-context3.py](source/06-context-window/04-observe-context3.py)

이 코드는 일반 사전으로 정답 빈도를 셉니다. 1단계의 `Counter`와 목적은 같지만 `dict.get(..., 0)`으로 개수를 직접 누적합니다. 신경망 학습·파일 저장은 없습니다. 실행 전에 마지막 두 토큰만 남겼을 때 어떤 정보가 사라지는지 먼저 예상합니다.

```bash
cd ~/llm-lab/source

python - <<'PY'
sentences = [
    "gpu current status failed",
    "network current status healthy",
]

for size in (2, 3):
    counts = {}
    for sentence in sentences:
        tokens = sentence.split()
        context = tuple(tokens[-(size + 1):-1])
        target = tokens[-1]
        counts.setdefault(context, {})
        counts[context][target] = counts[context].get(target, 0) + 1

    print(f"[마지막 {size}토큰으로 최종 단어 예측]")
    for context, targets in counts.items():
        print(" ".join(context), "->", targets)

    correct = sum(max(targets.values()) for targets in counts.values())
    print(f"고정된 최빈값 예측의 최대 정답 수: {correct}/{len(sentences)}")
    assert correct == (1 if size == 2 else 2)
    print()

print("검사 통과: 두 토큰 입력 충돌, 세 토큰 입력 구분")
PY
```

### 코드 읽기

`tokens[-(size + 1):-1]`는 마지막 정답 토큰은 빼고, 그 직전 `size`개를 선택합니다. `tuple`로 만들어 사전 키로 쓰고, `target`은 마지막 단어 하나입니다. 입력별 정답 횟수를 센 뒤 가장 흔한 정답의 개수를 합칩니다.

### 실행 예시

```text
[마지막 2토큰으로 최종 단어 예측]
current status -> {'failed': 1, 'healthy': 1}
고정된 최빈값 예측의 최대 정답 수: 1/2

[마지막 3토큰으로 최종 단어 예측]
gpu current status -> {'failed': 1}
network current status -> {'healthy': 1}
고정된 최빈값 예측의 최대 정답 수: 2/2

검사 통과: 두 토큰 입력 충돌, 세 토큰 입력 구분
```

이 결과는 **두 예문의 최종 단어에 대한 고정된 최빈값 예측의 최대 정답 수**입니다. 두 토큰 입력의 `1/2`, 세 토큰 입력의 `2/2`는 신경망 정확도나 새 데이터 성능이 아닙니다. 입력을 구분할 수 있게 만드는 것과 모델이 학습·일반화하는 것은 별개입니다.

### 1/3단계 확인 질문

- 두 예문의 입력을 마지막 두 토큰으로 제한하면 왜 모두 `current status`가 되나요?
- 입력을 세 토큰으로 늘리면 어떤 구분 정보가 다시 들어오나요?
- `2/2`가 나와도 아직 모델 학습에 성공했다고 말할 수 없는 이유는 무엇인가요?

출력 마지막의 `검사 통과`는 코드의 빈도 계산과 예상값 비교가 통과했다는 뜻입니다. **학습을 더 반복해도 입력에서 제외한 앞 단어를 볼 수는 없습니다.** 위 차이를 확인한 뒤 2/3단계의 모델 학습으로 넘어갑니다.

다음 절에서는 세 토큰 신경망을 학습합니다. 입력 길이를 바꾸면 데이터 슬라이스, `Linear`의 입력 크기, 생성 중 문맥 선택, 저장·복원 구조를 함께 맞춰야 합니다.

## 확장 2/3단계 · 세 토큰 모델 학습·저장

### 목표와 실행 위치

빈도 비교에서 확인한 구분 정보를 실제 신경망 입력에 넣습니다. **OCI VM의 `~/llm-lab/source`, 기존 PyTorch 가상환경**에서 실행합니다. 이 작은 모델은 CPU를 사용합니다. GPU VM이라고 모든 코드가 자동으로 GPU에서 실행되는 것은 아닙니다.

가상환경을 새로 설치하지 않고 기존 환경을 활성화합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version
```

`python: command not found`가 나오면 기존 가상환경 활성화 여부를 먼저 확인합니다. `.venv/bin/activate`도 없다면 설치를 반복하기 전에 실제 기존 환경 경로를 확인합니다.

### 세 칸의 입력 만들기

시작 단어 `gpu` 또는 `network` 뒤를 예측하는 방식은 유지합니다. 세 칸을 채우기 위해 시작 표시를 두 개 사용합니다.

```text
<bos> <bos> gpu        → current
<bos> gpu current      → status
gpu current status     → failed
current status failed  → <eos>

<bos> <bos> network    → current
<bos> network current  → status
network current status → healthy
current status healthy → <eos>
```

`range(3, len(seq))`는 입력 세 칸 다음부터 정답을 고르고, `seq[i-3:i]`는 그 직전 세 토큰을 가져옵니다. 문장을 각각 처리하므로 문장 사이에 가짜 학습 쌍을 만들지 않습니다.

### 전체 실행 코드

**소스:** [05-train-context3.py](source/06-context-window/05-train-context3.py). 파일 내용을 VM의 `~/llm-lab/source/train_context3.py`로 저장했다면 `python train_context3.py`로 실행할 수 있습니다. 또는 아래 전체를 그대로 붙여 넣습니다.

> [!WARNING]
> 실행하면 `~/llm-lab/next_token_context3.pt`를 새로 저장합니다. 기존 `next_token.pt`·`next_token_context2.pt`는 변경하지 않습니다. 저장 대상이 이미 있으면 보호 오류로 중단하며, 파일을 삭제하거나 보호 검사를 제거하지 않습니다.

```bash
python - <<'PY'
from pathlib import Path
import torch
from torch import nn

output = Path.home() / "llm-lab/next_token_context3.pt"
if output.exists():
    raise FileExistsError(f"기존 파일을 보호합니다: {output}")

sentences = ["gpu current status failed", "network current status healthy"]
sequences = [["<bos>", "<bos>"] + s.split() + ["<eos>"] for s in sentences]
vocab = sorted({token for seq in sequences for token in seq})
to_id = {token: i for i, token in enumerate(vocab)}

pairs = []
for seq in sequences:
    for i in range(3, len(seq)):
        pairs.append((seq[i-3:i], seq[i]))

assert len(pairs) == 8
assert (["gpu", "current", "status"], "failed") in pairs
assert (["network", "current", "status"], "healthy") in pairs

x = torch.tensor([[to_id[t] for t in ctx] for ctx, _ in pairs])
y = torch.tensor([to_id[target] for _, target in pairs])
assert x.shape == (8, 3) and y.shape == (8,)

torch.manual_seed(42)
torch.set_num_threads(1)

model = nn.Sequential(
    nn.Embedding(len(vocab), 8),
    nn.Flatten(start_dim=1),
    nn.Linear(24, len(vocab)),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

with torch.no_grad():
    initial_loss = loss_fn(model(x), y).item()

model.train()
for _ in range(300):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    assert torch.isfinite(loss).item(), "학습 loss가 유한하지 않습니다"
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    scores = model(x)
    assert torch.isfinite(scores).all().item(), "예측 점수가 유한하지 않습니다"
    final_loss = loss_fn(scores, y).item()
    predicted = scores.argmax(dim=1)

print(f"학습 전 loss: {initial_loss:.4f}")
print(f"학습 후 loss: {final_loss:.4f}")
for (context, target), guess in zip(pairs, predicted.tolist()):
    print(f"{' '.join(context)} -> {vocab[guess]} | 정답: {target}")

correct = (predicted == y).sum().item()
print(f"학습 쌍 정확도: {correct}/{len(pairs)}")
assert correct == len(pairs), "학습 쌍을 모두 맞히지 못했습니다"

with output.open("xb") as file:
    torch.save({
        "vocab": vocab,
        "context_size": 3,
        "embedding_dim": 8,
        "model_state_dict": model.state_dict(),
    }, file)

print(f"저장 완료: {output}")
PY
```

### 두 토큰 모델과 달라진 부분

| 항목 | 두 토큰 모델 | 이번 세 토큰 모델 |
|---|---|---|
| 예문 | `gpu status failed` 등 | 중간에 `current` 추가 |
| 시작 표시 | `<bos>` 1개 | `<bos>` 2개 |
| 입력 슬라이스 | `seq[i-2:i]` | `seq[i-3:i]` |
| 학습 입력 | `[6, 2]` | `[8, 3]` |
| 어휘 수 | 7 | 8: `current` 추가 |
| 임베딩 후 | `[6, 2, 8]` | `[8, 3, 8]` |
| 펼친 뒤 | `[6, 16]` | `[8, 24]` |
| 최종 점수 | `[6, 7]` | `[8, 8]` |
| 저장 문맥 길이 | `context_size=2` | `context_size=3` |
| 저장 파일 | `next_token_context2.pt` | `next_token_context3.pt` |

`Linear(24, len(vocab))`의 24는 세 토큰의 임베딩 벡터를 순서대로 연결한 입력 크기입니다. 점수 `[8, 8]`의 앞 8은 학습 쌍 수, 뒤 8은 다음 토큰 후보 수로 서로 다른 의미입니다. `y`는 정답 ID 8개인 `[8]` 형태입니다.

학습 흐름은 기존처럼 `zero_grad → loss → backward → step`입니다. 학습 전후 loss를 비교하고 각 입력의 예측·정답을 출력합니다. 8쌍을 모두 맞히는 검사 후에만 저장합니다. `output.open("xb")`는 배타적 생성으로 기존 파일 덮어쓰기를 막습니다.

### 실행 결과에서 확인할 것

아래는 **성공 시 확인할 조건**이며, 실제 실행 로그나 이미 측정한 수치가 아닙니다. loss 숫자는 실행 후 확인합니다.

- `gpu current status`의 예측이 `failed`, `network current status`의 예측이 `healthy`인지 확인합니다.
- `학습 쌍 정확도: 8/8`과 새 모델의 `저장 완료` 출력을 확인합니다.
- `0.0000`이 표시되어도 반올림된 값이며 정확한 0을 뜻하지 않습니다.
- 같은 학습 쌍으로 평가하므로 8/8을 새 문장에 대한 성능으로 해석하지 않습니다.

오류나 검사 실패가 나오면 출력을 확인하고, 성공한 것처럼 넘어가지 않습니다. 모델 복원·문장 생성·종료 검사는 **3/3단계의 후속 실습**으로 구분합니다.

### 복습

- 왜 `<bos>` 두 개가 필요한가요?
- 문맥 길이를 바꿀 때 왜 데이터 슬라이스뿐 아니라 `Linear` 입력 크기도 바꾸나요?
- 저장한 `context_size=3`과 같은 구조로 복원해야 하는 이유는 무엇인가요?

---

[← 이전: 다음 토큰 예측](05-next-token.md) · [학습 목차](README.md) · [다음: 학습 점검과 다음 단계 →](07-resume.md)
