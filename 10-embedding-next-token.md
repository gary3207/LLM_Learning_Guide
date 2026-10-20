# 10 · 임베딩과 다음 토큰 학습 연결

[← 이전: 작은 Transformer 블록](09-transformer.md) · [목차](README.md)

## 1. 목표와 앞 실습과의 차이

09장의 고정 X 대신 **학습되는 토큰 임베딩 + 학습되는 위치 임베딩**으로 입력을 만듭니다. 06장에서 사용한 두 예문을 다시 가져와, 단일-head Pre-LN 블록이 **각 위치에서 다음 토큰**을 예측하도록 학습합니다. 저장·복원·반복 생성은 하지 않습니다.

| 항목 | 09장 첫 블록 | 이번 실습 |
|---|---|---|
| 입력 | 직접 지정한 실수 벡터 | 토큰 ID → 학습되는 임베딩 |
| 크기 | 3위치·2좌표 | 두 문장(batch)·4위치·4좌표 |
| 정답 | 마지막 위치의 교육용 failed 하나 | 각 문장의 실제 다음 토큰 네 개 |
| 어휘 | 출력 후보 세 개 | 입력·정답에 등장하는 일곱 토큰 |
| 갱신 | 한 번 | 같은 두 예문으로 기본 SGD 200회 |

2좌표 LayerNorm 예제의 표현 제약을 줄이고 학습 연결을 관찰하기 위해 좌표를 4개, FFN을 `4→8→4`로 정합니다. 구조 흐름은 재사용하지만 크기가 달라 **앞의 2×2 행렬이나 학습된 값을 가져오지 않습니다**. seed 7로 모든 파라미터를 새로 초기화합니다. 이전 loss와 성능을 직접 비교하지 않습니다. 학습률도 이번 고정 실습에서는 0.1입니다.

```text
문장 → 공백으로 토큰 분리 → ID
                        ↓
      토큰 임베딩 + 위치 임베딩 → causal Pre-LN 블록
                        ↓
      모든 위치의 어휘 logits → 다음 토큰 정답과 loss
                        ↓
      역전파 → 임베딩·블록·출력층 함께 갱신
```

## 2. 입력과 정답을 한 칸씩 어긋나게 만들기

문장마다 `<eos>`를 붙인 다음 앞 네 토큰을 입력, 뒤 네 토큰을 정답으로 만듭니다.

```text
문장: gpu current status failed <eos>
입력: gpu current status failed
정답: current status failed <eos>

문장: network current status healthy <eos>
입력: network current status healthy
정답: current status healthy <eos>
```

예를 들어 입력의 `status` 위치는 앞의 `gpu current status`만 보고 정답 `failed`를 맞힙니다. 같은 batch 안에 뒤 입력 `failed`가 들어 있어도 **causal mask가 미래 위치를 가리므로 볼 수 없습니다**. 정답 배열 y는 loss에만 사용하고 모델 입력에 더하지 않습니다.

두 문장을 따로 행으로 만들어 문장 경계를 넘는 가짜 학습 쌍을 만들지 않습니다. 길이가 같아 padding은 없습니다. `<bos>`는 넣지 않으므로 첫 토큰 gpu/network 자체를 예측하는 학습은 없습니다. 두 문장 × 네 위치 = 총 여덟 정답 위치입니다.

### 임베딩은 무엇을 학습하나?

- 토큰 ID는 행을 찾는 정수 번호일 뿐 크기나 의미의 순서가 아닙니다. 예를 들어 이번 정렬에서 gpu는 ID 3입니다.
- `nn.Embedding(7, 4)`는 **일곱 행·네 좌표의 학습 가능한 표**입니다. gpu ID를 넣으면 그 행의 실수 벡터를 꺼냅니다. 학습은 이 표의 값도 바꿉니다.
- `nn.Embedding(4, 4)`는 위치 0·1·2·3의 벡터 표입니다. 토큰 표와 별개로 학습하며, 같은 위치 표를 두 문장이 공유합니다.
- 토큰 벡터와 위치 벡터는 좌표끼리 더합니다. 차원은 4로 유지합니다. 위치 표는 미래 차단을 대신하지 않습니다.

## 3. 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source`의 PyTorch 가상환경에서 실행합니다. `cd`는 폴더 이동, `source`는 기존 환경 활성화이며 `python -`부터 마지막 `PY`까지 한 번에 실행합니다. CPU·float64이며 설치·GPU 사용·파일 읽기·저장·기존 체크포인트 덮어쓰기는 없습니다. 메모리에서 새 모델을 200회 학습합니다.

기존 환경에서 torch를 못 찾으면 설치부터 반복하지 말고 활성화된 Python 경로를 확인합니다. assert가 필요하므로 `python -O`를 사용하지 않습니다.

**소스:** [02-embedding-next-token.py](source/09-transformer/02-embedding-next-token.py) — 같은 Transformer 실습 묶음의 source/09-transformer에 보관합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import sqrt
import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(1)
sentences = ["gpu current status failed", "network current status healthy"]
sequences = [s.split() + ["<eos>"] for s in sentences]
vocab = sorted({token for seq in sequences for token in seq})
to_id = {token: i for i, token in enumerate(vocab)}
x = torch.tensor([[to_id[t] for t in seq[:-1]] for seq in sequences], dtype=torch.long)
y = torch.tensor([[to_id[t] for t in seq[1:]] for seq in sequences], dtype=torch.long)
assert x.shape == y.shape == (2, 4)
assert vocab == ["<eos>", "current", "failed", "gpu", "healthy", "network", "status"]


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        options = {"dtype": torch.float64, "device": "cpu"}
        self.token = nn.Embedding(len(vocab), 4, **options)
        self.position = nn.Embedding(4, 4, **options)
        self.norm1 = nn.LayerNorm(4, eps=1e-5, **options)
        self.norm2 = nn.LayerNorm(4, eps=1e-5, **options)
        self.q = nn.Linear(4, 4, bias=False, **options)
        self.k = nn.Linear(4, 4, bias=False, **options)
        self.v = nn.Linear(4, 4, bias=False, **options)
        self.ff = nn.Sequential(nn.Linear(4, 8, **options), nn.ReLU(),
                                nn.Linear(8, 4, **options))
        self.head = nn.Linear(4, len(vocab), bias=False, **options)

    def forward(self, ids):
        # ponytail: 최대4토큰·단일head·padding 없음; 가변 길이 batch는 후속 범위입니다.
        assert ids.ndim == 2 and ids.shape[0] > 0 and 1 <= ids.shape[1] <= 4
        assert ids.dtype == torch.long and torch.all((ids >= 0) & (ids < len(vocab)))
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.token(ids) + self.position(positions)
        normalized = self.norm1(embedded)
        Q, K, V = self.q(normalized), self.k(normalized), self.v(normalized)
        scores = (Q @ K.transpose(-2, -1)) / sqrt(Q.shape[-1])
        future = torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool, device=ids.device).triu(1)
        attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        residual = embedded + attention @ V
        hidden = residual + self.ff(self.norm2(residual))
        return self.head(hidden), attention  # 모든 위치의 어휘 점수


model = TinyLanguageModel()
loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
with torch.no_grad():
    initial_logits, _ = model(x)
    initial_loss = loss_fn(initial_logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
initial_token = model.token.weight.detach().clone()
initial_position = model.position.weight.detach().clone()
before = {name: p.detach().clone() for name, p in model.named_parameters()}
model.train()
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    logits, attention = model(x)
    loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    if step == 0:
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.norm().item() > 0.0, name
            assert torch.equal(p.detach(), before[name]), name
        # <eos>는 정답에만 등장하므로 이 입력 임베딩 행의 기울기는 0입니다.
        assert torch.all(model.token.weight.grad[to_id["<eos>"]] == 0)
    optimizer.step()
    if step == 0:
        for name, p in model.named_parameters():
            assert torch.allclose(p.detach(), before[name] - 0.1 * p.grad, rtol=0.0, atol=1e-12), name
            assert not torch.equal(p.detach(), before[name]), name
model.eval()
with torch.no_grad():
    logits, attention = model(x)
    final_loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
    predicted = logits.argmax(dim=-1)
    changed = x.clone()
    changed[:, -1] = to_id["gpu"]
    assert torch.allclose(logits[:, :-1], model(changed)[0][:, :-1], rtol=0.0, atol=1e-12)
assert torch.isfinite(logits).all() and final_loss < initial_loss
assert torch.all(attention.triu(1) == 0.0)
assert torch.allclose(attention.sum(-1), torch.ones((2, 4), dtype=torch.float64))
assert not torch.equal(model.token.weight.detach(), initial_token)
assert not torch.equal(model.position.weight.detach(), initial_position)
correct = (predicted == y).sum().item()
assert correct == y.numel(), "이번 고정 설정의 학습 정답을 모두 맞히지 못했습니다"
print("어휘:", vocab)
print("shape: 입력=(2, 4), 정답=(2, 4), 임베딩=(2, 4, 4), logits=(2, 4, 7)")
print("기울기·변경 검사 통과: 토큰 임베딩·위치 임베딩·블록·어휘 출력층")
print(f"학습 전 loss: {initial_loss:.4f}")
print(f"200회 갱신 후 loss: {final_loss:.4f}")
for row, seq in enumerate(sequences):
    for position in range(x.shape[1]):
        context = " ".join(seq[:position+1])
        guess = vocab[predicted[row, position].item()]
        print(f"{context} -> {guess} | 정답: {seq[position+1]}")
print(f"학습 위치 정답 수: {correct}/{y.numel()}")
print("검사 통과: 미래 차단·임베딩 학습·다음 토큰 학습 연결. 저장·생성은 없습니다")
PY
```

## 4. 코드와 텐서 크기 해설

| 변수·연산 | 크기와 의미 |
|---|---|
| `sequences` | 각 문장 토큰 목록 + eos, 문장마다 분리 |
| `seq[:-1]` / `seq[1:]` | 마지막 토큰 제외 입력 / 첫 토큰 제외 정답 |
| `x`, `y` | 각각 `2×4`, 정수 ID. 2는 문장 수, 4는 위치 수 |
| `self.token(ids)` | `2×4×4`, 문장·위치·좌표 |
| `self.position(positions)` | `4×4`, 위치·좌표. 문장 축에는 같은 값을 공유해 더함 |
| `Q`, `K`, `V` | 각각 `2×4×4` |
| `attention` | `2×4×4`, 문장·질문 위치·참조 위치 |
| `self.head(hidden)` | `2×4×7`, 문장·위치·다음 토큰 후보 |
| `reshape(-1, len(vocab))` | logits를 `8×7`로, 문장·위치를 같은 순서로 펼침 |
| `y.reshape(-1)` | 정답을 길이 8로 펼침 |

`-1`은 나머지 크기를 자동 계산하라는 뜻입니다. **여기서 펼치는 것은 loss를 계산할 문장·위치 축이지, 앞 고정 문맥 모델처럼 입력 임베딩 좌표를 이어 붙이는 것이 아닙니다.** 각 위치는 어휘 일곱 개의 점수와 정답 번호 하나를 갖습니다. `CrossEntropyLoss`에 확률이 아닌 점수를 넣고 여덟 위치의 평균 loss를 계산합니다.

batch가 추가되어 K는 3차원입니다. 따라서 이전 2차원 `K.T` 대신 `K.transpose(-2, -1)`로 **마지막 두 축만** 뒤집습니다. 서로 다른 문장끼리 Attention을 계산하지 않습니다. `softmax(dim=-1)`은 각 문장의 각 질문 위치에서 참조 위치 축을 정규화합니다.

`model.parameters()`에 토큰·위치 임베딩도 포함합니다. 첫 backward에서 모든 파라미터 텐서의 유한한 비영 기울기를 확인하고, 값은 아직 바뀌지 않았는지 검사합니다. 첫 SGD 뒤 각 텐서가 `이전 값 - 0.1 × 기울기`로 바뀌었는지 확인합니다. 이후 매회 기울기를 초기화하고 새로 계산합니다.

비영 기울기는 텐서 전체의 norm 검사이지 **모든 행·원소가 0이 아니어야 한다는 뜻이 아닙니다**. eos는 정답에만 있고 입력에는 없어 그 토큰의 입력 임베딩 행은 기울기 0입니다. 어휘 출력층의 eos 후보 행과는 별개이며, 두 층은 가중치를 공유하지 않습니다.

`model.train()`은 학습 모드, `model.eval()`은 평가 모드입니다. 이 구조에는 dropout 등 모드에 따라 달라지는 층이 없지만 학습과 결과 확인의 경계를 표시합니다. `no_grad()`는 결과 확인에서 미분 그래프를 만들지 않으며 eval과는 별개의 기능입니다.

## 5. 로컬 실제 실행 예시와 해석

Python 3.12·PyTorch 2.14.1+cpu 격리 환경에서 실제 200회 학습하고 모든 assert를 통과한 출력입니다. 사용자 VM 완료를 뜻하지 않습니다. seed를 고정해도 환경에 따라 수치가 달라질 수 있습니다. 이 고정 실습은 최종 8/8을 검사하지만 모든 설정의 학습 성공을 보장하지 않습니다.

```text
어휘: ['<eos>', 'current', 'failed', 'gpu', 'healthy', 'network', 'status']
shape: 입력=(2, 4), 정답=(2, 4), 임베딩=(2, 4, 4), logits=(2, 4, 7)
기울기·변경 검사 통과: 토큰 임베딩·위치 임베딩·블록·어휘 출력층
학습 전 loss: 1.8279
200회 갱신 후 loss: 0.1662
gpu -> current | 정답: current
gpu current -> status | 정답: status
gpu current status -> failed | 정답: failed
gpu current status failed -> <eos> | 정답: <eos>
network -> current | 정답: current
network current -> status | 정답: status
network current status -> healthy | 정답: healthy
network current status healthy -> <eos> | 정답: <eos>
학습 위치 정답 수: 8/8
검사 통과: 미래 차단·임베딩 학습·다음 토큰 학습 연결. 저장·생성은 없습니다
```

각 줄의 문맥은 **정답 입력 문장의 접두 부분**입니다. 예측한 current를 다음 입력으로 붙여 반복 생성한 결과가 아닙니다. 학습 중에는 정답 토큰으로 구성한 입력을 제공하는 teacher forcing을 사용하고, 결과도 같은 학습 입력에서 확인합니다. `<eos>`를 정답으로 맞혔다는 것과 실제 생성 루프가 종료한다는 검증은 다릅니다.

두 문장에서 마지막 공통 토큰은 status이지만 `gpu current status → failed`, `network current status → healthy`를 구분하는지 확인합니다. 작은 학습 데이터의 적합 결과이며 새 문장에 대한 언어 이해·일반화·운영 품질을 검증하지 않습니다. 학습 전후 loss는 같은 모델·같은 여덟 정답 위치의 평균을 비교합니다.

로컬 환경에 NumPy가 없어 PyTorch import 시 초기화 경고가 있었지만 NumPy 변환을 사용하지 않아 실제 학습·검사를 완료했습니다. import 실패·assert 실패와 구분합니다. 실패하면 traceback과 직전 출력을 확인하며 검사를 삭제하지 않습니다.

## 6. 검사·한계·다음 단계

[독립 검사](source/09-transformer/test_language_model.py)는 실제 x/y를 토큰으로 복원해 한 칸 이동과 문장 경계를 확인합니다. batch를 각각 계산한 결과, 미래 입력을 바꾼 결과, 각 길이의 접두 문맥만 계산한 결과도 비교합니다. 개발용 PyTorch 환경에서 저장소 루트의 `python source/09-transformer/test_language_model.py`로 실행합니다. 이 명령은 이번 메모리 학습을 다시 실행하지만 체크포인트를 읽거나 쓰지 않습니다.

- 최대 네 입력 위치·단일 head·한 블록·동일 길이 두 문장 전용입니다.
- padding·미등록 토큰 처리·tokenizer 학습·multi-head·Attention 출력 사영 학습·최종 LayerNorm·dropout은 추가하지 않았습니다.
- 학습/검증/테스트를 나누지 않았고, 학습 예문 재현만 확인합니다.
- 모델은 메모리에만 있습니다. 명령 종료 뒤 다른 프로세스에서 복원할 파일은 없습니다.
- 다음 목표는 별도 체크포인트를 안전하게 저장하고 같은 어휘·구조로 복원한 뒤 생성 경로를 확인하는 것입니다. 이번에는 구현하지 않습니다.

- [ ] 토큰 ID와 그 ID로 찾는 학습 벡터를 구분한다.
- [ ] 위치 임베딩을 더해도 좌표 수가 유지되는 이유를 설명한다.
- [ ] 정답을 한 칸 이동하는 코드와 loss에서 펼치는 축을 찾는다.
- [ ] 정답 문맥에서 각 위치 예측과 이전 예측을 이어가는 생성을 구분한다.
- [ ] 8/8이 학습 예제 재현이지 일반화 성능이 아님을 설명한다.

공식 참고: [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

[← 이전: 작은 Transformer 블록](09-transformer.md) · [목차](README.md)
