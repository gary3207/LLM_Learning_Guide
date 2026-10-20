# 09 · 작은 Transformer 블록

[← 이전: Attention 입문](08-attention.md) · [목차](README.md)

## 1. 목표와 이전 실습과의 연결

08장 마지막 실습은 `X → causal Attention → 마지막 hidden → 어휘 점수 → loss`를 연결했습니다. 이번에는 **잔차 연결 두 개·LayerNorm 두 개·Feed-forward network(FFN)**를 붙여, 작은 단일-head causal Transformer 블록을 한 번 학습합니다.

입력 X·초기 QKV 행렬·초기 어휘 행렬·교육용 정답 failed·학습률 0.01은 앞과 같습니다. 앞 실습에서 갱신한 값을 가져오는 것이 아니라 **원래 초기값에서 새로 시작**합니다. 정규화를 Attention **앞에** 두는 Pre-LN 구조이므로 같은 X·행렬이어도 QKV·hidden·초기 loss는 앞 실습과 달라집니다. 이것은 기존 모델보다 성능이 좋아졌다는 비교가 아닙니다.

```text
                 ┌────────── 더하기 ──────────┐
X → LayerNorm1 → causal Attention             ↓
└──────────────────────────────────────────→ R
                 ┌────────── 더하기 ──────────┐
R → LayerNorm2 → Linear(2→4) → ReLU → Linear(4→2)
└──────────────────────────────────────────→ H
H 마지막 행 → 어휘 Linear(2→3) → logits → loss
```

수식으로는 `R = X + Attention(LN1(X))`, `H = R + FFN(LN2(R))`입니다. 입력은 위치 정보가 포함된 `gpu current gpu`의 교육용 고정 벡터입니다. 학습되는 토큰 임베딩이나 실제 문장 정답 데이터는 아직 없습니다.

## 2. 새 구성 세 가지

### 잔차 연결: 입력을 없애지 않고 계산 결과 더하기

`[1, 2] + [0.2, -0.1] = [1.2, 1.9]`처럼 같은 위치·좌표끼리 더합니다. 두 벡터를 이어 붙이지 않습니다. 입력이 `3×2`이면 Attention·FFN 출력도 `3×2`여야 더할 수 있습니다. 앞 실습은 Attention 출력 자체가 hidden이었지만 이번에는 기존 입력에 새 계산을 더합니다.

### LayerNorm: 각 토큰 벡터 내부 좌표 정규화

`nn.LayerNorm(2)`는 각 행의 두 좌표로 평균·분산을 계산합니다. 세 위치를 함께 평균내는 것이 아닙니다. 기본 수식은 `(x - 평균) / sqrt(분산 + eps) × gamma + beta`이며 gamma·beta도 학습합니다. 분산은 좌표 개수로 나눕니다.

예를 들어 `[1, 3]`의 평균은 2, 분산은 1이고 초기 gamma=1·beta=0일 때 `[-1, 1] / sqrt(1 + 1e-5)`가 됩니다. **확률 변환이 아니므로 음수가 가능하고 합 1일 필요가 없습니다.** eps는 나눗셈 안정화를 위한 값이며 학습률이 아닙니다. 이번에는 Attention과 FFN 각각의 입력을 정규화합니다. Post-LN과 순서를 혼동하지 않습니다.

### FFN: 각 위치의 좌표를 같은 작은 신경망으로 변환

`2 → 4 → 2`로 좌표 수를 늘렸다 되돌립니다. 가운데 `ReLU`는 음수를 0으로 바꾸는 비선형 연산입니다. 예를 들어 `[-1, 2, 0, 3] → [0, 2, 0, 3]`입니다. FFN은 모든 위치에 같은 가중치를 적용하지만 **다른 위치를 섞지 않습니다**. 위치 사이 정보 혼합은 Attention이 맡습니다.

## 3. 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source`의 PyTorch 가상환경에서 실행합니다. CPU·float64 계산이며 이전 셸의 변수나 모델 파일은 필요 없습니다. **설치·GPU 사용·파일 읽기·저장·체크포인트 덮어쓰기는 없습니다.** 메모리의 작은 블록만 한 번 갱신합니다.

- `cd`: 실습 폴더로 이동합니다.
- `source .venv/bin/activate`: 기존 가상환경의 Python을 사용합니다.
- `python --version`: 활성화한 Python 버전을 확인합니다.
- `python - <<'PY'`부터 마지막 `PY`까지가 한 번에 실행할 전체 코드입니다.
- `torch`를 못 찾으면 설치부터 반복하지 말고 기존 가상환경과 Python 경로를 확인합니다. `python -O`는 assert를 생략하므로 사용하지 않습니다.

**소스:** [01-transformer-block.py](source/09-transformer/01-transformer-block.py)

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

python - <<'PY'
from math import sqrt
import torch
from torch import nn

torch.manual_seed(7)
X = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
                 dtype=torch.float64, device="cpu")
vocabulary = ["failed", "healthy", "<eos>"]
y = torch.tensor([vocabulary.index("failed")], dtype=torch.long, device="cpu")


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        options = {"dtype": torch.float64, "device": "cpu"}
        self.norm1 = nn.LayerNorm(2, eps=1e-5, **options)
        self.norm2 = nn.LayerNorm(2, eps=1e-5, **options)
        self.q = nn.Linear(2, 2, bias=False, **options)
        self.k = nn.Linear(2, 2, bias=False, **options)
        self.v = nn.Linear(2, 2, bias=False, **options)
        self.ff = nn.Sequential(nn.Linear(2, 4, **options), nn.ReLU(),
                                nn.Linear(4, 2, **options))
        self.head = nn.Linear(2, 3, bias=False, **options)
        # nn.Linear는 x @ weight.T이므로 앞 실습의 행렬을 전치해 복사합니다.
        with torch.no_grad():
            for layer, values in [(self.q, [[1.0, 0.0], [1.0, 1.0]]),
                                  (self.k, [[0.0, 1.0], [1.0, 0.0]]),
                                  (self.v, [[2.0, 0.0], [0.0, 3.0]]),
                                  (self.head, [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]])]:
                layer.weight.copy_(torch.tensor(values, **options).T)

    def forward(self, x):
        # ponytail: 단일 head·3위치×2좌표 전용; batch·multi-head 확장은 후속 실습입니다.
        normalized = self.norm1(x)
        Q, K, V = self.q(normalized), self.k(normalized), self.v(normalized)
        future = torch.ones((3, 3), dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = (Q @ K.T) / sqrt(Q.shape[-1])
        attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        residual = x + attention @ V  # Attention 출력 사영은 항등, 학습 행렬은 생략합니다.
        hidden = residual + self.ff(self.norm2(residual))
        return attention, hidden, self.head(hidden[-1:])


model = TinyBlock()
loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=0.0, weight_decay=0.0)
optimizer.zero_grad(set_to_none=True)
attention, hidden, logits = model(X)
assert attention.shape == (3, 3) and hidden.shape == (3, 2) and logits.shape == (1, 3)
assert torch.all(attention.triu(diagonal=1) == 0.0)
assert torch.allclose(attention.sum(-1), torch.ones(3, dtype=torch.float64))
# 마지막 입력만 바꿔도 앞 두 위치 출력이 바뀌지 않는지 검사합니다.
with torch.no_grad():
    changed = X.clone()
    changed[-1] += torch.tensor([3.0, -2.0], dtype=torch.float64)
    _, changed_hidden, _ = model(changed)
    assert torch.allclose(hidden[:2], changed_hidden[:2], rtol=0.0, atol=1e-12)
loss = loss_fn(logits, y)
before = {name: p.detach().clone() for name, p in model.named_parameters()}
loss.backward()
for name, p in model.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert p.grad.norm().item() > 0.0, name
    assert torch.equal(p.detach(), before[name]), name
optimizer.step()
for name, p in model.named_parameters():
    assert torch.allclose(p.detach(), before[name] - learning_rate * p.grad, rtol=0.0, atol=1e-12), name
    assert not torch.equal(p.detach(), before[name]), name
with torch.no_grad():
    _, new_hidden, new_logits = model(X)
    new_loss = loss_fn(new_logits, y).item()
    old_probability = torch.softmax(logits.detach(), dim=-1)[0, y.item()].item()
    new_probability = torch.softmax(new_logits, dim=-1)[0, y.item()].item()
assert torch.isfinite(new_hidden).all() and torch.isfinite(new_logits).all()
assert new_loss < loss.item() and new_probability > old_probability
assert not X.requires_grad and X.grad is None
print("교육용 정답 토큰:", vocabulary[y.item()])
print("shape: X=(3, 2), Attention=(3, 3), hidden=(3, 2), logits=(1, 3)")
print("미래 차단 검사 통과: 마지막 입력을 바꿔도 앞 두 위치 출력 유지")
print("갱신 전 마지막 hidden:", [round(v, 4) for v in hidden.detach()[-1].tolist()])
print("갱신 후 마지막 hidden:", [round(v, 4) for v in new_hidden[-1].tolist()])
print("기울기·SGD 갱신 검사 통과: QKV·LayerNorm 2개·FFN·어휘 출력층")
print(f"failed 확률: {old_probability:.4f} → {new_probability:.4f}")
print(f"loss: {loss.item():.4f} → {new_loss:.4f}")
print("검사 통과: 작은 Transformer 블록을 거쳐 한 번 갱신했습니다. 저장·생성은 없습니다")
PY
```

## 4. 코드를 따라 읽기

| 코드 | 역할·크기 |
|---|---|
| `TinyBlock(nn.Module)` | 여러 층을 하나의 모듈로 묶음. `super().__init__()`로 부모 초기화 |
| `__init__` / `forward` | 처음 한 번 층을 만들기 / `model(X)` 때 현재 파라미터로 계산하기 |
| `torch.manual_seed(7)` | FFN 등의 초기화 난수 고정. QKV·어휘 가중치는 이후 앞 값으로 교체 |
| `X` | 3위치×2좌표, 고정 토큰+위치 입력 |
| `norm1` / `norm2` | 각각의 행 내부 두 좌표 정규화, 출력 크기 유지 |
| `q`, `k`, `v` | 각각 `3×2 → 3×2`, bias 없음 |
| `scores`, `attention` | 각각 `3×3`, 마지막 축은 참조할 입력 위치 |
| `residual` | `X + attention @ V`, `3×2` |
| `ff` | `3×2 → 3×4 → 3×2`, 두 Linear의 bias 포함 |
| `hidden` | 두 번째 잔차까지 더한 블록 출력, `3×2` |
| `head(hidden[-1:])` | 마지막 행 `1×2 → 1×3`, 어휘 점수. `[-1:]`은 행 차원 유지 |
| `y` | failed의 번호 하나, shape `(1,)`, dtype long |

앞 행벡터 계산은 `X @ W`였습니다. **`nn.Linear`는 `x @ weight.T + bias`**를 계산하므로 기존 W를 `.T`로 전치해 `weight`에 복사합니다. 어휘 head의 weight 크기도 기존 `2×3`이 아니라 `3×2`입니다. 초기값 복사는 `no_grad()` 안에서 하지만 이후 순전파·역전파에서는 정상적으로 학습합니다.

`self.head`는 세 후보의 어휘 점수를 만드는 층입니다. Attention의 head 출력 결합용 사영과는 다릅니다. 이 교육용 단일 head에서는 `attention @ V` 뒤의 학습 가능한 출력 사영을 생략해 항등 변환으로 둡니다. 표준 multi-head 구현 전체를 복제한 것은 아닙니다. 블록 뒤 별도 최종 LayerNorm도 이번에는 추가하지 않습니다.

`model.parameters()`는 등록된 QKV·LayerNorm·FFN·어휘 출력층의 학습 파라미터를 모두 optimizer에 전달합니다. `zero_grad → forward → loss → backward → step → 새 forward`의 순서는 앞과 같습니다. loss에는 softmax 전 logits를 전달하며, softmax는 비교용 어휘 확률 출력에서만 사용합니다.

### 미래를 안 보는지 동작으로 확인

세 번째 입력만 `[3, -2]`만큼 바꾸어 순전파합니다. causal mask와 위치별 LayerNorm·FFN이 올바르게 연결되어 있으면 **첫 두 위치 출력은 변하지 않아야 합니다**. 이것은 한 고정 입력의 동작 검사이며 모든 길이·batch의 일반적 정확성을 증명하지 않습니다.

## 5. 로컬 실제 실행 예시

로컬 격리 환경의 실제 CPU·float64 PyTorch 실행으로 assert를 통과한 참고 출력입니다. VM 실행 완료를 뜻하지 않습니다. seed를 고정해도 버전·장치·초기화 구현에 따라 수치가 다를 수 있으며, 아래 숫자를 assert의 고정 정답으로 강제하지 않습니다. 표시값은 소수 네 자리입니다.

```text
교육용 정답 토큰: failed
shape: X=(3, 2), Attention=(3, 3), hidden=(3, 2), logits=(1, 3)
미래 차단 검사 통과: 마지막 입력을 바꿔도 앞 두 위치 출력 유지
갱신 전 마지막 hidden: [2.1445, 0.8979]
갱신 후 마지막 hidden: [2.2036, 0.8151]
기울기·SGD 갱신 검사 통과: QKV·LayerNorm 2개·FFN·어휘 출력층
failed 확률: 0.8353 → 0.8509
loss: 0.1800 → 0.1614
검사 통과: 작은 Transformer 블록을 거쳐 한 번 갱신했습니다. 저장·생성은 없습니다
```

이번 초기 정답 확률이 앞 실습보다 큰 것은 **정규화·잔차·임의 초기 FFN을 추가해 다른 함수를 계산했기 때문**입니다. 구조의 성능 우위를 검증한 것이 아닙니다. 한 번 갱신 뒤 loss가 줄어든 것은 이 한 입력·교육용 정답에 대한 결과이며 모든 학습에서 매번 감소한다고 보장하지 않습니다.

로컬 검증 환경은 NumPy가 없어 import 때 초기화 경고를 냈지만, 이 코드는 NumPy 변환을 사용하지 않아 실제 텐서 계산·검사를 완료했습니다. Python/torch import 실패나 assert 실패와 구분합니다. 실패하면 traceback과 직전 출력으로 원인을 확인하며 검사를 삭제해 통과시키지 않습니다.

## 6. 검사·한계·복습

실습 자체는 shape·미래 비중 0·행 합 1·미래 입력 변경의 영향·각 파라미터의 유한한 비영 기울기·backward 후 불변·SGD 후 실제 갱신·새 loss를 검사합니다. [독립 검사](source/09-transformer/test_block.py)는 LayerNorm 수식·두 잔차·FFN·어휘 출력의 별도 계산을 실제 블록과 비교합니다. 저장소를 받은 개발용 PyTorch 환경에서 `python source/09-transformer/test_block.py`로 실행합니다. 앞 장의 모델 학습은 실행하지 않습니다.

- 고정 3위치·2좌표·단일 head·단일 블록·마지막 위치 한 정답 전용입니다.
- 2좌표 정규화는 서로 다른 입력을 비슷한 방향으로 압축할 수 있어 실용적인 표현 차원 설계가 아닙니다.
- dropout·학습 가능한 Attention 출력 사영·별도 최종 LayerNorm은 생략했습니다.
- 학습되는 임베딩·여러 문장/위치의 학습·별도 검증·저장·복원·문장 생성은 아직 없습니다.
- 이 블록을 완성된 언어 모델이나 챗봇으로 보지 않습니다.

- [ ] 잔차 덧셈과 벡터 이어 붙이기를 구분한다.
- [ ] LayerNorm이 위치 축이 아니라 각 토큰의 좌표 축에서 계산되는 이유를 설명한다.
- [ ] FFN과 Attention 중 다른 위치의 정보를 섞는 쪽을 고른다.
- [ ] 기존 행렬을 nn.Linear에 복사할 때 전치하는 이유를 설명한다.
- [ ] 미래 차단·한 번 갱신·일반화 평가가 서로 다른 검사임을 설명한다.

다음 [10장](10-embedding-next-token.md)에서는 **학습되는 토큰·위치 임베딩과 실제 다음 토큰 학습 쌍을 이 블록 구조에 연결**합니다.

## 공식 근거

- [LayerNorm 수식·축·학습 파라미터](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Linear의 weight 저장 규약](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [TransformerEncoderLayer 구성과 norm_first](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — Pre-LN 순서 참고이며 이 실습이 해당 클래스 전체와 같다는 뜻은 아닙니다.

[← 이전: Attention 입문](08-attention.md) · [목차](README.md) · [다음: 임베딩과 다음 토큰 학습 →](10-embedding-next-token.md)
