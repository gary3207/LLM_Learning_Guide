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
