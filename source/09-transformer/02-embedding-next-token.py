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
