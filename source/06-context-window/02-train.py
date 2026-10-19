# 원문 Python 본문 보존; 출처 주석만 추가.
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
