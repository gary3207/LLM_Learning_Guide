# 원문 Python 본문 보존; 출처 주석만 추가.
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
