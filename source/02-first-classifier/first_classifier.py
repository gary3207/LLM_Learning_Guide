import torch
from torch import nn

torch.manual_seed(42)
assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
device = "cuda"

# 1. 학습용 문의와 정답: GPU=0, 네트워크=1, 스토리지=2
labels = ["GPU", "네트워크", "스토리지"]
data = [
    ("gpu memory error", 0),
    ("gpu device unavailable", 0),
    ("cuda memory unavailable", 0),
    ("network connection error", 1),
    ("network timeout", 1),
    ("connection timeout", 1),
    ("disk storage full", 2),
    ("disk space error", 2),
    ("storage space full", 2),
]

# 2. 문장을 단어별 등장 횟수로 변환
vocab = sorted({word for text, _ in data for word in text.split()})

def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]

x = torch.tensor(
    [encode(text) for text, _ in data],
    dtype=torch.float32, device=device
)
y = torch.tensor(
    [label for _, label in data],
    dtype=torch.long, device=device
)

# 3. 모델과 학습 방법 준비
model = nn.Linear(len(vocab), len(labels)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

initial_loss = loss_fn(model(x), y).item()
print(f"학습 전 loss: {initial_loss:.4f}")

# 4. 전체 학습 데이터를 100번 반복
model.train()
for epoch in range(1, 101):
    scores = model(x)
    loss = loss_fn(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            current_loss = loss_fn(model(x), y).item()
        print(f"epoch {epoch:3d} | loss: {current_loss:.4f}")

# 5. 학습 결과 확인
model.eval()
with torch.no_grad():
    final_scores = model(x)
    final_loss = loss_fn(final_scores, y).item()
    accuracy = (final_scores.argmax(dim=1) == y).float().mean().item()
    print(f"\n학습 데이터 정확도: {accuracy:.1%}")

    for text in ["gpu memory", "network connection", "disk full"]:
        sample = torch.tensor(
            [encode(text)], dtype=torch.float32, device=device
        )
        prediction = model(sample).argmax(dim=1).item()
        print(f"{text} -> {labels[prediction]}")

assert final_loss < initial_loss, "학습 후 오차가 줄지 않았습니다."
print("\n검증 통과: 학습 후 오차 감소")
