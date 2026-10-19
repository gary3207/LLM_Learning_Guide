from pathlib import Path

import torch
from torch import nn

torch.manual_seed(42)
assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
device = "cuda"

# 1. 이전 실습과 동일한 학습 데이터
labels = ["GPU", "네트워크", "스토리지"]
data = [
    ("gpu memory error", 0),
    ("gpu device unavailable", 0),
    ("cuda memory unavailable", 0),
    ("network connection error", 1),
    ("network timeout", 1),
    ("connection timeout", 1),
    ("network unavailable", 1),
    ("disk storage full", 2),
    ("disk space error", 2),
    ("storage space full", 2),
]

vocab = sorted({word for text, _ in data for word in text.split()})


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


# 2. 학습 데이터를 GPU 텐서로 변환
x = torch.tensor(
    [encode(text) for text, _ in data],
    dtype=torch.float32,
    device=device,
)
y = torch.tensor(
    [label for _, label in data],
    dtype=torch.long,
    device=device,
)

# 기존 모델 파일이 있으면 덮어쓰지 않음
save_path = Path(__file__).resolve().parent.parent / "classifier_augmented.pt"
if save_path.exists():
    raise FileExistsError(f"기존 모델을 보호하기 위해 중단합니다: {save_path}")

# 3. 모델과 학습 방법 준비
model = nn.Linear(len(vocab), len(labels)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

initial_loss = loss_fn(model(x), y).item()
print(f"학습 전 loss: {initial_loss:.4f}")

# 4. 학습
model.train()
for epoch in range(1, 101):
    scores = model(x)
    loss = loss_fn(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | loss: {loss.item():.4f}")

# 5. 학습 결과 확인
model.eval()
with torch.no_grad():
    final_loss = loss_fn(model(x), y).item()
    predictions = model(x).argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    print(f"\n학습 데이터 정확도: {accuracy:.1%}")

    for text in ["gpu memory", "network connection", "disk full"]:
        inputs = torch.tensor(
            [encode(text)], dtype=torch.float32, device=device
        )
        prediction = model(inputs).argmax(dim=1).item()
        print(f"{text} -> {labels[prediction]}")

assert final_loss < initial_loss, "학습 후 오차가 줄지 않았습니다."
print("\n검증 통과: 학습 후 오차 감소")

# 6. 가중치와 문장 해석에 필요한 정보를 함께 저장
checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "labels": labels,
}
with save_path.open("xb") as file:
    torch.save(checkpoint, file)

assert save_path.stat().st_size > 0, "모델 파일이 비어 있습니다."
print(f"모델 저장 완료: {save_path}")
