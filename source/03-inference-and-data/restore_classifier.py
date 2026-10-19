from pathlib import Path

import torch
from torch import nn

# 1. 저장된 모델 파일 읽기
save_path = Path.home() / "llm-lab" / "classifier.pt"
checkpoint = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True,
)

# 2. 학습에 사용한 단어 목록과 분류 이름 복원
vocab = checkpoint["vocab"]
labels = checkpoint["labels"]

# 3. 같은 구조의 모델을 만들고 학습된 가중치 넣기
model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"모델 불러오기 완료: {save_path}")


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


# 4. 재학습 없이 이전과 같은 문의를 분류
examples = [
    ("gpu memory", "GPU"),
    ("network connection", "네트워크"),
    ("disk full", "스토리지"),
]

with torch.no_grad():
    for text, expected in examples:
        inputs = torch.tensor([encode(text)], dtype=torch.float32)
        prediction = model(inputs).argmax(dim=1).item()
        result = labels[prediction]

        print(f"{text} -> {result}")
        assert result == expected, f"예상과 다른 결과: {text} -> {result}"

print("\n검증 통과: 저장한 모델을 재학습 없이 불러와 예측 성공")
