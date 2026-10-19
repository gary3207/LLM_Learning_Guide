from pathlib import Path

import torch
from torch import nn

# 1. 이전 실습에서 직접 저장한 모델 복원
save_path = Path.home() / "llm-lab" / "classifier.pt"
checkpoint = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True,
)

vocab = checkpoint["vocab"]
labels = checkpoint["labels"]

model = nn.Linear(len(vocab), len(labels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def encode(text):
    words = text.lower().split()
    return [words.count(word) for word in vocab]


print("모델 불러오기 완료")
print("모델이 아는 단어:", ", ".join(vocab))

# 2. 학습 문장과 다른 조합으로 예측
examples = [
    ("cuda error", "GPU"),
    ("network unavailable", "네트워크"),
    ("disk full error", "스토리지"),
]

print("\n[1] 새로운 문장 예측")

with torch.no_grad():
    for text, expected in examples:
        inputs = torch.tensor([encode(text)], dtype=torch.float32)
        result = labels[model(inputs).argmax(dim=1).item()]
        mark = "일치" if result == expected else "불일치"
        print(f"{text} -> {result} | 의도: {expected} | {mark}")

    # 3. 단어 목록에 없는 문장
    text = "internet disconnected"
    vector = encode(text)
    unknown = [word for word in text.split() if word not in vocab]
    inputs = torch.tensor([vector], dtype=torch.float32)
    result = labels[model(inputs).argmax(dim=1).item()]

    print("\n[2] 모르는 단어 시험")
    print("입력:", text)
    print("모르는 단어:", unknown)
    print("변환된 숫자:", vector)
    print("모델의 예측:", result)
    assert not any(vector), "예상과 달리 등록된 단어가 있습니다."

    # 4. 단어 순서만 바꿔 비교
    first = encode("gpu memory")
    second = encode("memory gpu")

    print("\n[3] 단어 순서 시험")
    print("gpu memory:", first)
    print("memory gpu:", second)
    print("두 입력의 숫자가 같은가?:", first == second)
    assert first == second, "예상과 달리 입력 표현이 다릅니다."

print("\n검사 완료: 미등록 단어와 단어 순서의 처리 방식 확인")
