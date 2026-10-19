from pathlib import Path

import torch
from torch import nn

# 1. 저장된 모델 복원
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

# 2. 이전에 오분류한 문장을 같은 방식으로 숫자 변환
text = "network unavailable"
words = text.lower().split()
inputs = torch.tensor(
    [words.count(word) for word in vocab],
    dtype=torch.float32,
)

with torch.no_grad():
    # 분류별·단어별 기여도 = 학습된 가중치 × 단어 등장 횟수
    contributions = model.weight * inputs

    # 분류별 최종 점수 = 모든 단어의 기여도 합 + 편향
    calculated_scores = contributions.sum(dim=1) + model.bias
    model_scores = model(inputs)

    print("분석할 문장:", text)
    print("분류 순서:", " | ".join(labels))
    print("\n[1] 각 단어가 분류별 점수에 더하는 값")

    for index, word in enumerate(vocab):
        if inputs[index].item() > 0:
            values = contributions[:, index]
            formatted = " | ".join(f"{value.item():+.4f}" for value in values)
            print(f"{word}: {formatted}")

    print("\n[2] 편향과 최종 점수")

    for index, label in enumerate(labels):
        word_total = contributions[index].sum().item()
        bias = model.bias[index].item()
        total = calculated_scores[index].item()
        print(
            f"{label}: 단어 합 {word_total:+.4f}"
            f" + 편향 {bias:+.4f}"
            f" = 최종 {total:+.4f}"
        )

    prediction = model_scores.argmax().item()
    print("\n최종 예측:", labels[prediction])

    # 반올림해 출력하기 전의 실제 숫자로 검사
    assert torch.allclose(
        calculated_scores,
        model_scores,
        rtol=1e-5,
        atol=1e-6,
    ), "직접 합산한 점수와 모델 출력이 다릅니다."

print("\n검증 통과: 단어별 기여도 + 편향 = 모델 출력")
