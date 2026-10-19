from pathlib import Path

import torch
from torch import nn

root = Path.home() / "llm-lab"
models = []

for filename in ["classifier.pt", "classifier_augmented.pt"]:
    checkpoint = torch.load(
        root / filename, map_location="cpu", weights_only=True
    )
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]

    model = nn.Linear(len(vocab), len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append((model, vocab, labels))

assert models[0][1:] == models[1][1:], (
    "두 모델의 어휘 또는 분류 순서가 다릅니다."
)

groups = {
    "기존 학습 문장": [
        ("gpu memory error", "GPU"),
        ("gpu device unavailable", "GPU"),
        ("cuda memory unavailable", "GPU"),
        ("network connection error", "네트워크"),
        ("network timeout", "네트워크"),
        ("connection timeout", "네트워크"),
        ("disk storage full", "스토리지"),
        ("disk space error", "스토리지"),
        ("storage space full", "스토리지"),
    ],
    "추가한 학습 문장": [
        ("network unavailable", "네트워크"),
    ],
    "두 모델의 학습 데이터에 없는 문장": [
        ("network error", "네트워크"),
        ("connection unavailable", "네트워크"),
        ("disk unavailable", "스토리지"),
        ("cuda error", "GPU"),
    ],
}

with torch.no_grad():
    for group, examples in groups.items():
        print(f"\n[{group}]")
        correct = [0, 0]

        for text, expected in examples:
            predictions = []

            for index, (model, vocab, labels) in enumerate(models):
                words = text.lower().split()
                inputs = torch.tensor(
                    [words.count(word) for word in vocab],
                    dtype=torch.float32,
                )
                prediction = labels[model(inputs).argmax().item()]
                predictions.append(prediction)
                correct[index] += int(prediction == expected)

            print(
                f"{text}: {predictions[0]} -> {predictions[1]}"
                f" | 의도: {expected}"
            )

        print(
            f"의도와 일치: 보완 전 {correct[0]}/{len(examples)}"
            f" | 보완 후 {correct[1]}/{len(examples)}"
        )

print("\n비교 완료: 두 모델 파일은 읽기만 했습니다.")
