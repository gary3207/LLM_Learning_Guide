import json
from pathlib import Path

import torch
from torch import nn

root = Path.home() / "llm-lab"
dataset = json.loads(
    (root / "source/classifier_splits.json").read_text(encoding="utf-8")
)
examples = dataset["validation"]
assert examples, "검증 데이터가 비어 있습니다."

for filename in ["classifier.pt", "classifier_augmented.pt"]:
    checkpoint = torch.load(
        root / filename, map_location="cpu", weights_only=True
    )
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]
    assert labels == dataset["labels"], "라벨 순서가 다릅니다."

    model = nn.Linear(len(vocab), len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor(
        [[text.lower().split().count(word) for word in vocab]
         for text, _ in examples],
        dtype=torch.float32,
    )
    y = torch.tensor([label for _, label in examples], dtype=torch.long)

    with torch.no_grad():
        scores = model(x)
        loss = nn.functional.cross_entropy(scores, y).item()
        predictions = scores.argmax(dim=1)
        correct = (predictions == y).sum().item()

    print(f"\n[{filename} / validation]")
    for (text, expected), prediction in zip(examples, predictions.tolist()):
        print(f"{text}: {labels[prediction]} | 의도: {labels[expected]}")

    print(
        f"검증 정확도: {correct}/{len(examples)} "
        f"({correct / len(examples):.1%})"
    )
    print(f"검증 loss: {loss:.4f}")

print("\n검증 완료: 재학습·모델 저장·테스트 예측은 실행하지 않았습니다.")
