from pathlib import Path
import torch
from torch import nn

checkpoint = torch.load(
    Path.home() / "llm-lab/next_token.pt",
    map_location="cpu",
    weights_only=True,
)
vocab = checkpoint["vocab"]
assert isinstance(vocab, list) and vocab
assert all(isinstance(token, str) for token in vocab)
assert len(vocab) == len(set(vocab)) and "<eos>" in vocab

to_id = {token: index for index, token in enumerate(vocab)}
model = nn.Embedding(len(vocab), len(vocab))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(start, max_new_tokens=10):
    if start not in to_id or start == "<eos>":
        raise ValueError(f"시작에 사용할 수 없는 토큰: {start}")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens는 양의 정수여야 합니다.")

    words = [start]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            current = torch.tensor([to_id[words[-1]]], dtype=torch.long)
            scores = model(current)
            assert torch.isfinite(scores).all().item(), "잘못된 예측 점수"
            next_token = vocab[scores.argmax(dim=1).item()]

            if next_token == "<eos>":
                return " ".join(words), "<eos>"
            words.append(next_token)

    return " ".join(words), "길이 제한"

for start, expected in [
    ("gpu", "gpu memory error"),
    ("network", "network connection error"),
    ("disk", "disk storage full"),
]:
    text, reason = generate(start)
    print(f"{start} -> {text} | 종료: {reason}")
    assert text == expected and reason == "<eos>", "학습 문장 재현 확인 실패"

assert generate("gpu", max_new_tokens=1)[1] == "길이 제한"
print("검사 통과: 학습 문장 재현, 종료 토큰, 생성 길이 제한")
print("재학습·모델 파일 변경 없음")
