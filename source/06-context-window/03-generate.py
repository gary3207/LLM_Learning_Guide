from pathlib import Path
import torch
from torch import nn

path = Path.home() / "llm-lab/next_token_context2.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=True)

vocab = checkpoint["vocab"]
assert isinstance(vocab, list) and vocab
assert all(isinstance(token, str) for token in vocab)
assert len(vocab) == len(set(vocab))
assert "<bos>" in vocab and "<eos>" in vocab
assert checkpoint["context_size"] == 2
assert checkpoint["embedding_dim"] == 8

to_id = {token: i for i, token in enumerate(vocab)}
model = nn.Sequential(
    nn.Embedding(len(vocab), 8),
    nn.Flatten(start_dim=1),
    nn.Linear(16, len(vocab)),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(start, max_new_tokens=10):
    if start not in to_id or start in ("<bos>", "<eos>"):
        raise ValueError(f"시작에 사용할 수 없는 토큰: {start}")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens는 양의 정수여야 합니다")

    history = ["<bos>", start]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = history[-2:]
            x = torch.tensor([[to_id[t] for t in context]])
            scores = model(x)
            assert torch.isfinite(scores).all().item()

            next_token = vocab[scores.argmax(dim=1).item()]

            if next_token == "<eos>":
                return " ".join(history[1:]), "<eos>"
            if next_token == "<bos>":
                raise RuntimeError("생성 도중 시작 토큰이 나왔습니다")

            history.append(next_token)

    return " ".join(history[1:]), "길이 제한"

for start, expected in [
    ("gpu", "gpu status failed"),
    ("network", "network status healthy"),
]:
    text, reason = generate(start)
    print(f"{start} -> {text} | 종료: {reason}")
    assert text == expected and reason == "<eos>"

assert generate("gpu", max_new_tokens=1) == ("gpu status", "길이 제한")

print("검사 통과: 문맥별 문장 재현, 종료 토큰, 생성 길이 제한")
print("재학습·모델 파일 변경 없음")
