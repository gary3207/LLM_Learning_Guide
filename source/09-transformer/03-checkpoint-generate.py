"""10장과 같은 모델을 재현해 별도 저장하거나, 재학습 없이 복원·생성합니다."""
import argparse
from math import sqrt
from pathlib import Path
import torch
from torch import nn

vocab = ["<eos>", "current", "failed", "gpu", "healthy", "network", "status"]
to_id = {token: i for i, token in enumerate(vocab)}
architecture = "tiny-preln-d4-ff8-context4-singlehead-v1"


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        options = {"dtype": torch.float64, "device": "cpu"}
        self.token = nn.Embedding(len(vocab), 4, **options)
        self.position = nn.Embedding(4, 4, **options)
        self.norm1 = nn.LayerNorm(4, eps=1e-5, **options)
        self.norm2 = nn.LayerNorm(4, eps=1e-5, **options)
        self.q = nn.Linear(4, 4, bias=False, **options)
        self.k = nn.Linear(4, 4, bias=False, **options)
        self.v = nn.Linear(4, 4, bias=False, **options)
        self.ff = nn.Sequential(nn.Linear(4, 8, **options), nn.ReLU(),
                                nn.Linear(8, 4, **options))
        self.head = nn.Linear(4, len(vocab), bias=False, **options)

    def forward(self, ids):
        # ponytail: 최대4토큰·단일head·padding 없음; 긴 문맥은 후속 범위입니다.
        assert ids.ndim == 2 and ids.shape[0] > 0 and 1 <= ids.shape[1] <= 4
        assert ids.dtype == torch.long and torch.all((ids >= 0) & (ids < len(vocab)))
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.token(ids) + self.position(positions)
        normalized = self.norm1(embedded)
        Q, K, V = self.q(normalized), self.k(normalized), self.v(normalized)
        scores = (Q @ K.transpose(-2, -1)) / sqrt(Q.shape[-1])
        future = torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool, device=ids.device).triu(1)
        attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        residual = embedded + attention @ V
        hidden = residual + self.ff(self.norm2(residual))
        return self.head(hidden), attention


def save(path):
    # xb는 이미 있는 파일·심볼릭 링크를 덮어쓰지 않습니다. 학습 전에 확보합니다.
    with path.open("xb") as stream:
        torch.manual_seed(7)
        sentences = ["gpu current status failed", "network current status healthy"]
        sequences = [s.split() + ["<eos>"] for s in sentences]
        x = torch.tensor([[to_id[t] for t in seq[:-1]] for seq in sequences])
        y = torch.tensor([[to_id[t] for t in seq[1:]] for seq in sequences])
        model = TinyLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
        loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
        model.train()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            reference_logits, _ = model(x)
            final_loss = loss_fn(reference_logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
        assert torch.isfinite(reference_logits).all()
        assert torch.equal(reference_logits.argmax(-1), y)
        torch.save({"architecture": architecture, "vocab": vocab,
                    "state_dict": model.state_dict(), "reference_logits": reference_logits}, stream)
    print(f"200회 재현 학습 후 loss: {final_loss:.4f} | 학습 위치: 8/8")
    print(f"별도 저장 완료: {path.name}")


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != architecture or payload.get("vocab") != vocab:
        raise ValueError("이 실습의 모델 구조·어휘 순서와 다른 체크포인트입니다")
    model = TinyLanguageModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("가중치에 NaN 또는 무한대가 있습니다")
    model.eval()
    with torch.no_grad():
        probes = torch.tensor([[to_id[t] for t in s.split()] for s in
                              ["gpu current status failed", "network current status healthy"]])
        logits, _ = model(probes)
    if not torch.allclose(logits, payload["reference_logits"], rtol=0.0, atol=1e-12):
        raise ValueError("저장 전·복원 후 logits가 다릅니다")
    return model


@torch.no_grad()
def generate(model, prompt, max_new_tokens=4):
    if not isinstance(prompt, list) or not 1 <= len(prompt) <= 4 or any(
            not isinstance(t, str) or t not in to_id or t == "<eos>" for t in prompt):
        raise ValueError("시작 문맥은 eos를 제외한 등록 토큰 1~4개의 리스트여야 합니다")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4:
        raise ValueError("최대 생성 수는 정수 1~4여야 합니다")
    tokens = prompt.copy()
    for _ in range(max_new_tokens):
        # ponytail: 학습 위치는0~3뿐; 전체 문맥4개를 넘으면 중단, sliding window는 미검증입니다.
        if len(tokens) > 4:
            return tokens, "context"
        ids = torch.tensor([[to_id[t] for t in tokens]], dtype=torch.long)
        logits, _ = model(ids)
        if not torch.isfinite(logits).all():
            raise ValueError("생성 점수가 유한하지 않습니다")
        next_token = vocab[logits[0, -1].argmax().item()]
        tokens.append(next_token)
        if next_token == "<eos>":
            return tokens, "eos"
    return tokens, "limit"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["save", "generate"])
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "save":
        save(args.checkpoint)
    else:
        model = restore(args.checkpoint)
        print("복원 검사 통과: 같은 구조·어휘·저장 전후 logits 일치. 재학습 없음")
        for start, expected in [("gpu", ["gpu", "current", "status", "failed", "<eos>"]),
                                ("network", ["network", "current", "status", "healthy", "<eos>"])]:
            tokens, reason = generate(model, [start])
            assert tokens == expected and reason == "eos"
            assert generate(model, [start], 1) == ([start, "current"], "limit")
            print(f"{' '.join(tokens)} | 종료: {reason}")
        print("검사 통과: 자기 예측 이어 붙이기·양쪽 eos 종료·최대 생성 수 제한")
