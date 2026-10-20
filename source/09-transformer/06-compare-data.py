"""같은 원본의 복사본 둘을 추가 학습해 데이터 표현 보완을 비교합니다."""
import copy
from pathlib import Path
import runpy
import torch

# 13장의 test도 결과를 이미 보았으므로 이제 모두 개발용입니다.
groups = {
    "original": [("gpu current status", "failed"), ("network current status", "healthy")],
    "added": [("gpu status current", "failed"), ("network status current", "healthy")],
    "dev_order": [("current status gpu", "failed"), ("current status network", "healthy")],
}
prompts = [prompt for rows in groups.values() for prompt, _ in rows]
assert len(prompts) == len(set(prompts))
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original_bytes = checkpoint.read_bytes()
base = lm["restore"](checkpoint)
original_weights = {k: v.clone() for k, v in base.state_dict().items()}
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)


def batch(rows):
    assert rows
    for prompt, target in rows:
        tokens = prompt.split()
        assert len(tokens) == 3 and all(t in lm["to_id"] for t in tokens)
        assert target in ("failed", "healthy")
    x = torch.tensor([[lm["to_id"][t] for t in prompt.split()] for prompt, _ in rows])
    y = torch.tensor([lm["to_id"][target] for _, target in rows])
    return x, y


def continue_training(rows):
    model = copy.deepcopy(base)  # 가중치를 공유하지 않고 매번 같은 출발점에서 시작합니다.
    model.train()
    x, y = batch(rows)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    # ponytail: 각 후보100회 full-batch·마지막 상태 위치만; 설정 탐색은 후속 범위입니다.
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits[:, -1, :], y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(v, original_weights[k]) for k, v in model.state_dict().items())
    return model.eval()


models = {
    "base": base,
    "control": continue_training(groups["original"]),
    "augmented": continue_training(groups["original"] + groups["added"]),
}
print("후보마다 같은 원본에서 SGD 100회·lr 0.1·마지막 상태 위치만 추가 학습")
with torch.no_grad():
    for name, model in models.items():
        for group, rows in groups.items():
            x, y = batch(rows)
            scores = model(x)[0][:, -1, :]
            assert scores.shape == (len(rows), len(lm["vocab"])) and torch.isfinite(scores).all()
            loss = loss_fn(scores, y)
            assert torch.isfinite(loss)
            guesses = scores.argmax(-1)
            correct = (guesses == y).sum().item()
            print(f"[{name}/{group}] 정답: {correct}/{len(rows)} | 평균 loss: {loss.item():.4f}")
            print("  " + " / ".join(f"{prompt} -> {lm['vocab'][guess]}" for (prompt, _), guess in zip(rows, guesses.tolist())))
assert all(torch.equal(v, original_weights[k]) for k, v in base.state_dict().items())
assert checkpoint.read_bytes() == original_bytes
print("검사 통과: 원본 가중치·체크포인트 불변. 복사본만 학습·저장 없음")
