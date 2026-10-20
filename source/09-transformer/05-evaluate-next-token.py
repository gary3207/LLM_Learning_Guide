"""고정 모델의 마지막 위치 다음 토큰을 미리 정한 정답으로 평가합니다."""
from pathlib import Path
import runpy
import torch

# 교육용 규칙: 순서를 바꿔도 gpu의 상태 정답은 failed, network는 healthy.
# 실제 장비 상태·일반 영어 문법의 정답이 아니라 합성 평가 과제입니다.
splits = {
    "train": [("gpu current status", "failed"), ("network current status", "healthy")],
    "validation": [("gpu status current", "failed"), ("network status current", "healthy")],
    "test": [("current status gpu", "failed"), ("current status network", "healthy")],
}
prompts = [prompt for rows in splits.values() for prompt, _ in rows]
assert len(prompts) == len(set(prompts)), "분리 사이 입력이 중복됐습니다"
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original = checkpoint.read_bytes()
model = lm["restore"](checkpoint)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)

# ponytail: split마다 두 예제·마지막 한 위치만 평가; 더 큰 정답 데이터는 후속 범위입니다.
with torch.no_grad():
    for split, rows in splits.items():
        correct, total_loss = 0, 0.0
        assert rows
        for prompt, target in rows:
            tokens = prompt.split()
            assert 1 <= len(tokens) <= 4 and all(t in lm["to_id"] for t in tokens)
            assert target in ("failed", "healthy")
            ids = torch.tensor([[lm["to_id"][t] for t in tokens]], dtype=torch.long)
            logits, _ = model(ids)
            scores = logits[:, -1, :]  # (1, 7): 마지막 위치의 모든 어휘 후보
            assert scores.shape == (1, len(lm["vocab"])) and torch.isfinite(scores).all()
            label = torch.tensor([lm["to_id"][target]], dtype=torch.long)
            loss = loss_fn(scores, label)
            assert torch.isfinite(loss)
            guess = lm["vocab"][scores.argmax(-1).item()]
            correct += int(guess == target)
            total_loss += loss.item()
            print(f"{prompt} -> {guess} | 정답: {target} | loss: {loss.item():.4f}")
        print(f"[{split}] 정답: {correct}/{len(rows)} | 정확도: {correct/len(rows)*100:.1f}% | 평균 loss: {total_loss/len(rows):.4f}")
assert checkpoint.read_bytes() == original, "체크포인트가 변경됐습니다"
print("검사 통과: 입력 중복 없음·유한한 loss·체크포인트 불변. 재학습 없음")
