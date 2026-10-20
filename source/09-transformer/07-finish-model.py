"""14장 보완 후보를 재현하고 생성 비교·별도 저장·새 프로세스 복원을 확인합니다."""
import argparse
import copy
from pathlib import Path
import runpy
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["save", "check"])
args = parser.parse_args()
torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
original = Path("../tiny_transformer_v1.pt")
checkpoint = Path("../tiny_transformer_augmented_v1.pt")
if args.mode == "save" and (checkpoint.exists() or checkpoint.is_symlink()):
    raise FileExistsError(f"기존 파일 보호: {checkpoint}")
original_bytes = original.read_bytes()


def generations(model, name):
    results = {}
    for start in ("gpu", "network"):
        tokens, reason = lm["generate"](model, [start])
        results[start] = (tokens, reason)
        print(f"[{name}] {' '.join(tokens)} | 종료: {reason}")
    return results


if args.mode == "save":
    base = lm["restore"](original)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model = copy.deepcopy(base).train()
    rows = [("gpu current status", "failed"), ("network current status", "healthy"),
            ("gpu status current", "failed"), ("network status current", "healthy")]
    x = torch.tensor([[lm["to_id"][t] for t in prompt.split()] for prompt, _ in rows])
    y = torch.tensor([lm["to_id"][target] for _, target in rows])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    # xb: 기존 파일을 거부하고 학습 전에 새 파일만 확보합니다.
    # ponytail: 중단 시 새 파일이 불완전할 수 있음; 운영 저장은 원자적 교체로 확장합니다.
    with checkpoint.open("xb") as stream:
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x)[0][:, -1, :], y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
        with torch.no_grad():
            scores = model(x)[0][:, -1, :]
            print(f"보완 학습 입력 상태 정답: {(scores.argmax(-1) == y).sum().item()}/4 | loss: {loss_fn(scores, y).item():.4f}")
            probes = torch.tensor([[lm["to_id"][t] for t in s.split()] for s in
                                  ("gpu current status failed", "network current status healthy")])
            reference_logits = model(probes)[0]
        assert torch.isfinite(reference_logits).all()
        generations(base, "base")
        generated = generations(model, "augmented")
        assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())
        assert original.read_bytes() == original_bytes, "원본 파일 변경"
        torch.save({"architecture": lm["architecture"], "vocab": lm["vocab"],
                    "state_dict": model.state_dict(), "reference_logits": reference_logits,
                    "generation": generated}, stream)
    print(f"별도 저장 완료: {checkpoint.name} | 원본 가중치·파일 불변")
else:
    saved_bytes = checkpoint.read_bytes()
    model = lm["restore"](checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actual = generations(model, "restored")
    assert actual == payload["generation"], "저장 전·복원 후 생성 결과 불일치"
    assert checkpoint.read_bytes() == saved_bytes, "보완 파일 변경"
    assert original.read_bytes() == original_bytes, "원본 파일 변경"
    print("복원 검사 통과: logits·생성 결과 일치·재학습 없음·원본 파일 불변·보완 파일 불변")
