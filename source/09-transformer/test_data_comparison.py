"""Check controlled continued training and original preservation with real PyTorch."""
from pathlib import Path
import contextlib
import copy
import io
import math
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import torch

source = Path(__file__).resolve().parent
script = source / "06-compare-data.py"
assert script.is_file(), "데이터 표현 보완 비교 실습이 아직 없습니다"
expected = {
    "original": [("gpu current status", "failed"), ("network current status", "healthy")],
    "added": [("gpu status current", "failed"), ("network status current", "healthy")],
    "dev_order": [("current status gpu", "failed"), ("current status network", "healthy")],
}
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
    root = Path(directory)
    work = root / "source"
    work.mkdir()
    shutil.copyfile(source / "03-checkpoint-generate.py", work / "tiny_transformer_checkpoint.py")
    saved = subprocess.run([sys.executable, "tiny_transformer_checkpoint.py", "save", "../tiny_transformer_v1.pt"],
                           cwd=work, capture_output=True, text=True, encoding="utf-8")
    assert saved.returncode == 0, saved.stdout + saved.stderr
    checkpoint = root / "tiny_transformer_v1.pt"
    original_bytes = checkpoint.read_bytes()
    before_files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    previous = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(work)
        with contextlib.redirect_stdout(output):
            ns = runpy.run_path(str(script))
    finally:
        os.chdir(previous)
    assert ns["groups"] == expected
    base, models, lm = ns["base"], ns["models"], ns["lm"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert list(models) == ["base", "control", "augmented"] and models["base"] is base
    assert all(torch.equal(v, payload["state_dict"][k]) for k, v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())
    # Independently repeat the declared last-position SGD procedure, not the lesson helper.
    for name, rows in [("control", expected["original"]), ("augmented", expected["original"] + expected["added"])]:
        reference = copy.deepcopy(base).train()
        x = torch.tensor([[lm["to_id"][t] for t in prompt.split()] for prompt, _ in rows])
        y = torch.tensor([lm["to_id"][target] for _, target in rows])
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(reference(x)[0][:, -1, :], y).backward()
            optimizer.step()
        for p, q, initial in zip(models[name].parameters(), reference.parameters(), base.parameters()):
            assert p.data_ptr() != initial.data_ptr()
            assert torch.equal(p, q) and torch.isfinite(p).all()
        assert any(not torch.equal(p, q) for p, q in zip(models[name].parameters(), base.parameters()))
    for name, model in models.items():
        assert not model.training
        for group, rows in expected.items():
            correct, losses = 0, []
            for prompt, target in rows:
                ids = torch.tensor([[lm["to_id"][t] for t in prompt.split()]])
                with torch.no_grad():
                    values = model(ids)[0][0, -1].tolist()
                maximum = max(values)
                losses.append(maximum + math.log(math.fsum(math.exp(v - maximum) for v in values)) - values[lm["to_id"][target]])
                correct += lm["vocab"][max(range(len(values)), key=values.__getitem__)] == target
            assert f"[{name}/{group}] 정답: {correct}/2 | 평균 loss: {math.fsum(losses)/2:.4f}" in output.getvalue()
    assert checkpoint.read_bytes() == original_bytes
    assert before_files == sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    print(output.getvalue(), end="")
print("독립 검사 통과: 동일 출발점·정해진 데이터/100회 SGD·수동 loss·원본/파일 보존")
