"""Check real next-token metrics and fixed labels in a temporary VM layout."""
from pathlib import Path
import contextlib
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
script = source / "05-evaluate-next-token.py"
assert script.is_file(), "정답이 있는 다음 토큰 평가 실습이 아직 없습니다"
expected = {
    "train": [("gpu current status", "failed"), ("network current status", "healthy")],
    "validation": [("gpu status current", "failed"), ("network status current", "healthy")],
    "test": [("current status gpu", "failed"), ("current status network", "healthy")],
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
    original = checkpoint.read_bytes()
    previous = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(work)
        with contextlib.redirect_stdout(output):
            ns = runpy.run_path(str(script))
    finally:
        os.chdir(previous)
    assert ns["splits"] == expected
    model, to_id, vocab = ns["model"], ns["lm"]["to_id"], ns["lm"]["vocab"]
    for split, rows in expected.items():
        correct, losses = 0, []
        for prompt, target in rows:
            with torch.no_grad():
                scores = model(torch.tensor([[to_id[t] for t in prompt.split()]]))[0][0, -1]
            values = scores.tolist()
            maximum = max(values)
            loss = maximum + math.log(math.fsum(math.exp(v - maximum) for v in values)) - values[to_id[target]]
            guess = vocab[max(range(len(values)), key=values.__getitem__)]
            correct += guess == target
            losses.append(loss)
            assert f"{prompt} -> {guess} | 정답: {target} | loss: {loss:.4f}" in output.getvalue()
        summary = f"[{split}] 정답: {correct}/{len(rows)} | 정확도: {correct/len(rows)*100:.1f}% | 평균 loss: {math.fsum(losses)/len(rows):.4f}"
        assert summary in output.getvalue(), summary
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, payload["state_dict"][name]) for name, value in model.state_dict().items())
    assert checkpoint.read_bytes() == original
    assert len(list(root.rglob("*.pt"))) == 1
    print(output.getvalue(), end="")
print("독립 검사 통과: 고정 정답·분리·수동 logsumexp loss·정답률·가중치/파일 불변")
