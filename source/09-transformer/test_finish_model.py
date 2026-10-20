"""Exercise save and fresh-process check in the documented VM layout."""
from pathlib import Path
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import torch

source = Path(__file__).resolve().parent
script = source / "07-finish-model.py"
assert script.is_file(), "추가 학습·생성 비교·저장·복원 통합 실습이 없습니다"
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
    root = Path(directory)
    work = root / "source"
    work.mkdir()
    shutil.copyfile(source / "03-checkpoint-generate.py", work / "tiny_transformer_checkpoint.py")
    shutil.copyfile(script, work / "tiny_transformer_finish.py")
    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=work, capture_output=True, text=True, encoding="utf-8")
    original = root / "tiny_transformer_v1.pt"
    candidate = root / "tiny_transformer_augmented_v1.pt"
    prepared = run("tiny_transformer_checkpoint.py", "save", str(original))
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    original_bytes = original.read_bytes()
    saved = run("tiny_transformer_finish.py", "save")
    assert saved.returncode == 0, saved.stdout + saved.stderr
    candidate_bytes = candidate.read_bytes()
    checked = run("tiny_transformer_finish.py", "check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "재학습 없음" in checked.stdout and "원본 파일 불변" in checked.stdout
    assert candidate.read_bytes() == candidate_bytes and original.read_bytes() == original_bytes
    refused = run("tiny_transformer_finish.py", "save")
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
    assert candidate.read_bytes() == candidate_bytes and original.read_bytes() == original_bytes
    lm = runpy.run_path(str(source / "03-checkpoint-generate.py"))
    model = lm["restore"](candidate)
    payload = torch.load(candidate, weights_only=True, map_location="cpu")
    for prompt in ("gpu", "network"):
        tokens, reason = lm["generate"](model, [prompt])
        assert payload["generation"][prompt] == (tokens, reason)
        assert f"[augmented] {' '.join(tokens)} | 종료: {reason}" in saved.stdout
        assert f"[restored] {' '.join(tokens)} | 종료: {reason}" in checked.stdout
    # Exact candidate weights must reproduce the already-verified lesson14 candidate.
    previous = Path.cwd()
    try:
        os.chdir(work)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            earlier = runpy.run_path(str(source / "06-compare-data.py"))
    finally:
        os.chdir(previous)
    assert all(torch.equal(v, earlier["models"]["augmented"].state_dict()[k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(v, earlier["base"].state_dict()[k]) for k, v in model.state_dict().items())
    payload["generation"]["gpu"] = (["gpu", "<eos>"], "eos")
    torch.save(payload, candidate)
    rejected = run("tiny_transformer_finish.py", "check")
    assert rejected.returncode != 0 and "생성 결과 불일치" in rejected.stderr
    assert original.read_bytes() == original_bytes
    print(saved.stdout, end="")
    print(checked.stdout, end="")
print("독립 검사 통과: 이전 후보 가중치 일치·새 프로세스 복원·생성 재현·덮어쓰기 거부·불일치 탐지")
