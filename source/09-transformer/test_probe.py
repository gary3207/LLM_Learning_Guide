"""Run the probe in the documented VM layout; temporary artifacts only."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parent
probe = source / "04-probe-inputs.py"
assert probe.is_file(), "새 입력 관찰 실습이 아직 없습니다"
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
    root = Path(directory)
    work = root / "source"
    work.mkdir()
    checkpoint_script = work / "tiny_transformer_checkpoint.py"
    shutil.copyfile(source / "03-checkpoint-generate.py", checkpoint_script)
    shutil.copyfile(probe, work / probe.name)
    def run(*args):
        result = subprocess.run([sys.executable, *args], cwd=work,
                                capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    run(checkpoint_script.name, "save", "../tiny_transformer_v1.pt")
    checkpoint = root / "tiny_transformer_v1.pt"
    original = checkpoint.read_bytes()
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.pt"))
    output = run(probe.name)
    assert "gpu current status -> gpu current status failed <eos> | 종료: eos" in output
    assert "network current status -> network current status healthy <eos> | 종료: eos" in output
    for prompt in ["gpu status", "network status", "status current gpu", "gpu current status healthy"]:
        assert f"입력: {prompt} -> " in output, prompt
    assert output.count("입력: ") == 6
    assert "미등록 입력 거부: storage status" in output
    assert "검사 통과: 학습 접두 문맥 재현·미등록 입력 거부·체크포인트 불변" in output
    assert checkpoint.read_bytes() == original
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*.pt")) == files
    print(output, end="")
print("독립 검사 통과: VM 배치·새 입력 관찰·기존 체크포인트 보호")
