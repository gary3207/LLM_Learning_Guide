"""Real save -> fresh-process restore; temporary artifacts only."""
from pathlib import Path
import os
import runpy
import subprocess
import sys
import tempfile
import torch

script = Path(__file__).with_name("03-checkpoint-generate.py")
assert script.is_file(), "저장·복원·생성 실습이 아직 없습니다"
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
    checkpoint = Path(directory) / "tiny_transformer_v1.pt"
    def run(mode):
        return subprocess.run([sys.executable, str(script), mode, str(checkpoint)],
                              capture_output=True, text=True, encoding="utf-8")
    saved = run("save")
    assert saved.returncode == 0, saved.stdout + saved.stderr
    original = checkpoint.read_bytes()
    restored = run("generate")
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert "gpu current status failed <eos>" in restored.stdout
    assert "network current status healthy <eos>" in restored.stdout
    assert checkpoint.read_bytes() == original, "복원이 체크포인트를 변경했습니다"
    refused = run("save")
    assert refused.returncode != 0 and checkpoint.read_bytes() == original
    module = runpy.run_path(str(script))  # import must not train or save
    old = runpy.run_path(str(script.with_name("02-embedding-next-token.py")))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, value in old["model"].state_dict().items():
        assert torch.equal(payload["state_dict"][name], value), name
    model = module["restore"](checkpoint)
    generate = module["generate"]
    assert generate(model, ["gpu"], 1) == (["gpu", "current"], "limit")
    assert generate(model, ["gpu", "current", "status", "failed"], 1)[1] == "eos"
    for prompt, limit in [([], 4), (["unknown"], 4), (["<eos>"], 4),
                          (["gpu"] * 5, 4), (["gpu"], 0), (["gpu"], True)]:
        try:
            generate(model, prompt, limit)
        except ValueError:
            pass
        else:
            raise AssertionError((prompt, limit))
    payload["vocab"] = list(reversed(payload["vocab"]))
    bad = Path(directory) / "wrong-vocab.pt"
    torch.save(payload, bad)
    try:
        module["restore"](bad)
    except ValueError:
        pass
    else:
        raise AssertionError("어휘 순서 불일치를 허용했습니다")
    print(saved.stdout, end="")
    print(restored.stdout, end="")
print("독립 검사 통과: 새 프로세스 복원·이전 가중치 일치·원본 보호·입력 거부")
