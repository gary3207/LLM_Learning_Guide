"""Stdlib selection/data/overwrite checks; no model execution."""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name("01-improve-qlora.py")


def check():
    assert SCRIPT.is_file(), "quality comparison script missing"
    module = runpy.run_path(str(SCRIPT))
    data_module = runpy.run_path(str(SCRIPT.parents[1] / "31-evaluation-data/01-prepare-data.py"))
    data = data_module["build_dataset"]()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        path = Path(tmp) / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert module["read_data"](path) == data
        bad = copy.deepcopy(data)
        bad["splits"]["train"].append(bad["splits"]["test"][0])
        path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            module["read_data"](path)
        except ValueError:
            pass
        else:
            raise AssertionError("changed dataset accepted")
    candidates = [{"step": step, "summary": {"dev": {
        "correct": correct, "total": 4, "mean_loss": loss},
        "train": {"correct": 8 - correct, "total": 8, "mean_loss": 0.01}}}
        for step, correct, loss in ((0, 1, 0.2), (20, 2, 0.8), (60, 2, 0.4))]
    assert module["choose"](candidates) == 60
    candidates[0]["summary"]["dev"]["correct"] = 3
    assert module["choose"](candidates) == 0  # Accuracy before loss, not train score.
    candidates[0]["summary"]["dev"]["correct"] = 2
    candidates[0]["summary"]["dev"]["mean_loss"] = 0.4
    assert module["choose"](candidates) == 0  # Exact ties prefer fewer updates.
    for field, value in (("mean_loss", float("nan")), ("correct", True), ("total", 3)):
        bad = copy.deepcopy(candidates)
        bad[0]["summary"]["dev"][field] = value
        try:
            module["choose"](bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dev field accepted: {field}")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        root = Path(tmp)
        output = root / "existing"
        output.mkdir()
        protected = output / "keep.txt"
        protected.write_text("preserve", encoding="utf-8")
        refused = subprocess.run([sys.executable, str(SCRIPT), "train", "--output", str(output)],
                                 capture_output=True, text=True)
        assert refused.returncode != 0 and "FileExistsError" in refused.stderr
        assert protected.read_text(encoding="utf-8") == "preserve"
        # Verify the documented flat deployment without importing historical training scripts.
        flat = root / "flat"
        flat.mkdir()
        (flat / "improve_qlora.py").write_bytes(SCRIPT.read_bytes())
        for name, (_, relative) in {
            "quality_data_helpers.py": (None, "31-evaluation-data/01-prepare-data.py"),
            "quality_eval_helpers.py": (None, "19-model-evaluation/01-evaluate-models.py"),
            "quality_api_helpers.py": (None, "21-api-serving/01-serve-qlora.py"),
        }.items():
            (flat / name).write_bytes((SCRIPT.parents[1] / relative).read_bytes())
        flat_module = runpy.run_path(str(flat / "improve_qlora.py"))
        data_path = flat / "data.json"
        data_path.write_text(json.dumps(data), encoding="utf-8")
        assert flat_module["read_data"](data_path) == data
        guide = root / "guide"
        (guide / "source/32-quality-improvement").mkdir(parents=True)
        (guide / "source/check_guide.py").write_text("", encoding="utf-8")
        (guide / "source/32-quality-improvement/broken.py").write_text("def broken(:", encoding="utf-8")
        checker = runpy.run_path(str(SCRIPT.parents[1] / "check_guide.py"))
        assert any("32-quality-improvement/broken.py" in row for row in checker["check"](guide)["errors"]), (
            "guide checker silently omitted chapter 32")
    print("PASS fixed dataset/contamination refusal and dev-only deterministic selection")
    print("PASS flat helpers, overwrite refusal, invalid dev rejection, chapter-32 checker coverage")


if __name__ == "__main__":
    check()
