"""Stdlib-only CLI preservation, flat helper and guide-coverage checks."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name("01-resume-qlora.py")


def check():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        root = Path(tmp)
        output = root / "existing"
        output.mkdir()
        (output / "resumed").mkdir()
        (output / "verify.json").write_text("preserve", encoding="utf-8")
        for mode in ("prepare", "resume", "verify"):
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), mode,
                "--output", str(output)], capture_output=True, text=True)
            assert result.returncode != 0 and "FileExistsError" in result.stderr
        assert (output / "verify.json").read_text(encoding="utf-8") == "preserve"
        flat = root / "flat"
        flat.mkdir()
        copies = {"resume_qlora.py": SCRIPT,
            "improve_qlora.py": SCRIPT.parents[1] / "32-quality-improvement/01-improve-qlora.py",
            "quality_data_helpers.py": SCRIPT.parents[1] / "31-evaluation-data/01-prepare-data.py",
            "quality_eval_helpers.py": SCRIPT.parents[1] / "19-model-evaluation/01-evaluate-models.py",
            "quality_api_helpers.py": SCRIPT.parents[1] / "21-api-serving/01-serve-qlora.py"}
        for name, source in copies.items():
            (flat / name).write_bytes(source.read_bytes())
        q = runpy.run_path(str(flat / "resume_qlora.py"))["quality"]()
        expected = q["helpers"]()["data"]["build_dataset"]()
        data = flat / "data.json"
        data.write_text(json.dumps(expected), encoding="utf-8")
        assert q["read_data"](data) == expected
        guide = root / "guide"
        (guide / "source/33-training-resume").mkdir(parents=True)
        (guide / "source/check_guide.py").write_text("", encoding="utf-8")
        (guide / "source/33-training-resume/broken.py").write_text("def invalid(:", encoding="utf-8")
        checker = runpy.run_path(str(SCRIPT.parents[1] / "check_guide.py"))
        assert any("33-training-resume/broken.py" in error for error in checker["check"](guide)["errors"]), (
            "guide checker omitted chapter 33")
    print("PASS preservation in all modes, actual flat helper paths, chapter33 guide coverage")


if __name__ == "__main__":
    check()
