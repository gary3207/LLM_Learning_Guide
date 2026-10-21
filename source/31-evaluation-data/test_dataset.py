"""Stdlib-only dataset checks; never loads or trains a model."""
import ast
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name("01-prepare-data.py")


def check():
    assert SCRIPT.is_file(), "missing dataset preparation script"
    module = runpy.run_path(str(SCRIPT))
    data = module["build_dataset"]()
    assert {name: len(rows) for name, rows in data["splits"].items()} == {
        "train": 8, "dev": 4, "test": 4}
    for rows in data["splits"].values():
        assert sum(row["answer"] == "failed" for row in rows) == len(rows) // 2
        assert sum(row["answer"] == "healthy" for row in rows) == len(rows) // 2
    # Inspect the old constant without importing its model-loading entrypoint.
    previous = SCRIPT.parents[1] / "19-model-evaluation/01-evaluate-models.py"
    tree = ast.parse(previous.read_text(encoding="utf-8"))
    old = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "ROWS" for t in node.targets))
    train = {(r["prompt"], r["answer"]) for r in data["splits"]["train"]}
    legacy = {(r["prompt"], r["answer"]) for r in data["legacy_dev"]}
    assert {(p, a) for group, p, a in old if group == "train"} <= train
    assert {(p, a) for group, p, a in old if group != "train"} == legacy
    assert "validate" in module, "missing split leakage validator"
    module["validate"](data)
    mutations = []
    bad = copy.deepcopy(data)
    bad["splits"]["dev"][0]["prompt"] = "  DEVICE GPU STATUS? Reply with one word.  "
    mutations.append(bad)
    bad = copy.deepcopy(data)
    bad["splits"]["test"][0]["prompt"] = data["legacy_dev"][0]["prompt"]
    mutations.append(bad)
    bad = copy.deepcopy(data)
    bad["splits"]["test"][0]["answer"] = "healthy"
    mutations.append(bad)
    bad = copy.deepcopy(data)
    bad["splits"]["dev"].append(bad["splits"]["train"].pop())
    mutations.append(bad)
    bad = copy.deepcopy(data)
    bad["splits"]["test"] = []
    mutations.append(bad)
    bad = copy.deepcopy(data)
    bad["splits"]["dev"][1]["prompt"] = "Something different for network."
    mutations.append(bad)
    for bad in mutations:
        try:
            module["validate"](bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dataset accepted")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        output = Path(tmp) / "splits.json"
        command = [sys.executable, "-B", str(SCRIPT), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == data
        before = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        assert second.returncode != 0 and output.read_bytes() == before
        root = Path(tmp) / "guide"
        (root / "source/31-evaluation-data").mkdir(parents=True)
        (root / "source/check_guide.py").write_text("", encoding="utf-8")
        (root / "source/31-evaluation-data/broken.py").write_text("def invalid(:", encoding="utf-8")
        checker = runpy.run_path(str(SCRIPT.parents[1] / "check_guide.py"))
        result = checker["check"](root)
        assert any("31-evaluation-data/broken.py" in error for error in result["errors"]), (
            "guide checker silently omitted chapter 31")
    print("PASS dataset counts/balance, legacy separation, JSON CLI roundtrip, overwrite refusal")
    print(f"PASS leakage/label/template guards: {len(mutations)} invalid datasets rejected")
    print("PASS guide checker covers chapter 31")


if __name__ == "__main__":
    check()
