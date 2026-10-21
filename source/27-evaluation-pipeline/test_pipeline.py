"""Runnable stdlib checks. Synthetic inference is explicitly not model quality."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

script = Path(__file__).with_name("01-pipeline.py")
assert script.is_file(), "chapter27 pipeline is missing"
spec = importlib.util.spec_from_file_location("pipeline", script)
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
tracker, registry = pipeline.tracker, pipeline.registry
gate = tracker.gate
# Reuse only trusted fixture functions, never execute chapter26's tests/integration.
previous = script.parents[1] / "26-run-tracking/test_tracking.py"
tree = ast.parse(previous.read_text(encoding="utf-8"))
helpers: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
           node.name in {"hashes", "fixture", "synthetic", "flat_copy"}]
assert {node.name for node in helpers} == {"hashes", "fixture", "synthetic", "flat_copy"}
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(previous), "exec"))
hashes, fixture, synthetic, flat_copy = (globals()[name] for name in ("hashes", "fixture", "synthetic", "flat_copy"))


def invoke(root, store):
    with contextlib.redirect_stdout(io.StringIO()):
        return pipeline.main(["--runs", str(root / "runs"), "--registry", str(store), "--device", "cpu"])


with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    before = hashes(root)
    with patch.object(gate, "run", side_effect=synthetic) as evaluate:
        assert invoke(root, store) == 2
        evaluate.assert_called_once()
    directory = next((root / "runs").iterdir())
    result = registry.read_json(directory / "pipeline.json")
    receipt = registry.read_json(directory / "run.json")
    assert result["run_id"] == receipt["run_id"] == directory.name
    assert result["decision"] == "halted_quality" and result["exit_code"] == 2
    assert result["stages"] == {"Evaluate": "blocked", "Verify": "passed", "Decision": "halted_quality"}
    assert result["run_file"] == registry.fingerprint(directory / "run.json")
    assert result["quality_file"] == receipt["quality_file"]
    assert result["selection"] == receipt["inputs"]["selection"]
    assert result["pipeline_source"] == dict(path=str(script.resolve()), **registry.fingerprint(script))
    assert result["followup"] == "skipped" and result["deployed"] is False
    assert all(hashes(root)[name] == digest for name, digest in before.items())
print("PASS: SYNTHETIC blocked Evaluate->Verify->Decision; exactly one evaluator, linked receipts, no changed inputs")


def approved(*args, **kwargs):
    report = synthetic(*args, **kwargs)
    for row in report["rows"]:
        row["text"] = row["answer"]
    report["summary"], report["approved"] = gate.decide(report["rows"])
    args[2].write_text(json.dumps(report), encoding="utf-8")
    return report


with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    invalid_history = root / "runs" / ("a" * 32)
    invalid_history.mkdir(parents=True)
    (invalid_history / "run.json").write_text("{", encoding="utf-8")
    with patch.object(gate, "run", side_effect=approved) as evaluate, \
            patch.object(tracker, "list_runs", side_effect=AssertionError("history is not this pipeline")):
        assert invoke(root, store) == 0
        evaluate.assert_called_once()
    directory = next(p for p in (root / "runs").iterdir() if p != invalid_history)
    result = registry.read_json(directory / "pipeline.json")
    assert result["decision"] == "ready_for_next_step"
    assert result["stages"] == {"Evaluate": "approved", "Verify": "passed", "Decision": "ready_for_next_step"}
    assert result["deployed"] is False and result["followup"] == "skipped"
    assert (invalid_history / "run.json").read_text() == "{"
print("PASS: SYNTHETIC approved handoff=0, not deployed; invalid history ignored")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    diagnostic = io.StringIO()
    with patch.object(gate, "run", side_effect=RuntimeError("SYNTHETIC evaluator failure")) as evaluate, \
            patch.object(tracker, "validate_retained_report", side_effect=AssertionError("followup ran")) as verify, \
            contextlib.redirect_stderr(diagnostic):
        assert invoke(root, store) == 1
        evaluate.assert_called_once()
        verify.assert_not_called()
    directory = next((root / "runs").iterdir())
    receipt = registry.read_json(directory / "run.json")
    result = registry.read_json(directory / "pipeline.json")
    assert result["decision"] == "error" and result["exit_code"] == 1
    assert result["stages"] == {"Evaluate": "error", "Verify": "skipped", "Decision": "skipped"}
    assert "quality_file" not in result and not (directory / "quality.json").exists()
    assert receipt["status"] == "error" and result["error_type"] == "RuntimeError"
    assert "SYNTHETIC evaluator failure" in diagnostic.getvalue()
print("PASS: SYNTHETIC evaluator error=1; Verify/Decision skipped, no fabricated quality")

actual_run = tracker.run
for mutation in ("quality bytes", "returned receipt", "invalid receipt", "invalid rows", "selection"):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        _, store, _ = fixture(root)
        def changed_after_evaluation(*args):
            receipt = actual_run(*args)
            directory = args[0] / receipt["run_id"]
            if mutation == "quality bytes":
                with (directory / "quality.json").open("a", encoding="utf-8") as stream:
                    stream.write(" ")
            elif mutation == "returned receipt":
                receipt["duration_seconds"] += 1
            elif mutation == "invalid receipt":
                receipt["schema"] = True
                (directory / "run.json").write_text(json.dumps(receipt), encoding="utf-8")
            elif mutation == "invalid rows":
                report = registry.read_json(directory / "quality.json")
                report["rows"][0]["text"] = "wrong"
                (directory / "quality.json").write_text(json.dumps(report), encoding="utf-8")
                receipt["quality_file"] = registry.fingerprint(directory / "quality.json")
                (directory / "run.json").write_text(json.dumps(receipt), encoding="utf-8")
            else:
                with (store / "selected.json").open("a", encoding="utf-8") as stream:
                    stream.write(" ")
            return receipt
        with patch.object(gate, "run", side_effect=approved) as evaluate, \
                patch.object(tracker, "run", side_effect=changed_after_evaluation), \
                contextlib.redirect_stderr(io.StringIO()):
            assert invoke(root, store) == 1, mutation
            evaluate.assert_called_once()
        directory = next((root / "runs").iterdir())
        result = registry.read_json(directory / "pipeline.json")
        assert result["decision"] == "error" and result["stages"]["Verify"] == "error", mutation
        assert result["stages"]["Decision"] == "skipped" and result["followup"] == "skipped"
        assert result["run_file"] == registry.fingerprint(directory / "run.json")
        assert (directory / "quality.json").exists(), "partial evidence lost"
print("PASS: SYNTHETIC post-evaluation tamper/mismatch/schema/rows/selection fails closed; evidence retained")

import subprocess
for arguments, code in ((["--device", "invalid"], 1), (["--serve"], 1), (["--help"], 0)):
    result = subprocess.run([sys.executable, "-B", str(script), *arguments], capture_output=True)
    assert result.returncode == code, (arguments, result.returncode)

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model, store, performance = fixture(root)
    before = hashes(root)
    with patch.object(gate, "run", side_effect=AssertionError("unsafe ML")) as evaluate, \
            contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        for unsafe in (model / "new", store / "new", performance, performance / "new"):
            assert pipeline.main(["--runs", str(unsafe), "--registry", str(store), "--device", "cpu"]) == 1
        evaluate.assert_not_called()
    assert hashes(root) == before
    with patch.object(gate, "run", side_effect=synthetic):
        historical = actual_run(root / "runs", store, "cpu")
    directory = root / "runs" / historical["run_id"]
    old = hashes(root)
    with patch.object(tracker, "run", return_value=historical), contextlib.redirect_stderr(io.StringIO()):
        assert invoke(root, store) == 1, "historical run reused"
    assert hashes(root) == old and not (directory / "pipeline.json").exists()
    for bad_id in ("../outside", "A" * 32, str(root.resolve())):
        with patch.object(tracker, "run", return_value=dict(historical, run_id=bad_id)), \
                contextlib.redirect_stderr(io.StringIO()):
            assert invoke(root, store) == 1
        assert hashes(root) == old
    def collision(*args):
        receipt = actual_run(*args)
        (args[0] / receipt["run_id"] / "pipeline.json").write_text("retain", encoding="utf-8")
        return receipt
    with patch.object(tracker, "run", side_effect=collision), patch.object(gate, "run", side_effect=synthetic), \
            contextlib.redirect_stderr(io.StringIO()):
        assert invoke(root, store) == 1
    assert [p.read_text() for p in (root / "runs").glob("*/pipeline.json")] == ["retain"]
print("PASS: usage/error=1/help=0; unsafe roots before ML; only fresh safe run; exclusive pipeline write")


def load(path):
    spec = importlib.util.spec_from_file_location("copied_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for timing in ("before", "during", "after", "tracker after"):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        _, store, _ = fixture(root)
        flat = flat_copy(root)
        copied_script = flat / "smollm2_pipeline.py"
        copied_script.write_bytes(script.read_bytes())
        copied = load(copied_script)
        target = Path(copied.tracker.__file__) if timing == "tracker after" else copied_script
        def mutate():
            with target.open("a", encoding="utf-8") as stream:
                stream.write("\n# SYNTHETIC source changed\n")
        if timing == "before":
            mutate()
        def boundary(*args, **kwargs):
            report = synthetic(*args, **kwargs)
            if timing == "during":
                mutate()
            return report
        copied_run = copied.tracker.run
        def returned(*args):
            receipt = copied_run(*args)
            if timing in ("after", "tracker after"):
                mutate()
            return receipt
        with patch.object(copied.tracker.gate, "run", side_effect=boundary) as evaluate, \
                patch.object(copied.tracker, "run", side_effect=returned), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert copied.main(["--runs", str(root / "runs"), "--registry", str(store), "--device", "cpu"]) == 1, timing
            if timing == "before":
                evaluate.assert_not_called()
                assert not (root / "runs").exists()
            else:
                evaluate.assert_called_once()
                directory = next((root / "runs").iterdir())
                result = registry.read_json(directory / "pipeline.json")
                assert result["decision"] == "error" and result["stages"]["Decision"] == "skipped"
print("PASS: SYNTHETIC own source before/during/after and tracker source after evaluation fail closed")

for changed in ("pipeline", "run", "quality"):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        _, store, _ = fixture(root)
        flat = flat_copy(root)
        copied_script = flat / "smollm2_pipeline.py"
        copied_script.write_bytes(script.read_bytes())
        copied = load(copied_script)
        validate = copied.tracker.validate_retained_report
        def late_change(report, receipt):
            validate(report, receipt)
            directory = root / "runs" / receipt["run_id"]
            target = copied_script if changed == "pipeline" else directory / (changed + ".json")
            with target.open("a", encoding="utf-8") as stream:
                stream.write(" ")
        with patch.object(copied.tracker.gate, "run", side_effect=approved), \
                patch.object(copied.tracker, "validate_retained_report", side_effect=late_change), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert copied.main(["--runs", str(root / "runs"), "--registry", str(store), "--device", "cpu"]) == 1, changed
        result = registry.read_json(next((root / "runs").iterdir()) / "pipeline.json")
        assert result["stages"]["Decision"] == "skipped", changed
print("PASS: SYNTHETIC changes during Verify cannot become a successful handoff")

import py_compile
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    flat = flat_copy(root)
    copied_script = flat / "smollm2_pipeline.py"
    copied_script.write_bytes(script.read_bytes())
    root_tracker = flat / "smollm2_track_run.py"
    root_tracker.write_bytes(root_tracker.read_bytes() + b"\nCACHE_SENTINEL = 1\n")
    for target, old, new in ((root_tracker, b"CACHE_SENTINEL = 1", b"CACHE_SENTINEL = 2"),
            (flat / "smollm2_quality_gate.py", b'"max_new_tokens": 8', b'"max_new_tokens": 9')):
        stamp = target.stat()
        py_compile.compile(str(target), doraise=True)
        contents = target.read_bytes()
        changed = contents.replace(old, new, 1)
        assert changed != contents and len(changed) == len(contents)
        target.write_bytes(changed)
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    before = hashes(flat)
    settings = sys.pycache_prefix, sys.dont_write_bytecode
    copied = load(copied_script)
    assert copied.tracker.CACHE_SENTINEL == 2
    assert copied.tracker.gate.GENERATION["max_new_tokens"] == 9
    assert (sys.pycache_prefix, sys.dont_write_bytecode) == settings
    assert hashes(flat) == before, "imports changed inputs/cache"
    assert copied.tracker.sources()["tracker"]["sha256"] == hashlib.sha256(root_tracker.read_bytes()).hexdigest()
print("PASS: flat seven modules; same-size/mtime root+nested stale-pyc ignored, settings/caches unchanged")

if "--integration" not in sys.argv:
    raise SystemExit(0)

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse the trusted chapter26 fixture construction ONLY, not its test/generation.
start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == "api" for t in node.targets))
end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.Expr) and
           isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and
           isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "registry" and
           node.value.func.attr == "select")
construction = tree.body[start:end + 1]
namespace = dict(globals(), script=previous.with_name("01-track-run.py"))
exec(compile(ast.Module(body=construction, type_ignores=[]), str(previous), "exec"), namespace)
root, store, base, model_dir, performance = (namespace[name] for name in ("root", "store", "base", "model_dir", "performance"))
flat = flat_copy(root)
copied_script = flat / "smollm2_pipeline.py"
copied_script.write_bytes(script.read_bytes())
original_quality = root / "original25-quality.json"
original_quality.write_text('{"fixture":"existing report, not a captured run"}', encoding="utf-8")
before = {"base": hashes(base), "model": hashes(model_dir), "registry": hashes(store),
          "flat": hashes(flat), "performance": registry.fingerprint(performance),
          "original_quality": registry.fingerprint(original_quality), "sources": tracker.sources(),
          "pipeline_source": registry.fingerprint(script)}
command = [sys.executable, "-B", str(copied_script), "--runs", str(root / "runs"),
           "--registry", str(store), "--device", "cpu"]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=240)
print(result.stdout.strip())
if result.stderr:
    print(result.stderr, file=sys.stderr)
assert result.returncode in (0, 2), result.stdout + result.stderr
directories = list((root / "runs").iterdir())
assert len(directories) == 1
directory = directories[0]
receipt = registry.read_json(directory / "run.json")
report = registry.read_json(directory / "quality.json")
outcome = registry.read_json(directory / "pipeline.json")
tracker.validate_retained_receipt(receipt, directory.name)
tracker.validate_retained_report(report, receipt)
assert report["training_executed"] is False and report["runtime"]["device"] == "cpu"
assert report["runtime"]["quantized_modules"] == 210 and len(report["rows"]) == 8
assert all(0 < row["generated_tokens"] <= 8 for row in report["rows"])
assert (report["summary"], report["approved"]) == gate.decide(report["rows"])
expected = ("ready_for_next_step", 0) if report["approved"] else ("halted_quality", 2)
assert (outcome["decision"], outcome["exit_code"]) == expected and result.returncode == expected[1]
assert outcome["run_id"] == receipt["run_id"] == directory.name
assert outcome["run_file"] == registry.fingerprint(directory / "run.json")
assert outcome["quality_file"] == receipt["quality_file"] == registry.fingerprint(directory / "quality.json")
assert outcome["selection"] == receipt["inputs"]["selection"] == report["model_version"]
assert outcome["pipeline_source"] == dict(path=str(copied_script.resolve()), **registry.fingerprint(copied_script))
assert outcome["followup"] == "skipped" and outcome["deployed"] is False
assert "READY http" not in result.stdout
assert before == {"base": hashes(base), "model": hashes(model_dir), "registry": hashes(store),
          "flat": hashes(flat), "performance": registry.fingerprint(performance),
          "original_quality": registry.fingerprint(original_quality), "sources": tracker.sources(),
          "pipeline_source": registry.fingerprint(script)}
evidence = {"command": command, "exit_code": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "pipeline_path": str(directory / "pipeline.json"),
            "pipeline": outcome, "receipt": receipt, "quality": report,
            "pipeline_file": registry.fingerprint(directory / "pipeline.json"), "immutable": before,
            "scope": "real CPU NF4 zero-B untrained fixture; no optimizer/server; not learned model or CUDA quality"}
with (root / "pipeline-evidence.json").open("x", encoding="utf-8") as stream:
    json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
print("PASS: real CPU NF4 eight generations; retained Evaluate/Verify/Decision linked; inputs unchanged; CLI", result.returncode)
print("EVIDENCE", root / "pipeline-evidence.json")
