"""Stdlib recording checks; --integration adds actual CPU NF4 (no training)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import contextlib
import io
from unittest.mock import patch

script = Path(__file__).with_name("01-track-run.py")
assert script.is_file(), "run tracker is missing"
spec = importlib.util.spec_from_file_location("tracker", script)
assert spec is not None and spec.loader is not None
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)
gate = tracker.gate
registry = gate.selected.registry


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def fixture(root):
    model, store, performance = root / "model", root / "registry", root / "benchmark.json"
    model.mkdir()
    for name in registry.REQUIRED:
        (model / name).write_text("{}", encoding="utf-8")
    (model / "qlora_settings.json").write_text(json.dumps({"base_model": gate.selected.api.MODEL,
        "revision": gate.selected.api.REVISION}), encoding="utf-8")
    (model / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": gate.selected.api.MODEL,
        "revision": gate.selected.api.REVISION}), encoding="utf-8")
    performance.write_text('{"summary":{"synthetic":true}}', encoding="utf-8")
    registry.register(store, "v1", model, performance)
    registry.select(store, "v1")
    return model, store, performance


def synthetic(store, device, output, serve=False):
    """Declared SYNTHETIC gate.run boundary: tests recording, NOT inference."""
    assert serve is False
    record = registry.show(store)
    rows = [dict(group=g, prompt=p, answer=a, text="wrong") for g, p, a in gate.scorer.ROWS]
    summary, approved = gate.decide(rows)
    report = {"schema": 1, "model_version": dict(version=record["version"],
        manifest_sha256=registry.fingerprint(registry.version_path(store, record["version"]))["sha256"],
        base_model=record["base_model"], revision=record["revision"]),
        "policy": {"dataset_sha256": hashlib.sha256(json.dumps(gate.scorer.ROWS,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()},
        "generation": dict(gate.GENERATION, bos_token_id=1, eos_token_id=2, pad_token_id=2),
        "runtime": {"device": device, "versions": "SYNTHETIC recording boundary"},
        "rows": rows, "summary": summary, "approved": approved, "training_executed": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    return report


with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model, store, performance = fixture(root)
    before = hashes(root)
    with patch.object(gate, "run", side_effect=synthetic) as evaluate:
        receipt = tracker.run(root / "runs", store, "cpu")
        evaluate.assert_called_once()
    assert receipt["status"] == "blocked" and receipt["approved"] is False
    assert len(receipt["run_id"]) == 32 and int(receipt["run_id"], 16) >= 0
    directory = root / "runs" / receipt["run_id"]
    assert json.loads((directory / "run.json").read_text()) == receipt
    report = json.loads((directory / "quality.json").read_text())
    assert receipt["quality_file"] == registry.fingerprint(directory / "quality.json")
    assert receipt["inputs"]["selection"] == report["model_version"]
    assert receipt["inputs"]["dataset_sha256"] == report["policy"]["dataset_sha256"]
    assert receipt["runtime"] == report["runtime"] and receipt["generation"] == report["generation"]
    assert receipt["duration_seconds"] >= 0
    assert receipt["started_at_utc"] <= receipt["ended_at_utc"]
    expected = {"tracker": script, "gate": Path(gate.__file__), "selected": Path(gate.selected.__file__),
        "api": Path(gate.selected.api.__file__), "registry": Path(registry.__file__), "scorer": Path(gate.scorer.__file__)}
    assert receipt["inputs"]["sources"] == {name: dict(path=str(path.resolve()), **registry.fingerprint(path))
        for name, path in expected.items()}
    assert receipt["inputs"]["registry_record"] == registry.show(store)
    assert all((root / name).read_bytes() and hashes(root)[name] == digest for name, digest in before.items())
    assert tracker.list_runs(root / "runs")[0]["status"] == "blocked"
print("PASS: SYNTHETIC single-call recording/readback, UTC/duration, exact six sources, dataset/selection/report linkage, inputs unchanged")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model, store, performance = fixture(root)
    before = hashes(root)
    with patch.object(gate, "run", side_effect=AssertionError("unsafe evaluation")) as evaluate:
        for collision in (model / "new", store / "new", performance, performance / "new"):
            try:
                tracker.run(collision, store, "cpu")
            except (ValueError, FileExistsError, NotADirectoryError):
                pass
            else:
                raise AssertionError("unsafe output root accepted")
        evaluate.assert_not_called()
    assert hashes(root) == before
    assert not (model / "new").exists() and not (store / "new").exists()
    # UUID collision must refuse without touching an existing run.
    with patch.object(tracker.uuid, "uuid4") as identifier, patch.object(gate, "run") as evaluate:
        identifier.return_value.hex = "a" * 32
        existing = root / "runs" / ("a" * 32)
        existing.mkdir(parents=True)
        (existing / "partial").write_text("retain")
        try:
            tracker.run(root / "runs", store, "cpu")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing run reused")
        evaluate.assert_not_called()
        assert (existing / "partial").read_text() == "retain"
print("PASS: unsafe roots rejected before folders/ML; exclusive run UUID; no overwrites")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    diagnostic = io.StringIO()
    with patch.object(gate, "run", side_effect=RuntimeError("ML boundary failure")) as evaluate, \
            contextlib.redirect_stderr(diagnostic):
        receipt = tracker.run(root / "runs", store, "cpu")
        evaluate.assert_called_once()
    assert receipt["status"] == "error" and receipt["error_type"] == "RuntimeError"
    assert "RuntimeError: ML boundary failure" in diagnostic.getvalue(), "actionable error diagnostic lost"
    directory = root / "runs" / receipt["run_id"]
    assert json.loads((directory / "run.json").read_text()) == receipt
    assert not (directory / "quality.json").exists()
    assert "approved" not in receipt and "quality_file" not in receipt
print("PASS: actual exception retains error receipt without invented quality/approval")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    mutations = [lambda r: r.update(approved="false"), lambda r: r.update(approved=True),
        lambda r: r["model_version"].update(version="other"),
        lambda r: r["policy"].update(dataset_sha256="0" * 64),
        lambda r: r["generation"].update(do_sample=True), lambda r: r.update(training_executed=True),
        lambda r: r.update(rows=[]), lambda r: r.update(schema=999)]
    for mutate in mutations:
        def bad_report(*args, **kwargs):
            report = synthetic(*args, **kwargs)
            mutate(report)
            args[2].write_text(json.dumps(report), encoding="utf-8")
            return report
        with patch.object(gate, "run", side_effect=bad_report):
            receipt = tracker.run(root / "runs", store, "cpu")
        assert receipt["status"] == "error", "invalid report accepted: " + repr(receipt)
        assert (root / "runs" / receipt["run_id"] / "quality.json").exists(), "partial evidence lost"
    for mutate in (lambda: gate.scorer.ROWS.append(("x", "y", "z")),
                   lambda: gate.GENERATION.update(max_new_tokens=99),
                   lambda: (store / "selected.json").write_text((store / "selected.json").read_text() + " ")):
        old_rows, old_generation = list(gate.scorer.ROWS), dict(gate.GENERATION)
        old_choice = (store / "selected.json").read_bytes()
        def unstable(*args, **kwargs):
            report = synthetic(*args, **kwargs)
            mutate()
            return report
        try:
            with patch.object(gate, "run", side_effect=unstable):
                receipt = tracker.run(root / "runs", store, "cpu")
            assert receipt["status"] == "error", "unstable input accepted"
        finally:
            gate.scorer.ROWS[:] = old_rows
            gate.GENERATION.clear(); gate.GENERATION.update(old_generation)
            (store / "selected.json").write_bytes(old_choice)
    # A returned report differing from the retained bytes is not a completed run.
    def different_return(*args, **kwargs):
        report = synthetic(*args, **kwargs)
        report["runtime"]["device"] = "not-the-retained-report"
        return report
    with patch.object(gate, "run", side_effect=different_return):
        assert tracker.run(root / "runs", store, "cpu")["status"] == "error"
print("PASS: invalid or mismatched reports and changed dataset/config/selection fail closed; partial evidence retained")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    assert tracker.list_runs(root / "not-created") == []
    with patch.object(gate, "run", side_effect=synthetic):
        receipt = tracker.run(root / "runs", store, "cpu")
    directory = root / "runs" / receipt["run_id"]
    quality = directory / "quality.json"
    original = quality.read_bytes()
    quality.write_bytes(original + b" ")
    assert tracker.list_runs(root / "runs")[0]["status"] == "invalid", "tampered report trusted"
    quality.write_bytes(original)
    # Later code changes are not corruption of the retained historical receipt/report.
    with patch.object(tracker, "sources", side_effect=AssertionError("historical list must not rehash live code")):
        assert tracker.list_runs(root / "runs")[0]["status"] == "blocked"
    (directory / "run.json").write_text("{", encoding="utf-8")
    assert tracker.list_runs(root / "runs")[0]["status"] == "invalid", "corrupt receipt is not BLOCKED"
    (directory / "run.json").unlink()
    assert tracker.list_runs(root / "runs")[0]["status"] == "incomplete"
    with patch.object(gate, "run", side_effect=RuntimeError("failure")):
        failed = tracker.run(root / "runs", store, "cpu")
    states = {r["run_id"]: r["status"] for r in tracker.list_runs(root / "runs")}
    assert states == {receipt["run_id"]: "incomplete", failed["run_id"]: "error"}
print("PASS: query verifies retained report; tampering/corrupt JSON invalid, missing receipt incomplete, errors distinct, later code changes irrelevant")


def flat_copy(root):
    flat = root / "source"
    flat.mkdir()
    names = {"tracker": "smollm2_track_run.py", "gate": "smollm2_quality_gate.py",
        "selected": "smollm2_serve_selected.py", "api": "smollm2_api_metrics.py",
        "registry": "smollm2_registry.py", "scorer": "smollm2_evaluate.py"}
    for key, meta in tracker.sources().items():
        (flat / names[key]).write_bytes(Path(meta["path"]).read_bytes())
    return flat


with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    _, store, _ = fixture(root)
    flat = flat_copy(root)
    spec = importlib.util.spec_from_file_location("flat_tracker", flat / "smollm2_track_run.py")
    assert spec is not None and spec.loader is not None
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    changed = flat / "smollm2_quality_gate.py"
    original = changed.read_bytes()
    changed.write_bytes(original + b"\n# changed after import\n")
    with patch.object(copied.gate, "run", side_effect=AssertionError("stale module evaluated")) as evaluate:
        try:
            copied.run(root / "runs", store, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("stale loaded source accepted")
        evaluate.assert_not_called()
    assert not (root / "runs").exists()
    changed.write_bytes(original)
    def changing_source(*args, **kwargs):
        report = synthetic(*args, **kwargs)
        changed.write_bytes(original + b"\n# changed during evaluation\n")
        return report
    with patch.object(copied.gate, "run", side_effect=changing_source):
        receipt = copied.run(root / "runs", store, "cpu")
    assert receipt["status"] == "error"
print("PASS: flat six-file import; changed loaded code rejected before ML or detected after evaluation")

import contextlib
import io
import subprocess
import sys
for status, expected_exit in (("approved", 0), ("blocked", 2), ("error", 1)):
    with patch.object(tracker, "run", return_value={"run_id": "a" * 32, "status": status}) as evaluate, \
            contextlib.redirect_stdout(io.StringIO()):
        assert tracker.main(["run", "--runs", "new-runs", "--registry", "existing-registry", "--device", "cpu"]) == expected_exit
        evaluate.assert_called_once_with(Path("new-runs"), Path("existing-registry"), "cpu")
for arguments in (["run", "--device", "invalid"], ["--unknown"], [], ["run", "--serve"]):
    invalid = subprocess.run([sys.executable, "-B", str(script), *arguments], capture_output=True)
    assert invalid.returncode == 1, "CLI usage errors must not masquerade as BLOCKED"
assert subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True).returncode == 0
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    result = subprocess.run([sys.executable, "-B", str(script), "list", "--runs", str(Path(temporary) / "absent")],
        capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout) == []
print("PASS: CLI approved=0/blocked=2/error=1; invalid arguments=1; help/list need no ML")

# Existing branches also retain valid approvals and expose interrupted writes honestly.
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model_dir, store, performance = fixture(root)
    def approved_report(*args, **kwargs):
        report = synthetic(*args, **kwargs)
        for row in report["rows"]:
            row["text"] = row["answer"]
        report["summary"], report["approved"] = gate.decide(report["rows"])
        args[2].write_text(json.dumps(report), encoding="utf-8")
        return report
    with patch.object(gate, "run", side_effect=approved_report):
        receipt = tracker.run(root / "runs", store, "cpu")
    assert receipt["status"] == "approved" and receipt["approved"] is True
    assert tracker.list_runs(root / "runs")[0]["status"] == "approved"
    for target in (model_dir / "tokenizer.json", performance, registry.version_path(store, "v1")):
        original = target.read_bytes()
        def changed_registry(*args, **kwargs):
            report = synthetic(*args, **kwargs)
            target.write_bytes(original + b" ")
            return report
        try:
            with patch.object(gate, "run", side_effect=changed_registry):
                assert tracker.run(root / "runs", store, "cpu")["status"] == "error"
        finally:
            target.write_bytes(original)
    actual_open = Path.open
    def failed_receipt_write(path, *args, **kwargs):
        if path.name == "run.json":
            raise OSError("deliberate receipt write failure")
        return actual_open(path, *args, **kwargs)
    with patch.object(gate, "run", side_effect=synthetic), patch.object(Path, "open", failed_receipt_write):
        try:
            tracker.run(root / "interrupted", store, "cpu")
        except OSError:
            pass
        else:
            raise AssertionError("failed receipt write claimed completion")
    partial = tracker.list_runs(root / "interrupted")
    assert len(partial) == 1 and partial[0]["status"] == "incomplete"
    assert (root / "interrupted" / partial[0]["run_id"] / "quality.json").exists()
print("PASS: SYNTHETIC approval retained; registered artifacts checked after run; failed receipt write preserves incomplete evidence")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    import py_compile
    import sys
    root = Path(temporary)
    flat = flat_copy(root)
    old = flat / "smollm2_quality_gate.py"
    stamp = old.stat()
    original = old.read_bytes()
    py_compile.compile(str(old), doraise=True)
    source_changed = original.replace(b'"max_new_tokens": 8', b'"max_new_tokens": 9', 1)
    assert source_changed != original and len(source_changed) == len(original)
    old.write_bytes(source_changed)
    os.utime(old, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    before_import = hashes(flat)
    settings = sys.pycache_prefix, sys.dont_write_bytecode
    spec = importlib.util.spec_from_file_location("cache_regression", flat / "smollm2_track_run.py")
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.gate.GENERATION["max_new_tokens"] == 9, "cached code differs from fingerprinted source"
    assert loaded.sources()["gate"]["sha256"] == hashlib.sha256(source_changed).hexdigest()
    assert (sys.pycache_prefix, sys.dont_write_bytecode) == settings
    assert hashes(flat) == before_import, "source-only dependency import wrote files"
print("PASS: same-size/same-mtime stale bytecode ignored; loaded gate equals recorded source; import settings restored")

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    import copy
    root = Path(temporary)
    _, store, _ = fixture(root)
    with patch.object(gate, "run", side_effect=approved_report):
        baseline = tracker.run(root / "runs", store, "cpu")
    directory = root / "runs" / baseline["run_id"]
    quality, manifest = directory / "quality.json", directory / "run.json"
    baseline_report = json.loads(quality.read_text())
    assert baseline["status"] == "approved"
    assert tracker.list_runs(root / "runs")[0]["status"] == "approved"
    cases = {
        "receipt schema bool": lambda r, q: r.update(schema=True),
        "report schema bool": lambda r, q: q.update(schema=True),
        "bad start": lambda r, q: r.update(started_at_utc="not-a-date"),
        "missing end": lambda r, q: r.pop("ended_at_utc"),
        "non-UTC end": lambda r, q: r.update(ended_at_utc="2026-01-01T00:00:00+09:00"),
        "negative duration": lambda r, q: r.update(duration_seconds=-1),
        "nonfinite duration": lambda r, q: r.update(duration_seconds=float("nan")),
        "bool duration": lambda r, q: r.update(duration_seconds=True),
        "device": lambda r, q: r.update(requested_device="auto"),
        "null identity": lambda r, q: (r["inputs"].update(selection=None), q.update(model_version=None)),
        "null dataset": lambda r, q: (r["inputs"].update(dataset_sha256=None), q["policy"].update(dataset_sha256=None)),
        "no sources": lambda r, q: r["inputs"].pop("sources"),
        "missing source": lambda r, q: r["inputs"]["sources"].pop("scorer"),
        "source bool bytes": lambda r, q: r["inputs"]["sources"]["gate"].update(bytes=True),
        "source bad hash": lambda r, q: r["inputs"]["sources"]["gate"].update(sha256="z" * 64),
        "registry schema bool": lambda r, q: r["inputs"]["registry_record"].update(schema=True),
        "registry missing files": lambda r, q: r["inputs"]["registry_record"].update(model_files={}),
        "registry identity": lambda r, q: r["inputs"]["registry_record"].update(revision="0" * 40),
        "no rows": lambda r, q: q.pop("rows"),
        "missing row": lambda r, q: q["rows"].pop(),
        "duplicate row": lambda r, q: q["rows"].__setitem__(1, dict(q["rows"][0])),
        "row text type": lambda r, q: q["rows"][0].update(text=None),
        "row identity type": lambda r, q: q["rows"][0].update(prompt=None),
        "false approval": lambda r, q: q["rows"][0].update(text="wrong"),
        "summary bool count": lambda r, q: q["summary"].update(correct=True),
        "summary wrong total": lambda r, q: q["summary"].update(total=7),
        "group wrong count": lambda r, q: q["summary"]["groups"]["train"].update(total=1),
        "policy scoring": lambda r, q: q["policy"].update(scoring="case-insensitive"),
        "policy criterion": lambda r, q: q["policy"].update(criterion="any correct"),
        "runtime type": lambda r, q: (r.update(runtime=[]), q.update(runtime=[])),
        "generation bool tokens": lambda r, q: (r["generation"].update(max_new_tokens=True), q["generation"].update(max_new_tokens=True)),
        "generation bool sampling": lambda r, q: (r["generation"].update(do_sample="false"), q["generation"].update(do_sample="false")),
        "report bool beams": lambda r, q: q["generation"].update(num_beams=True),
        "huge duration": lambda r, q: r.update(duration_seconds=10 ** 400),
        "infinite duration": lambda r, q: r.update(duration_seconds=float("inf")),
        "model path type": lambda r, q: r["inputs"]["registry_record"].update(model_directory=None),
        "model file traversal": lambda r, q: r["inputs"]["registry_record"]["model_files"].update({"../outside": {"sha256": "0" * 64, "bytes": 1}}),
        "model file fingerprint": lambda r, q: r["inputs"]["registry_record"]["model_files"]["tokenizer.json"].update(bytes=False),
        "dataset row mismatch": lambda r, q: q["rows"][0].update(prompt="a different prompt"),
        "error missing type": lambda r, q: (r.update(status="error"), r.pop("approved"), r.pop("quality_file"), r.pop("runtime"), r.pop("generation")),
        "error fabricated approval": lambda r, q: r.update(status="error", error_type="RuntimeError"),
    }
    accepted = []
    for name, mutate in cases.items():
        receipt, report = copy.deepcopy(baseline), copy.deepcopy(baseline_report)
        mutate(receipt, report)
        quality.write_text(json.dumps(report), encoding="utf-8")
        if "quality_file" in receipt:
            receipt["quality_file"] = registry.fingerprint(quality)
        manifest.write_text(json.dumps(receipt), encoding="utf-8")
        before = hashes(directory)
        if tracker.list_runs(root / "runs")[0]["status"] != "invalid":
            accepted.append(name)
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = tracker.main(["list", "--runs", str(root / "runs")])
        if exit_code != 1:
            accepted.append(name + " CLI exit")
        assert hashes(directory) == before, "historical query modified retained evidence"
    assert not accepted, "malformed history accepted: " + repr(accepted)
    quality.write_text(json.dumps(baseline_report), encoding="utf-8")
    baseline["quality_file"] = registry.fingerprint(quality)
    # Wall-clock drift is not evidence of a negative monotonic duration.
    baseline["ended_at_utc"] = "2000-01-01T00:00:00+00:00"
    manifest.write_text(json.dumps(baseline), encoding="utf-8")
    with patch.object(gate, "decide", side_effect=AssertionError("live scorer used")), \
            patch.object(tracker, "sources", side_effect=AssertionError("live sources used")), \
            patch.object(registry, "show", side_effect=AssertionError("live registry used")), \
            patch.object(gate.scorer, "ROWS", []):
        assert tracker.list_runs(root / "runs")[0]["status"] == "approved"
print("PASS: historical schema/rows/summary mutation matrix; invalid CLI=1; read-only and independent of live code/dataset/registry")

if "--integration" not in sys.argv:
    raise SystemExit(0)

import ast
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

api = gate.selected.api
training_source = script.parents[1] / "20-qlora/01-qlora.py"
tree = ast.parse(training_source.read_text(encoding="utf-8"))
assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
    any(isinstance(t, ast.Name) and t.id in {"model_id", "revision", "settings"} for t in node.targets)]
namespace = {}
exec(compile(ast.Module(body=assignments, type_ignores=[]), str(training_source), "exec"), namespace)
settings = namespace["settings"]
base = Path(snapshot_download(api.MODEL, revision=api.REVISION, local_files_only=True))
before_base, before_source = hashes(base), {key: value for key, value in tracker.sources().items()}
torch.set_num_threads(1)
torch.manual_seed(0)
# Retain this integration's actual reports/fixture in scratch for independent readback.
root = Path(tempfile.mkdtemp(prefix="tracking26-cpu-", dir=os.environ.get("TMPDIR")))
model_dir, store, performance = root / "zero_adapter", root / "registry", root / "synthetic_benchmark.json"
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(api.MODEL, revision=api.REVISION,
    local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
    attn_implementation="eager", device_map={"": "cpu"},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
    **settings["lora"], revision=api.REVISION)).eval()
assert all(torch.count_nonzero(p).item() == 0 for n, p in model.named_parameters() if "lora_B" in n)
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)
(model_dir / "qlora_settings.json").write_text(json.dumps(settings), encoding="utf-8")
del model
performance.write_text('{"summary":{"synthetic_untrained_fixture_only":true}}', encoding="utf-8")
registry.register(store, "zero-fixture-v1", model_dir, performance)
registry.select(store, "zero-fixture-v1")
# A separate existing chapter-25 result is never read/imported/replaced by the tracker.
original_quality = root / "original25-quality.json"
original_quality.write_text('{"fixture":"existing report, not a captured run"}', encoding="utf-8")
flat = flat_copy(root)
before_fixture, before_registry, before_flat = hashes(model_dir), hashes(store), hashes(flat)
before_performance, before_quality = performance.read_bytes(), original_quality.read_bytes()
runs = root / "runs"
command = [sys.executable, "-B", str(flat / "smollm2_track_run.py"), "run", "--runs", str(runs),
           "--registry", str(store), "--device", "cpu"]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=240)
print(result.stdout.strip())
assert result.returncode == 2, result.stdout + result.stderr
assert "READY" not in result.stdout and "BLOCKED" in result.stdout
assert len(list(runs.iterdir())) == 1
run_dir = next(runs.iterdir())
receipt = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
report = json.loads((run_dir / "quality.json").read_text(encoding="utf-8"))
assert receipt["status"] == "blocked" and receipt["approved"] is False
assert report["runtime"]["quantized_modules"] == 210 and report["runtime"]["device"] == "cpu"
assert len(report["rows"]) == 8 and all(0 < row["generated_tokens"] <= 8 for row in report["rows"])
assert (report["summary"], report["approved"]) == gate.decide(report["rows"])
assert report["summary"]["correct"] == 0 and report["training_executed"] is False
assert receipt["quality_file"] == registry.fingerprint(run_dir / "quality.json")
assert receipt["inputs"]["registry_record"] == registry.show(store)
assert receipt["inputs"]["selection"] == report["model_version"]
assert receipt["inputs"]["dataset_sha256"] == report["policy"]["dataset_sha256"]
assert receipt["runtime"] == report["runtime"] and receipt["generation"] == report["generation"]
for meta in receipt["inputs"]["sources"].values():
    assert dict(sha256=meta["sha256"], bytes=meta["bytes"]) == registry.fingerprint(Path(meta["path"]))
assert all(Path(meta["path"]).parent == flat for meta in receipt["inputs"]["sources"].values())
listing_command = [sys.executable, "-B", str(flat / "smollm2_track_run.py"), "list", "--runs", str(runs)]
listing = subprocess.run(listing_command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
assert listing.returncode == 0, listing.stderr
listed = json.loads(listing.stdout)
assert len(listed) == 1 and listed[0]["run_id"] == receipt["run_id"] and listed[0]["status"] == "blocked"
assert listed[0]["quality_file"] == receipt["quality_file"] and listed[0]["summary"] == report["summary"]
print("LIST", listing.stdout.strip())
assert hashes(model_dir) == before_fixture and hashes(store) == before_registry
assert hashes(flat) == before_flat and performance.read_bytes() == before_performance
assert original_quality.read_bytes() == before_quality
assert hashes(base) == before_base and tracker.sources() == before_source
# Read-only query must flag later report tampering; restore the bytes for independent inspection.
saved = (run_dir / "quality.json").read_bytes()
(run_dir / "quality.json").write_bytes(saved + b" ")
tampered = subprocess.run(listing_command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
assert tampered.returncode == 1 and json.loads(tampered.stdout)[0]["status"] == "invalid"
(run_dir / "quality.json").write_bytes(saved)
evidence = {"command": command, "exit_code": result.returncode, "stdout": result.stdout,
    "list_command": listing_command, "list_exit_code": listing.returncode, "list": listed,
    "receipt": receipt, "quality": report,
    "receipt_file": registry.fingerprint(run_dir / "run.json"), "quality_file": registry.fingerprint(run_dir / "quality.json"),
    "immutable": {"base": before_base, "fixture": before_fixture, "registry": before_registry,
        "flat_sources": before_flat, "production_sources": before_source,
        "performance": registry.fingerprint(performance), "original25_quality": registry.fingerprint(original_quality)},
    "tampered_query_exit": tampered.returncode, "tampered_query": json.loads(tampered.stdout)}
with (root / "evidence.json").open("x", encoding="utf-8") as stream:
    json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
print("PASS: real CPU NF4 zero-B eight generations; NEW run/quality receipts, CLI BLOCKED=2, query=0, tamper query=1")
print("PASS: source/base/adapter/registry/performance/original25 report unchanged; no optimizer or server; not trained/CUDA quality")
print("EVIDENCE", root / "evidence.json")
