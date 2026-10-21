"""Stdlib gate checks; --integration adds actual CPU NF4, never training."""
import importlib.util
from pathlib import Path

script = Path(__file__).with_name("01-quality-gate.py")
assert script.is_file(), "quality gate is missing"
spec = importlib.util.spec_from_file_location("quality_gate", script)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def perfect():
    return [dict(group=g, prompt=p, answer=a, text=" " + a + "\n") for g, p, a in gate.scorer.ROWS]


summary, approved = gate.decide(perfect())
assert approved and summary["correct"] == summary["total"] == 8
assert summary["groups"] == {"train": {"correct": 2, "total": 2},
    "probe": {"correct": 2, "total": 2}, "diagnostic": {"correct": 4, "total": 4}}
print("PASS: fixed eight exact answers approved with original groups")
for text in ("healthy", "failed extra", "Failed", "", "unrelated"):
    rows = perfect(); rows[0]["text"] = text
    assert gate.decide(rows)[1] is False
for rows in ([], perfect()[:-1], perfect() + perfect()[:1], list(reversed(perfect())),
             [perfect()[0]] * 8, [dict(row, answer="healthy") for row in perfect()],
             [dict(row, group="final_test") for row in perfect()],
             [dict(row, text=None) for row in perfect()], [None] * 8, None):
    try:
        gate.decide(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed/changed fixed dataset accepted: " + repr(rows))
print("PASS: wrong/extra/case rejected; malformed/missing/duplicate/reordered/relabeled rows fail closed")

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

registry = gate.selected.registry
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model_dir, store, performance = root / "model", root / "registry", root / "benchmark.json"
    model_dir.mkdir()
    for name in registry.REQUIRED:
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "qlora_settings.json").write_text(json.dumps({"base_model": gate.selected.api.MODEL,
        "revision": gate.selected.api.REVISION}), encoding="utf-8")
    (model_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": gate.selected.api.MODEL,
        "revision": gate.selected.api.REVISION}), encoding="utf-8")
    performance.write_text('{"summary":{"synthetic":true}}', encoding="utf-8")
    registry.register(store, "v1", model_dir, performance)
    registry.select(store, "v1")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    tokenizer, model = SimpleNamespace(bos_token_id=1, eos_token_id=2), SimpleNamespace(device="cpu")
    # Explicit ML boundary: synthetic verdict fixtures test orchestration, NOT inference quality.
    with patch.object(gate.selected.api, "load_model", return_value=(tokenizer, model, 210)) as load, \
         patch.object(gate, "evaluate", create=True) as evaluate, \
         patch.object(gate.selected.api, "HTTPServer") as server, \
         patch.object(gate, "version", return_value="unit-boundary", create=True):
        bad = perfect(); bad[0]["text"] = "failed extra"
        evaluate.return_value = bad
        output = root / "blocked.json"
        report = gate.run(store, "cpu", output, serve=True, port=0)
        assert report["approved"] is False and report["training_executed"] is False
        assert report == json.loads(output.read_text(encoding="utf-8"))
        server.assert_not_called()
        evaluate.assert_called_once_with(tokenizer, model)
        load.assert_called_once()
        saved = output.read_bytes()
        load.reset_mock()
        for collision in (output, performance, model_dir / "new.json", store / "new.json"):
            try:
                gate.run(store, "cpu", collision, serve=True)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError("unsafe report output accepted")
        load.assert_not_called()
        assert output.read_bytes() == saved
        evaluate.return_value = perfect()
        server.return_value.__enter__.return_value.server_port = 0
        approved = gate.run(store, "cpu", root / "approved.json", serve=True, port=0)
        assert approved["approved"] is True
        load.assert_called_once()
        live = server.return_value.__enter__.return_value
        assert live.tokenizer is tokenizer and live.model is model
        assert live.identity == approved["model_version"] and live.metrics is False
        server.assert_called_once_with(("127.0.0.1", 0), gate.selected.Handler)
        live.serve_forever.assert_called_once()
        assert all(p.read_bytes() == data for p, data in before.items())
        server.reset_mock()
        for changed in (model_dir / "tokenizer.json", performance,
                        registry.version_path(store, "v1"), store / "selected.json"):
            original = changed.read_bytes()
            def mutate(*args):
                changed.write_bytes(original + b" ")
                return perfect()
            # selected.json whitespace is semantically unchanged: change its version instead.
            if changed.name == "selected.json":
                def mutate(*args):
                    changed.write_text('{"version":"missing","manifest_sha256":"bad"}', encoding="utf-8")
                    return perfect()
            evaluate.side_effect = mutate
            target = root / "changed.json"
            try:
                gate.run(store, "cpu", target, serve=True)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("changed startup input accepted: " + str(changed))
            finally:
                changed.write_bytes(original)
            assert not target.exists()
        server.assert_not_called()
print("PASS: BLOCKED never binds; exclusive report readback; collisions before ML; PASS serves same objects/identity")
print("PASS: model/report/manifest/selection changes during evaluation reject before verdict write/bind")

import sys
import subprocess
for arguments in (["--device", "invalid"], ["--port", "-1"], ["--port", "bad"], ["--unknown"]):
    invalid = subprocess.run([sys.executable, "-B", str(script), *arguments], capture_output=True)
    assert invalid.returncode not in (0, 2), "CLI error must not masquerade as normal BLOCKED exit 2"
assert subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True).returncode == 0
print("PASS: CLI errors exit 1, distinct from quality BLOCKED exit 2; --help works without ML")
if "--integration" not in sys.argv:
    raise SystemExit(0)
assert callable(getattr(gate, "evaluate", None)), "actual generation evaluator is missing"

import ast
import hashlib
import socket
import subprocess
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


api = gate.selected.api
# Reuse chapter 21's fixture construction. AST constants only; never import chapter 20 training.
old = script.parents[1] / "20-qlora/01-qlora.py"
tree = ast.parse(old.read_text(encoding="utf-8"))
assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id in {"model_id", "revision", "settings"} for t in node.targets)]
namespace = {}
exec(compile(ast.Module(body=assignments, type_ignores=[]), str(old), "exec"), namespace)
settings = namespace["settings"]
base = Path(snapshot_download(api.MODEL, revision=api.REVISION, local_files_only=True))
before_base, before_source = hashes(base), hashes(script.parent)
torch.set_num_threads(1)
torch.manual_seed(0)
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    fixture, store, performance = root / "zero_adapter", root / "registry", root / "synthetic_benchmark.json"
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
    model.save_pretrained(fixture, safe_serialization=True)
    tokenizer.save_pretrained(fixture)
    (fixture / "qlora_settings.json").write_text(json.dumps(settings), encoding="utf-8")
    del model
    performance.write_text('{"summary":{"synthetic_untrained_fixture_only":true}}', encoding="utf-8")
    registry.register(store, "zero-fixture-v1", fixture, performance)
    registry.select(store, "zero-fixture-v1")
    before_fixture, before_store, before_performance = hashes(fixture), hashes(store), performance.read_bytes()
    # Exercise the documented VM flat-file dependency path, including legacy chapter 19.
    flat = root / "source"; flat.mkdir()
    for source, name in [(script, "smollm2_quality_gate.py"),
            (script.parents[1] / "24-registered-serving/01-serve-selected.py", "smollm2_serve_selected.py"),
            (script.parents[1] / "21-api-serving/01-serve-qlora.py", "smollm2_api_metrics.py"),
            (script.parents[1] / "23-model-registry/01-model-registry.py", "smollm2_registry.py"),
            (script.parents[1] / "19-model-evaluation/01-evaluate-models.py", "smollm2_evaluate.py")]:
        (flat / name).write_bytes(source.read_bytes())
    before_flat = hashes(flat)
    output = root / "quality.json"
    # Hold a port open: a BLOCKED path attempting bind fails instead of returning the required exit 2.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0)); occupied.listen()
        command = [sys.executable, "-B", str(flat / "smollm2_quality_gate.py"), "--registry", str(store),
                   "--device", "cpu", "--output", str(output), "--serve", "--port", str(occupied.getsockname()[1])]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=240)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "READY" not in result.stdout and "BLOCKED" in result.stdout
    print(result.stdout.strip())
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"]["quantized_modules"] == 210
    assert len(report["rows"]) == 8 and all(0 < row["generated_tokens"] <= 8 for row in report["rows"])
    expected_correct = sum(row["text"].strip() == row["answer"] for row in report["rows"])
    assert report["summary"]["correct"] == expected_correct < 8
    assert (report["summary"], report["approved"]) == gate.decide(report["rows"])
    assert report["approved"] is False and report["training_executed"] is False
    saved = output.read_bytes()
    refusal = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
    assert refusal.returncode not in (0, 2) and "overwrite refused" in refusal.stderr
    assert output.read_bytes() == saved
    assert hashes(fixture) == before_fixture and hashes(store) == before_store
    assert performance.read_bytes() == before_performance and hashes(flat) == before_flat
    print("PASS: real CPU NF4 zero-B eight generations; CLI exit 2/BLOCKED; no READY/bind; report decision recomputed; overwrite refused")
assert hashes(base) == before_base and hashes(script.parent) == before_source
print("PASS: base/model/registry/performance/source unchanged; no optimizer steps; CPU plumbing only, not trained/CUDA quality")
