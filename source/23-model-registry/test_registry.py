"""Local metadata registry checks with synthetic files, not a trained-model test."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).with_name("01-model-registry.py")
assert script.is_file(), "model registry implementation missing"
spec = importlib.util.spec_from_file_location("registry", script)
assert spec is not None and spec.loader is not None
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model = root / "smollm2_qlora_v1"
    model.mkdir()
    settings = {"base_model": "fixture/base", "revision": "0" * 40}
    for name, data in {"qlora_settings.json": settings,
                       "adapter_config.json": {"base_model_name_or_path": "fixture/base", "revision": "0" * 40},
                       "tokenizer_config.json": {}, "tokenizer.json": {}}.items():
        (model / name).write_text(json.dumps(data), encoding="utf-8")
    (model / "adapter_model.safetensors").write_bytes(b"synthetic bytes; never loaded as a model")
    report = root / "serving_benchmark_v1.json"
    report.write_text(json.dumps({"summary": {"requests": 10, "generation_tokens_per_second": 2.0}}), encoding="utf-8")
    store = root / "registry"
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    report_before = report.read_bytes()

    record = registry.register(store, "qlora-v1", model, report)
    assert record["version"] == "qlora-v1" and record["base_model"] == "fixture/base"
    assert record["benchmark_summary"]["requests"] == 10
    receipt = store / "versions/qlora-v1.json"
    original = receipt.read_bytes()
    assert registry.select(store, "qlora-v1")["version"] == "qlora-v1"
    assert registry.show(store)["version"] == "qlora-v1"
    selected = (store / "selected.json").read_bytes()

    def rejected(action, exception):
        try:
            action()
        except exception:
            return
        raise AssertionError("invalid registry operation succeeded")

    rejected(lambda: registry.register(store, "qlora-v1", model, report), FileExistsError)
    assert receipt.read_bytes() == original
    for version in ["../escape", "", ".", "a/b", "a\\b"]:
        rejected(lambda: registry.select(store, version), ValueError)
    rejected(lambda: registry.select(store, "missing"), FileNotFoundError)
    rejected(lambda: registry.register(model / "nested-registry", "v1", model, report), ValueError)

    weights = model / "adapter_model.safetensors"
    weights.write_bytes(b"changed")
    rejected(lambda: registry.select(store, "qlora-v1"), ValueError)
    assert (store / "selected.json").read_bytes() == selected
    weights.write_bytes(before[weights.name])
    extra = model / "unexpected.txt"
    extra.write_text("new file", encoding="utf-8")
    rejected(lambda: registry.show(store), ValueError)
    extra.unlink()
    report.write_text('{"summary": {"requests": 11}}', encoding="utf-8")
    rejected(lambda: registry.select(store, "qlora-v1"), ValueError)
    report.write_bytes(report_before)
    receipt.write_bytes(original + b"\n")
    rejected(lambda: registry.show(store), ValueError)
    receipt.write_bytes(original)
    assert registry.show(store)["version"] == "qlora-v1"
    assert {p.name: p.read_bytes() for p in model.iterdir()} == before
    assert report.read_bytes() == report_before
    output = subprocess.check_output([sys.executable, str(script), "--registry", str(store), "show"], text=True)
    assert json.loads(output)["version"] == "qlora-v1"
    source = root / "source"
    source.mkdir()
    for args in [["register", "qlora-v1"], ["select", "qlora-v1"], ["show"]]:
        output = subprocess.check_output([sys.executable, str(script), *args], cwd=source, text=True)
    assert json.loads(output)["model_directory"] == str(model.resolve())
    assert (root / "model_registry_v1/selected.json").is_file()
    bad_report = root / "bad-report.json"
    bad_report.write_text('{"summary": null}', encoding="utf-8")
    rejected(lambda: registry.register(store, "bad-report", model, bad_report), ValueError)
    assert not (store / "versions/bad-report.json").exists()
    collision = root / "collision-registry"
    (collision / "versions").mkdir(parents=True)
    collision_report = collision / "selected.json"
    collision_report.write_bytes(report_before)
    legacy = dict(record, report_path=str(collision_report.resolve()), report_file=registry.fingerprint(collision_report))
    (collision / "versions/qlora-v1.json").write_text(json.dumps(legacy), encoding="utf-8")
    try:
        registry.select(collision, "qlora-v1")
    except ValueError:
        pass
    else:
        assert collision_report.read_bytes() == report_before, "select overwrote its read-only input report"
        raise AssertionError("report/output collision accepted")
    assert collision_report.read_bytes() == report_before
    rejected(lambda: registry.register(collision, "new-version", model, collision_report), ValueError)
    assert not (collision / "versions/new-version.json").exists()
    assert collision_report.read_bytes() == report_before
print("PASS: register/select/show; source unchanged; duplicate/path/missing-version/input-change/receipt-change rejected")
