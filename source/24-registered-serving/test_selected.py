"""Registry/startup contract checks with synthetic files, not model inference."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

script = Path(__file__).with_name("01-serve-selected.py")
assert script.is_file(), "selected-registry server is missing"
spec = importlib.util.spec_from_file_location("selected_api", script)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
registry = module.registry

with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    root = Path(temporary)
    model, store, report = root / "model", root / "store", root / "report.json"
    model.mkdir()
    for name in registry.REQUIRED:
        (model / name).write_text("{}", encoding="utf-8")
    (model / "qlora_settings.json").write_text(json.dumps({"base_model": module.api.MODEL,
        "revision": module.api.REVISION}), encoding="utf-8")
    (model / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": module.api.MODEL,
        "revision": module.api.REVISION}), encoding="utf-8")
    report.write_text('{"summary":{}}', encoding="utf-8")
    registry.register(store, "v1", model, report)
    registry.select(store, "v1")
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    calls = []
    # Only intercept the ML boundary. No fake HTTP or generation success.
    module.api.load_model = lambda directory, device: calls.append((directory, device)) or (None, None, 0)
    tokenizer, loaded, count, identity = module.load_selected(store, "cpu")
    assert (tokenizer, loaded, count) == (None, None, 0)
    assert calls == [(model.resolve(), "cpu")]
    assert identity == {"version": "v1", "base_model": module.api.MODEL,
        "revision": module.api.REVISION,
        "manifest_sha256": registry.fingerprint(registry.version_path(store, "v1"))["sha256"]}
    assert all(p.read_bytes() == data for p, data in original.items())
    (model / "tokenizer.json").write_text("changed", encoding="utf-8")
    calls.clear()
    try:
        module.load_selected(store, "cpu")
    except ValueError as exc:
        assert "model file" in str(exc)
    else:
        raise AssertionError("changed model accepted")
    assert not calls, "loader called before integrity check"
    (model / "tokenizer.json").write_bytes(original[model / "tokenizer.json"])
    def change_during_load(directory, device):
        report.write_text('{"summary":{"changed":true}}', encoding="utf-8")
        return None, None, 0
    module.api.load_model = change_during_load
    try:
        module.load_selected(store, "cpu")
    except ValueError as exc:
        assert "report changed" in str(exc)
    else:
        raise AssertionError("startup-time file change accepted")
    report.write_bytes(original[report])
    path = registry.version_path(store, "v1")
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        module.load_selected(store, "cpu")
    except ValueError as exc:
        assert "manifest changed" in str(exc)
    else:
        raise AssertionError("changed selected manifest accepted")
print("PASS: selected path/identity; read-only startup; changed model/receipt rejected before loading; startup report change rejected")
