"""Single-user local QLoRA metadata registry. No training, copying weights or deployment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

REQUIRED = {"adapter_model.safetensors", "adapter_config.json", "qlora_settings.json",
            "tokenizer.json", "tokenizer_config.json"}


def fingerprint(path):
    with path.open("rb") as stream:
        return {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(), "bytes": path.stat().st_size}


def model_files(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("model must be a real directory, not a symbolic link")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symbolic links are not supported: " + str(path))
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = fingerprint(path)
    if not REQUIRED <= files.keys():
        raise ValueError("missing QLoRA/tokenizer files: " + str(sorted(REQUIRED - files.keys())))
    return files


def version_path(store, version):
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", version):
        raise ValueError("version must be 1..64 ASCII letters/digits/underscores/hyphens, starting with a letter/digit")
    return store / "versions" / (version + ".json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_locations(store, model, report):
    if model.is_symlink() or report.is_symlink():
        raise ValueError("input symbolic links are not supported")
    model, report = model.resolve(), report.resolve()
    if store.resolve().is_relative_to(model):
        raise ValueError("registry must be outside the model directory")
    if report == (store / "selected.json").resolve():
        raise ValueError("performance report must not be the registry selection output")
    return model, report


def register(store, version, model, report):
    target = version_path(store, version)
    if target.exists():
        raise FileExistsError("version already registered: " + version)
    model, report = input_locations(store, model, report)
    files = model_files(model)
    settings = read_json(model / "qlora_settings.json")
    adapter = read_json(model / "adapter_config.json")
    base, revision = settings["base_model"], settings["revision"]
    if not isinstance(base, str) or not base.strip() or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("base_model and pinned 40-hex revision required")
    if adapter.get("base_model_name_or_path") != base or adapter.get("revision") != revision:
        raise ValueError("adapter and QLoRA base/revision mismatch")
    performance = read_json(report)
    if not isinstance(performance, dict) or not isinstance(performance.get("summary"), dict):
        raise ValueError("performance report must be a JSON object with summary")
    record = {"schema": 1, "version": version, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "base_model": base, "revision": revision, "model_directory": str(model), "model_files": files,
              "report_path": str(report), "report_file": fingerprint(report), "benchmark_summary": performance["summary"],
              "report_linkage": "attached_at_registration; historical model/report pairing not proven"}
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return record


def verify(store, version):
    record = read_json(version_path(store, version))
    if record.get("schema") != 1 or record.get("version") != version:
        raise ValueError("registry schema/version mismatch")
    model, report = input_locations(store, Path(record["model_directory"]), Path(record["report_path"]))
    if model_files(model) != record["model_files"]:
        raise ValueError("model file set/content changed since registration")
    if fingerprint(report) != record["report_file"]:
        raise ValueError("performance report changed since registration")
    return record


def select(store, version):
    verify(store, version)
    choice = {"version": version, "manifest_sha256": fingerprint(version_path(store, version))["sha256"]}
    # ponytail: one local writer; add coordination only when multiple writers are required.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=store, prefix=".selected-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(choice, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, store / "selected.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return choice


def show(store):
    choice = read_json(store / "selected.json")
    path = version_path(store, choice["version"])
    if fingerprint(path)["sha256"] != choice["manifest_sha256"]:
        raise ValueError("selected version manifest changed")
    return verify(store, choice["version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("version")
    registration.add_argument("--model", type=Path, default=Path("../smollm2_qlora_v1"))
    registration.add_argument("--report", type=Path, default=Path("../serving_benchmark_v1.json"))
    selection = commands.add_parser("select")
    selection.add_argument("version")
    commands.add_parser("show")
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.registry, args.version, args.model, args.report)
        print("REGISTERED", result["version"], "model files:", len(result["model_files"]))
    elif args.command == "select":
        result = select(args.registry, args.version)
        print("SELECTED", result["version"])
    else:
        print(json.dumps(show(args.registry), ensure_ascii=False, indent=2, allow_nan=False))
