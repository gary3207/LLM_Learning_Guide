"""New read-only evaluations with local receipts; no historical-run reconstruction."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

here = Path(__file__).resolve().parent
path = here / "smollm2_quality_gate.py"
if not path.is_file():
    path = here.parent / "25-quality-gate/01-quality-gate.py"
spec = importlib.util.spec_from_file_location("smollm2_quality_gate", path)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
# Ignore existing .pyc without deleting inputs or writing a replacement cache.
cache_settings = sys.pycache_prefix, sys.dont_write_bytecode
try:
    sys.pycache_prefix = str(here / (".uncached-" + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    spec.loader.exec_module(gate)
finally:
    sys.pycache_prefix, sys.dont_write_bytecode = cache_settings
registry = gate.selected.registry


def sources():
    modules = {"gate": gate, "selected": gate.selected, "api": gate.selected.api,
               "registry": registry, "scorer": gate.scorer}
    paths = dict(tracker=Path(__file__), **{name: Path(module.__file__) for name, module in modules.items()})
    return {name: dict(path=str(path.resolve()), **registry.fingerprint(path)) for name, path in paths.items()}


def inputs(store):
    record = registry.show(store)
    return {"sources": sources(), "registry_record": record,
            "selection_file": registry.fingerprint(store / "selected.json"),
            "selection": dict(version=record["version"], base_model=record["base_model"],
                revision=record["revision"], manifest_sha256=registry.fingerprint(
                    registry.version_path(store, record["version"]))["sha256"]),
            "dataset_sha256": hashlib.sha256(json.dumps(gate.scorer.ROWS, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            "generation": dict(gate.GENERATION)}


LOADED_SOURCES = sources()


def validate_report(report, before):
    if (not isinstance(report, dict) or type(report.get("schema")) is not int or report["schema"] != 1 or
            type(report.get("approved")) is not bool or report.get("training_executed") is not False or
            report.get("model_version") != before["selection"] or
            report.get("policy", {}).get("dataset_sha256") != before["dataset_sha256"] or
            not isinstance(report.get("runtime"), dict) or not isinstance(report.get("generation"), dict) or
            any(report["generation"].get(key) != value for key, value in before["generation"].items())):
        raise ValueError("quality report schema/identity/dataset/config mismatch")
    summary, approved = gate.decide(report.get("rows"))
    if report.get("summary") != summary or report["approved"] is not approved:
        raise ValueError("quality report decision mismatch")


def run(runs, store, device):
    runs, store = Path(runs), Path(store)
    before = inputs(store)
    if before["sources"] != LOADED_SOURCES:
        raise ValueError("source files changed after import; restart with stable code")
    target, record = runs.resolve(), before["registry_record"]
    performance = Path(record["report_path"]).resolve()
    if (device not in ("cpu", "cuda") or runs.is_symlink() or
            target.is_relative_to(Path(record["model_directory"]).resolve()) or
            target.is_relative_to(store.resolve()) or target == performance or
            target.is_relative_to(performance)):
        raise ValueError("runs must be outside model/registry/performance report; device must be cpu/cuda")
    directory = runs / uuid.uuid4().hex
    directory.mkdir(parents=True)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    receipt = {"schema": 1, "run_id": directory.name, "started_at_utc": started,
               "inputs": before, "requested_device": device,
               "scope": "new evaluation; trusted single local writer; hashes are not backups/signatures or a bit-exact replay guarantee"}
    try:
        report = gate.run(store, device, directory / "quality.json", serve=False)
        # ponytail: before/after checks assume one trusted writer, not an atomic snapshot.
        if inputs(store) != before:
            raise ValueError("evaluation inputs changed during run")
        retained = registry.read_json(directory / "quality.json")
        if retained != report:
            raise ValueError("returned and retained quality reports differ")
        validate_report(retained, before)
        receipt.update(status="approved" if report["approved"] else "blocked", approved=report["approved"],
                       runtime=report["runtime"], generation=report["generation"],
                       quality_file=registry.fingerprint(directory / "quality.json"))
    except Exception as exc:
        receipt.update(status="error", error_type=type(exc).__name__)
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    receipt.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), duration_seconds=time.monotonic() - clock)
    with (directory / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def retained_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty retained string required")
    return value


def retained_hex(value, length=64):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError("invalid retained hexadecimal identity")


def retained_fingerprint(value):
    retained_hex(value["sha256"])
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("invalid retained byte count")


def retained_utc(value):
    stamp = datetime.fromisoformat(retained_text(value))
    offset = stamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("aware UTC timestamp required")


def retained_generation(value):
    import math
    for key in ("max_new_tokens", "num_beams"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("positive integer generation limit required")
    for key in ("do_sample", "use_cache"):
        if type(value[key]) is not bool:
            raise ValueError("boolean generation option required")
    penalty = value["repetition_penalty"]
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("positive finite repetition penalty required")


def validate_retained_receipt(receipt, run_id):
    import math
    import re
    if (type(receipt["schema"]) is not int or receipt["schema"] != 1 or
            receipt["run_id"] != run_id or receipt["status"] not in ("approved", "blocked", "error")):
        raise ValueError("run receipt schema/identity/status mismatch")
    retained_hex(run_id, 32)
    retained_utc(receipt["started_at_utc"])
    retained_utc(receipt["ended_at_utc"])
    duration = receipt["duration_seconds"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid retained duration")
    if receipt["requested_device"] not in ("cpu", "cuda"):
        raise ValueError("invalid requested device")
    before = receipt["inputs"]
    retained_hex(before["dataset_sha256"])
    retained_fingerprint(before["selection_file"])
    selection = before["selection"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", retained_text(selection["version"])):
        raise ValueError("invalid retained version")
    retained_text(selection["base_model"])
    retained_hex(selection["revision"], 40)
    retained_hex(selection["manifest_sha256"])
    for name in ("tracker", "gate", "selected", "api", "registry", "scorer"):
        meta = before["sources"][name]
        retained_text(meta["path"])
        retained_fingerprint(meta)
    model = before["registry_record"]
    if (type(model["schema"]) is not int or model["schema"] != 1 or
            any(model[key] != selection[key] for key in ("version", "base_model", "revision"))):
        raise ValueError("invalid retained registry identity")
    retained_utc(model["registered_at_utc"])
    for key in ("model_directory", "report_path"):
        retained_text(model[key])
    retained_fingerprint(model["report_file"])
    files = model["model_files"]
    if not isinstance(files, dict) or not {"adapter_model.safetensors", "adapter_config.json",
            "qlora_settings.json", "tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise ValueError("missing retained model files")
    for name, meta in files.items():
        name = retained_text(name)
        if "\\" in name or name.startswith("/") or ":" in name or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("invalid retained model filename")
        retained_fingerprint(meta)
    if not isinstance(model["benchmark_summary"], dict):
        raise ValueError("invalid retained benchmark summary")
    retained_generation(before["generation"])
    if receipt["status"] == "error":
        retained_text(receipt["error_type"])
        if any(key in receipt for key in ("approved", "runtime", "generation", "quality_file")):
            raise ValueError("error receipt contains fabricated completion")
    else:
        if (type(receipt["approved"]) is not bool or
                receipt["status"] != ("approved" if receipt["approved"] else "blocked") or "error_type" in receipt):
            raise ValueError("invalid completion status")
        retained_fingerprint(receipt["quality_file"])
        retained_generation(receipt["generation"])
        if (not isinstance(receipt["runtime"], dict) or
                any(receipt["generation"].get(k) != v for k, v in before["generation"].items())):
            raise ValueError("invalid retained runtime/generation")
        device = retained_text(receipt["runtime"]["device"])
        if device != receipt["requested_device"] and not (receipt["requested_device"] == "cuda" and
                re.fullmatch(r"cuda:[0-9]+", device)):
            raise ValueError("retained runtime device mismatch")


def validate_retained_report(report, receipt):
    before = receipt["inputs"]
    retained_generation(report["generation"])
    if (type(report["schema"]) is not int or report["schema"] != 1 or
            type(report["approved"]) is not bool or report["approved"] is not receipt["approved"] or
            report["model_version"] != before["selection"] or report["training_executed"] is not False or
            report["runtime"] != receipt["runtime"] or report["generation"] != receipt["generation"]):
        raise ValueError("retained report/receipt linkage mismatch")
    policy = report["policy"]
    if (policy["dataset_sha256"] != before["dataset_sha256"] or
            policy.get("scoring", "strip-only, case-sensitive exact match") != "strip-only, case-sensitive exact match" or
            policy.get("criterion", "all 8 correct; pedagogical strict threshold, not a production criterion") !=
            "all 8 correct; pedagogical strict threshold, not a production criterion"):
        raise ValueError("invalid retained scoring policy")
    rows = report["rows"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("eight retained rows required")
    identities, groups = [], {}
    for row in rows:
        identity = tuple(retained_text(row[key]) for key in ("group", "prompt", "answer"))
        if not isinstance(row["text"], str) or identity in identities:
            raise ValueError("invalid or duplicate retained row")
        identities.append(identity)
        group = groups.setdefault(identity[0], {"correct": 0, "total": 0})
        group["total"] += 1
        group["correct"] += row["text"].strip() == identity[2]
    digest = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    summary = report["summary"]
    for counts in (summary, *summary["groups"].values()):
        if any(type(counts[key]) is not int for key in ("correct", "total")):
            raise ValueError("integer summary counts required")
    correct = sum(group["correct"] for group in groups.values())
    if (digest != before["dataset_sha256"] or
            summary != {"correct": correct, "total": 8, "groups": groups} or report["approved"] is not (correct == 8)):
        raise ValueError("retained rows/dataset/decision mismatch")


def list_runs(runs):
    runs = Path(runs)
    if not runs.exists():
        return []
    results = []
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir():
            continue
        item = {"run_id": directory.name, "status": "incomplete"}
        try:
            if directory.is_symlink():
                raise ValueError("run directory is a symbolic link")
            manifest = directory / "run.json"
            if manifest.exists():
                receipt = registry.read_json(manifest)
                validate_retained_receipt(receipt, directory.name)
                item = {key: receipt[key] for key in ("run_id", "status", "started_at_utc", "duration_seconds")}
                if receipt["status"] == "error":
                    item["error_type"] = receipt["error_type"]
                else:
                    quality = directory / "quality.json"
                    if registry.fingerprint(quality) != receipt["quality_file"]:
                        raise ValueError("quality fingerprint mismatch")
                    report = registry.read_json(quality)
                    # Historical query uses retained metadata, never current source/dataset/registry.
                    validate_retained_report(report, receipt)
                    item.update(approved=receipt["approved"], summary=report["summary"],
                                model_version=report["model_version"], quality_file=receipt["quality_file"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            item = {"run_id": directory.name, "status": "invalid", "error_type": type(exc).__name__}
        results.append(item)
    return results


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("run", help="execute a NEW evaluation, never import an old report")
    evaluation.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    evaluation.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    listing = commands.add_parser("list", help="query retained receipts and verify report fingerprints")
    listing.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        if args.command == "list":
            result = list_runs(args.runs)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 1 if any(row["status"] in ("invalid", "incomplete") for row in result) else 0
        receipt = run(args.runs, args.registry, args.device)
        print(json.dumps({"run_id": receipt["run_id"], "status": receipt["status"],
                          "run_directory": str((args.runs / receipt["run_id"]).resolve())}, ensure_ascii=False))
        return {"approved": 0, "blocked": 2, "error": 1}[receipt["status"]]
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
