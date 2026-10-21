"""One new evaluation -> verification -> CLI handoff. Never deploys."""
import importlib.util
import json
from pathlib import Path
import sys
import uuid

here = Path(__file__).resolve().parent
path = here / "smollm2_track_run.py"
if not path.is_file():
    path = here.parent / "26-run-tracking/01-track-run.py"
spec = importlib.util.spec_from_file_location("smollm2_track_run", path)
assert spec is not None and spec.loader is not None
tracker = importlib.util.module_from_spec(spec)
# Ignore stale root AND dependency bytecode without changing existing caches.
cache_settings = sys.pycache_prefix, sys.dont_write_bytecode
try:
    sys.pycache_prefix = str(here / (".uncached-" + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    spec.loader.exec_module(tracker)
finally:
    sys.pycache_prefix, sys.dont_write_bytecode = cache_settings
registry = tracker.registry
LOADED_PIPELINE = registry.fingerprint(Path(__file__))


def run(runs, store, device):
    runs, store = Path(runs), Path(store)
    source = dict(path=str(Path(__file__).resolve()), **registry.fingerprint(Path(__file__)))
    if {key: source[key] for key in ("sha256", "bytes")} != LOADED_PIPELINE:
        raise ValueError("pipeline source changed after import; restart with stable code")
    # Names only: no historical receipt/report is consulted or validated.
    existing = {entry.name for entry in runs.iterdir()} if runs.exists() else set()
    receipt = tracker.run(runs, store, device)
    tracker.retained_hex(receipt["run_id"], 32)
    directory = runs / receipt["run_id"]
    if (receipt["run_id"] in existing or runs.is_symlink() or directory.is_symlink() or
            not directory.is_dir() or directory.resolve().parent != runs.resolve() or
            (directory / "run.json").is_symlink()):
        raise ValueError("receipt must identify this new real run directory")
    result = {"schema": 1, "run_id": directory.name, "decision": "error", "exit_code": 1,
              "stages": {"Evaluate": receipt["status"], "Verify": "error", "Decision": "skipped"},
              "run_file": registry.fingerprint(directory / "run.json"),
              "pipeline_source": source, "followup": "skipped", "deployed": False}
    try:
        if registry.fingerprint(Path(__file__)) != LOADED_PIPELINE:
            raise ValueError("pipeline source changed during evaluation")
        retained = registry.read_json(directory / "run.json")
        if retained != receipt:
            raise ValueError("returned and retained run receipts differ")
        tracker.validate_retained_receipt(retained, directory.name)
        if retained["status"] == "error":
            result["stages"]["Verify"] = "skipped"
            result["error_type"] = retained["error_type"]
        else:
            quality = directory / "quality.json"
            if registry.fingerprint(quality) != retained["quality_file"]:
                raise ValueError("quality fingerprint mismatch")
            report = registry.read_json(quality)
            tracker.validate_retained_report(report, retained)
            # ponytail: one trusted writer; before/after checks are not an atomic snapshot.
            if (tracker.inputs(store) != retained["inputs"] or
                    registry.fingerprint(Path(__file__)) != LOADED_PIPELINE or
                    registry.fingerprint(directory / "run.json") != result["run_file"] or
                    registry.fingerprint(quality) != retained["quality_file"]):
                raise ValueError("evaluation inputs/evidence/source changed before handoff")
            decision, code = ("ready_for_next_step", 0) if retained["approved"] else ("halted_quality", 2)
            result.update(decision=decision, exit_code=code, quality_file=retained["quality_file"],
                          selection=retained["inputs"]["selection"])
            result["stages"].update(Verify="passed", Decision=decision)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    with (directory / "pipeline.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        result = run(args.runs, args.registry, args.device)
        print(json.dumps(result, ensure_ascii=False))
        return result["exit_code"]
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
