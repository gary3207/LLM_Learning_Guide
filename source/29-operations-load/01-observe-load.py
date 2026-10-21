"""Bounded loopback diagnostic client for chapter 24's --metrics server.

Not deployment or quality approval. Keep registry/model/report and server selection
unchanged for the entire run (trusted single writer). Timeouts do not cancel
server generation. Concurrent clients queue on the existing serial HTTPServer;
this does not measure parallel GPU inference, TTFT, or GPU utilization.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
path = HERE / "smollm2_serve_selected.py"
if not path.is_file():
    path = HERE.parent / "24-registered-serving/01-serve-selected.py"
spec = importlib.util.spec_from_file_location("smollm2_serve_selected", path)
assert spec is not None and spec.loader is not None
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)
bench = selected.dependency("smollm2_benchmark", "21-api-serving/02-benchmark.py")


def exchange(port, timeout, prompt=None):
    start = perf_counter()
    result: dict = {"status": None}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = None if prompt is None else json.dumps({"prompt": prompt, "max_new_tokens": 8})
        connection.request("GET" if prompt is None else "POST",
                           "/health" if prompt is None else "/generate", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        result["status"] = response.status
        raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("response exceeds 64 KiB limit")
        result["response_body"] = raw.decode("utf-8", errors="replace")
        payload = json.loads(raw.decode("utf-8"))
        json.dumps(payload, allow_nan=False)
        result["payload"] = payload
        del result["response_body"]
    except (OSError, http.client.HTTPException, ValueError, RecursionError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    result["e2e_seconds"] = perf_counter() - start
    return result


def check_health(result, expected):
    result["ok"] = False
    payload = result.get("payload")
    if "error" in result:
        return result
    if result["status"] != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
        result["error"] = "health must return HTTP 200 and status ok"
    elif payload.get("model_version") != expected:
        result["error"] = "health model_version does not match frozen registry identity"
    elif not isinstance(payload.get("device"), str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", payload["device"]):
        result["error"] = "health device must be cpu or cuda[:index]"
    elif type(payload.get("quantized_modules")) is not int or payload["quantized_modules"] <= 0:
        result["error"] = "health quantized_modules must be a positive integer"
    else:
        result["ok"] = True
    return result


def validate_generation(payload, expected, device, elapsed):
    if not isinstance(payload, dict) or payload.get("model_version") != expected:
        raise ValueError("generation model_version does not match frozen registry identity")
    if not isinstance(payload.get("text"), str):
        raise ValueError("generation text must be a string")
    payload["text"].encode("utf-8")
    if type(payload.get("input_tokens")) is not int or not 1 <= payload["input_tokens"] <= 256:
        raise ValueError("input_tokens must be an integer in 1..256")
    if type(payload.get("generated_tokens")) is not int or not 0 <= payload["generated_tokens"] <= 8:
        raise ValueError("generated_tokens must be an integer in 0..8")
    if not isinstance(payload.get("metrics"), dict) or "cuda_memory" not in payload["metrics"]:
        raise ValueError("missing metrics; start chapter 24 server with --metrics")
    memory = payload["metrics"]["cuda_memory"]
    if device == "cpu":
        if memory is not None:
            raise ValueError("CPU cuda_memory must be null")
    else:
        keys = {"allocated_before_bytes", "reserved_before_bytes", "peak_allocated_bytes", "peak_reserved_bytes"}
        if (not isinstance(memory, dict) or set(memory) != keys
                or any(type(v) is not int or v < 0 for v in memory.values())):
            raise ValueError("CUDA memory must contain four nonnegative integer allocator observations")
        if not (memory["allocated_before_bytes"] <= memory["reserved_before_bytes"] <= memory["peak_reserved_bytes"]
                and memory["allocated_before_bytes"] <= memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"]):
            raise ValueError("CUDA allocator memory ordering is inconsistent")
    bench.summarize([dict(payload, e2e_seconds=elapsed)])


def run(registry=Path("../model_registry_v1"), port=8002, requests=6, workers=3,
        timeout=30, output=Path("../serving_load_v1.json")):
    for name, value, upper in (("port", port, 65535), ("requests", requests, 24), ("workers", workers, 4)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in 1..{upper}")
    if workers > requests:
        raise ValueError("workers must not exceed requests")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be finite and in (0, 60] seconds")
    registry, output = Path(registry), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"report already exists or is a symlink: {output}")
    if not output.parent.is_dir():
        raise ValueError("report parent directory must already exist")
    record = selected.registry.show(registry)
    assert selected.__file__ is not None
    source_root = HERE if Path(selected.__file__).resolve().parent == HERE else HERE.parent
    if any(output.resolve().is_relative_to(root.resolve()) for root in
           (registry, Path(record["model_directory"]), source_root)):
        raise ValueError("report must be outside registry, model, and source directories")
    expected = {key: record[key] for key in ("version", "base_model", "revision")}
    expected["manifest_sha256"] = selected.registry.fingerprint(
        selected.registry.version_path(registry, record["version"]))["sha256"]
    health = check_health(exchange(port, timeout), expected)
    if not health["ok"]:
        raise ValueError("health preflight failed: " + health["error"])
    probe = exchange(port, timeout, "")
    probe.update(name="invalid_empty_prompt", expected_status=400, excluded_from_measurements=True)
    probe["ok"] = ("error" not in probe and probe["status"] == 400
                   and isinstance(probe.get("payload"), dict)
                   and isinstance(probe["payload"].get("error"), str) and bool(probe["payload"]["error"]))
    if not probe["ok"] and "error" not in probe:
        probe["error"] = "invalid empty-prompt probe expected HTTP 400 with error string"

    def generate(index):
        prompt = bench.PROMPTS[index % len(bench.PROMPTS)]
        result = exchange(port, timeout, prompt)
        row = dict(result, prompt=prompt, ok=False)
        if "error" in result:
            return row
        if result["status"] != 200:
            return dict(row, error=f"unexpected HTTP {result['status']}")
        try:
            validate_generation(result["payload"], expected, health["payload"]["device"], row["e2e_seconds"])
        except (ValueError, KeyError, TypeError) as exc:
            return dict(row, error=str(exc), payload=result["payload"])
        del row["payload"]
        return dict(row, **{key: result["payload"][key] for key in
                           ("text", "input_tokens", "generated_tokens", "metrics", "model_version")}, ok=True)

    warmup = [generate(i) for i in range(2)]
    phases = []
    for name, count in (("sequential", 1), ("concurrent", workers)):
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=count) as pool:
            rows = list(pool.map(generate, range(requests)))
        wall = perf_counter() - start
        good = [row for row in rows if row["ok"]]
        summary = None
        if good:
            summary = bench.summarize(good)
            summary["tokens_per_sum_client_seconds"] = summary.pop("e2e_tokens_per_second")
            summary.update(tokens_per_phase_wall_second=sum(r["generated_tokens"] for r in good) / wall,
                           successful_requests_per_phase_wall_second=len(good) / wall,
                           e2e_p95_seconds=sorted(r["e2e_seconds"] for r in good)[math.ceil(.95 * len(good)) - 1],
                           mean_client_minus_generation_seconds=sum(
                               r["e2e_seconds"] - r["metrics"]["generation_seconds"] for r in good) / len(good))
        statuses = {}
        maximum_memory = None
        for row in good:
            memory = row["metrics"]["cuda_memory"]
            if memory is not None:
                if maximum_memory is None:
                    maximum_memory = {key: memory[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
                else:
                    for key in maximum_memory:
                        maximum_memory[key] = max(maximum_memory[key], memory[key])
        for row in rows:
            if row["status"] is not None:
                key = str(row["status"])
                statuses[key] = statuses.get(key, 0) + 1
        phases.append({"name": name, "workers": count, "rows": rows,
                       "phase_wall_seconds": wall, "max_observed_cuda_memory": maximum_memory,
                       "no_http_status_count": sum(row["status"] is None for row in rows),
                       "failure_count": len(rows) - len(good), "http_status_counts": statuses,
                       "summary": summary})
    health_after = check_health(exchange(port, timeout), expected)
    report = {"ok": health_after["ok"] and probe["ok"] and all(r["ok"] for r in warmup)
                     and all(p["failure_count"] == 0 for p in phases),
              "diagnostic_only": True, "deployed": False,
              "quality_assessment": "not_performed", "expected_identity": expected,
              "health_before": health, "health_after": health_after,
              "limitations": "Serial HTTPServer; client minus generation is not an exact queue time. "
                             "Timeouts do not cancel server generation. No TTFT or GPU utilization measurement. "
                             "Trusted single writer: no selection/model/report changes during run.",
              "settings": {"host": "127.0.0.1", "port": port, "requests_per_phase": requests,
                           "workers": workers, "timeout_seconds": timeout, "max_new_tokens": 8,
                           "response_limit_bytes": 65536},
              "invalid_prompt_probe": probe, "warmup": warmup, "phases": phases}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"), help="verified selected registry")
    parser.add_argument("--port", type=int, default=8002, help="loopback server port, 1..65535")
    parser.add_argument("--requests", type=int, default=6, help="requests per measured phase, 1..24")
    parser.add_argument("--workers", type=int, default=3, help="concurrent clients, 1..4 and <= requests")
    parser.add_argument("--timeout", type=float, default=30, help="socket timeout seconds, finite (0, 60]")
    parser.add_argument("--output", type=Path, default=Path("../serving_load_v1.json"), help="new exclusive JSON report")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        report = run(**vars(args))
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for phase in report["phases"]:
        print("SUMMARY", json.dumps({key: value for key, value in phase.items() if key != "rows"}, allow_nan=False))
    print("SAVED", args.output)
    if not report["ok"]:
        print(f"ERROR unexpected diagnostic failures; inspect warmup, phases, probe and health in {args.output}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
