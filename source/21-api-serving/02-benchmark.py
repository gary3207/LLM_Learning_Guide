"""Sequential loopback benchmark. No training, concurrency or streaming."""
import argparse
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
from urllib.request import Request, build_opener, ProxyHandler

PROMPTS = ["Device gpu status? Reply with one word.",
           "Device network status? Reply with one word."]


def summarize(rows):
    if not rows:
        raise ValueError("no measurements")
    for row in rows:
        times = [row["e2e_seconds"], row["metrics"]["generation_seconds"]]
        if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("measurement duration must be finite and positive")
        if type(row["generated_tokens"]) is not int or row["generated_tokens"] < 0:
            raise ValueError("generated_tokens must be a nonnegative integer")
    latencies = [r["e2e_seconds"] for r in rows]
    tokens = sum(r["generated_tokens"] for r in rows)
    return {"requests": len(rows), "e2e_median_seconds": statistics.median(latencies),
            "e2e_min_seconds": min(latencies), "e2e_max_seconds": max(latencies),
            "generation_tokens_per_second": tokens / sum(r["metrics"]["generation_seconds"] for r in rows),
            "e2e_tokens_per_second": tokens / sum(latencies)}


def run(port, repeats, output):
    if output.exists():
        raise FileExistsError(f"report already exists: {output}")
    base = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    with opener.open(base + "/health", timeout=120) as response:
        health = json.load(response)
    rows = []
    # ponytail: sequential requests only; use a serving/load-test stack when concurrency is the question.
    for phase, count in [("warmup", 1), ("measured", repeats)]:
        for iteration in range(count):
            for prompt in PROMPTS:
                request = Request(base + "/generate", data=json.dumps(
                    {"prompt": prompt, "max_new_tokens": 8}).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                start = perf_counter()
                with opener.open(request, timeout=120) as response:
                    payload = response.read()
                elapsed = perf_counter() - start
                result = json.loads(payload)
                if "metrics" not in result:
                    raise ValueError("start the server with --metrics")
                row = dict(result, phase=phase, iteration=iteration + 1, prompt=prompt, e2e_seconds=elapsed)
                summarize([row])
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = summarize([r for r in rows if r["phase"] == "measured"])
    report = {"health": health, "max_new_tokens": 8, "repeats_per_prompt": repeats,
              "warmup_requests": len(PROMPTS), "summary": summary, "rows": rows}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("SUMMARY", json.dumps(summary), flush=True)
    print("SAVED", output, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("../serving_benchmark_v1.json"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.repeats <= 100:
        parser.error("port must be 1..65535; repeats must be 1..100")
    run(args.port, args.repeats, args.output)
