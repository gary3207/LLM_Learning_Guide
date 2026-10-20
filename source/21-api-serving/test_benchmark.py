"""Stdlib check for measured-response aggregation; no model or GPU needed."""
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("02-benchmark.py")
assert path.exists(), "benchmark client is missing"
spec = importlib.util.spec_from_file_location("benchmark", path)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
rows = [{"e2e_seconds": 2.0, "generated_tokens": 4, "metrics": {"generation_seconds": 1.0}},
        {"e2e_seconds": 4.0, "generated_tokens": 2, "metrics": {"generation_seconds": 2.0}}]
summary = bench.summarize(rows)
assert summary == {"requests": 2, "e2e_median_seconds": 3.0, "e2e_min_seconds": 2.0,
                   "e2e_max_seconds": 4.0, "generation_tokens_per_second": 2.0,
                   "e2e_tokens_per_second": 1.0}, summary
assert bench.summarize([{**rows[0], "generated_tokens": 0}])["generation_tokens_per_second"] == 0
for invalid in [[], [{**rows[0], "e2e_seconds": 0}], [{**rows[0], "generated_tokens": True}],
                [{**rows[0], "metrics": {"generation_seconds": float('nan')}}]]:
    try:
        bench.summarize(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid measurement accepted")
print("PASS: weighted token rates, latency summary, zero tokens and invalid measurements")
