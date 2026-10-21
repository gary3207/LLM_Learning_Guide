"""Stdlib HTTP fixtures: synthetic metrics/text, NOT model inference or quality."""
import contextlib
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
CLIENT = HERE / "01-observe-load.py"


def load(path=CLIENT):
    spec = importlib.util.spec_from_file_location("observe_load", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, client):
    registry = client.selected.registry
    model = root / "model"
    model.mkdir()
    for name in registry.REQUIRED:
        (model / name).write_text("synthetic fixture", encoding="utf-8")
    base, revision = client.selected.api.MODEL, client.selected.api.REVISION
    (model / "qlora_settings.json").write_text(json.dumps({"base_model": base, "revision": revision}))
    (model / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "revision": revision}))
    performance = root / "performance.json"
    performance.write_text('{"summary": {"fixture": true}}')
    store = root / "registry"
    registry.register(store, "v1", model, performance)
    registry.select(store, "v1")
    identity = {"version": "v1", "base_model": base, "revision": revision,
                "manifest_sha256": registry.fingerprint(registry.version_path(store, "v1"))["sha256"]}
    return store, identity


@contextlib.contextmanager
def server(identity, alter=None):
    state = {"calls": [], "generated": 0, "active": 0, "max_active": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, payload):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # A timed-out client does not cancel this fixture's work.

        def do_GET(self):
            state["calls"].append(("GET", self.path))
            payload = {"status": "ok", "device": "cpu", "quantized_modules": 2,
                       "model_version": dict(identity)}
            status, payload = alter("health", len(state["calls"]), 200, payload) if alter else (200, payload)
            self.reply(status, payload)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(("POST", data["prompt"]))
            if not data["prompt"]:
                status, payload = (400, {"error": "blank prompt"})
                if alter:
                    status, payload = alter("probe", 0, status, payload)
                return self.reply(status, payload)
            state["generated"] += 1
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.006)
            payload = {"text": "synthetic", "input_tokens": 4, "generated_tokens": 2,
                       "metrics": {"generation_seconds": 0.002, "cuda_memory": None},
                       "model_version": dict(identity)}
            try:
                status, payload = alter("generate", state["generated"], 200, payload) if alter else (200, payload)
                self.reply(status, payload)
            finally:
                state["active"] -= 1

    with HTTPServer(("127.0.0.1", 0), Handler) as http:
        thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield http.server_port, state
        finally:
            http.shutdown()
            thread.join(timeout=3)


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CLIENT.is_file(), "bounded load client is missing")
        self.client = load()
        self.temp = tempfile.TemporaryDirectory(prefix="operations-load-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store, self.identity = fixture(self.root, self.client)

    def run_client(self, port, **kwargs):
        return self.client.run(registry=self.store, port=port,
                               output=self.root / "load.json", **kwargs)

    def test_default_diagnostic_run_saved_with_two_phases(self):
        with server(self.identity) as (port, state):
            report = self.run_client(port)
        self.assertTrue(report["ok"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["deployed"])
        self.assertEqual(report["quality_assessment"], "not_performed")
        self.assertEqual(report["expected_identity"], self.identity)
        self.assertEqual(report["invalid_prompt_probe"]["status"], 400)
        self.assertTrue(report["invalid_prompt_probe"]["ok"])
        self.assertEqual(len(report["warmup"]), 2)
        self.assertEqual([p["name"] for p in report["phases"]], ["sequential", "concurrent"])
        self.assertEqual([p["workers"] for p in report["phases"]], [1, 3])
        for phase in report["phases"]:
            self.assertEqual(len(phase["rows"]), 6)
            self.assertEqual(phase["failure_count"], 0)
            self.assertEqual(phase["http_status_counts"], {"200": 6})
            self.assertEqual([r["prompt"] for r in phase["rows"]], self.client.bench.PROMPTS * 3)
        self.assertEqual(state["generated"], 14)
        self.assertEqual(len(state["calls"]), 17)
        self.assertEqual(state["max_active"], 1)
        self.assertEqual(json.loads((self.root / "load.json").read_text()), report)

    def test_concurrent_queue_has_bounded_clients_and_wall_denominator(self):
        from unittest.mock import patch
        original = self.client.http.client.HTTPConnection
        lock = threading.Lock()
        counts = {"active": 0, "max": 0}

        class TrackedConnection(original):
            tracked = False

            def request(self, *args, **kwargs):
                with lock:
                    self.tracked = True
                    counts["active"] += 1
                    counts["max"] = max(counts["max"], counts["active"])
                return super().request(*args, **kwargs)

            def close(self):
                with lock:
                    if self.tracked:
                        counts["active"] -= 1
                        self.tracked = False
                super().close()

        with server(self.identity) as (port, state), patch.object(self.client.http.client, "HTTPConnection", TrackedConnection):
            report = self.run_client(port, requests=24, workers=4)
        self.assertEqual(counts, {"active": 0, "max": 4})
        self.assertEqual(state["max_active"], 1)
        for phase in report["phases"]:
            rows, summary = phase["rows"], phase["summary"]
            self.assertNotIn("e2e_tokens_per_second", summary)
            self.assertAlmostEqual(summary["tokens_per_sum_client_seconds"], 48 / sum(r["e2e_seconds"] for r in rows))
            self.assertAlmostEqual(summary["tokens_per_phase_wall_second"], 48 / phase["phase_wall_seconds"])
            self.assertAlmostEqual(summary["successful_requests_per_phase_wall_second"], 24 / phase["phase_wall_seconds"])
            self.assertAlmostEqual(summary["generation_tokens_per_second"], 1000)
            self.assertEqual(summary["e2e_p95_seconds"], sorted(r["e2e_seconds"] for r in rows)[22])
            self.assertAlmostEqual(summary["mean_client_minus_generation_seconds"],
                                   sum(r["e2e_seconds"] - 0.002 for r in rows) / 24)
            self.assertIsNone(phase["max_observed_cuda_memory"])
        self.assertIn("not an exact queue", report["limitations"])

    def test_preflight_rejects_bounds_protected_outputs_and_changed_registry_without_http(self):
        invalid = [{"requests": v} for v in (0, 25, True, 1.5)]
        invalid += [{"workers": v} for v in (0, 5, True, 1.5)]
        invalid += [{"port": v} for v in (0, 65536, True, 1.5)]
        invalid += [{"timeout": v} for v in (0, 61, True, float("nan"), float("inf"), "1")]
        invalid += [{"requests": 1, "workers": 2}]
        with server(self.identity) as (port, state):
            for overrides in invalid:
                with self.assertRaises(ValueError, msg=str(overrides)):
                    self.client.run(registry=self.store, output=self.root / "load.json", **({"port": port} | overrides))
            existing = self.root / "existing.json"
            existing.write_text("keep")
            targets = [existing, self.store / "new.json", self.root / "model/new.json",
                       HERE.parent / "new-report.json", self.root / "missing/report.json"]
            link = self.root / "dangling.json"
            try:
                link.symlink_to(self.root / "absent")
            except OSError:
                pass  # Windows without symlink permission still exercises exclusive files.
            else:
                targets.append(link)
            for target in targets:
                with self.subTest(target=str(target)), self.assertRaises((ValueError, OSError)):
                    self.client.run(registry=self.store, port=port, output=target)
            self.assertEqual(existing.read_text(), "keep")
            (self.root / "model/tokenizer.json").write_text("changed")
            with self.assertRaises(ValueError):
                self.run_client(port)
            self.assertEqual(state["calls"], [])
        self.assertFalse((self.root / "load.json").exists())

    def test_http_failures_persist_partial_results_without_fake_metrics(self):
        def fail(kind, index, status, payload):
            if kind == "generate" and index in (3, 5, 6):
                return 500, {"error": "synthetic inference failure"}
            return status, payload
        with server(self.identity, fail) as (port, state):
            report = self.run_client(port, requests=2, workers=2)
        self.assertFalse(report["ok"])
        first, second = report["phases"]
        self.assertEqual(first["failure_count"], 1)
        self.assertEqual(first["summary"]["requests"], 1)
        self.assertEqual(first["http_status_counts"], {"500": 1, "200": 1})
        self.assertEqual(second["failure_count"], 2)
        self.assertIsNone(second["summary"])
        for phase in report["phases"]:
            for row in phase["rows"]:
                if not row["ok"]:
                    self.assertEqual(row["status"], 500)
                    self.assertIn("500", row["error"])
                    self.assertNotIn("metrics", row)
                    self.assertGreater(row["e2e_seconds"], 0)
        self.assertTrue(report["invalid_prompt_probe"]["ok"])
        self.assertEqual(state["generated"], 6)  # No retry; expected 400 is separate.
        self.assertEqual(json.loads((self.root / "load.json").read_text()), report)

    def test_invalid_success_payload_is_retained_as_failure(self):
        mutations = [
            lambda p: p.pop("metrics"),
            lambda p: p["model_version"].update(version="wrong"),
            lambda p: p["model_version"].pop("revision"),
            lambda p: p["model_version"].update(extra="wrong"),
            lambda p: p.update(text=4),
            lambda p: p.update(input_tokens=True),
            lambda p: p.update(input_tokens=-1),
            lambda p: p.update(generated_tokens=True),
            lambda p: p.update(generated_tokens=-1),
            lambda p: p.update(generated_tokens=9),
            lambda p: p["metrics"].update(generation_seconds=0),
            lambda p: p["metrics"].update(generation_seconds=float("inf")),
            lambda p: p["metrics"].pop("cuda_memory"),
            lambda p: p["metrics"].update(cuda_memory={}),
        ]
        for index, mutate in enumerate(mutations):
            def alter(kind, n, status, payload):
                if kind == "generate" and n == 3:
                    mutate(payload)
                return status, payload
            output = self.root / f"invalid-{index}.json"
            with self.subTest(index=index), server(self.identity, alter) as (port, state):
                report = self.client.run(self.store, port, 1, 1, output=output)
                self.assertFalse(report["ok"])
                phase = report["phases"][0]
                self.assertEqual(phase["failure_count"], 1)
                row = phase["rows"][0]
                self.assertEqual(row["status"], 200)
                self.assertFalse(row["ok"])
                self.assertTrue(row["error"])
                self.assertNotIn("metrics", row)
                self.assertIsNone(phase["summary"])
                self.assertTrue(report["phases"][1]["rows"][0]["ok"])
                self.assertEqual(json.loads(output.read_text()), report)

    def test_transport_failures_are_bounded_retained_and_not_retried(self):
        cases = {"json": b"not-json", "oversize": b"x" * 65537,
                 "redirect": {"error": "no redirects"}, "timeout": None,
                 "nonfinite": b'{"bad":NaN}'}
        for name, raw in cases.items():
            def alter(kind, n, status, payload):
                if kind == "generate" and n == 3:
                    if name == "timeout":
                        time.sleep(.12)
                    else:
                        return (302 if name == "redirect" else 200), raw
                return status, payload
            output = self.root / f"transport-{name}.json"
            with self.subTest(case=name), server(self.identity, alter) as (port, state):
                report = self.client.run(self.store, port, 1, 1, timeout=.06, output=output)
                self.assertFalse(report["ok"])
                row = report["phases"][0]["rows"][0]
                self.assertFalse(row["ok"])
                self.assertTrue(row["error"])
                self.assertNotIn("metrics", row)
                self.assertGreater(row["e2e_seconds"], 0)
                self.assertEqual(row["status"], None if name == "timeout" else 302 if name == "redirect" else 200)
                self.assertEqual(state["generated"], 4)
                for phase in report["phases"]:
                    self.assertIn("no_http_status_count", phase, "unknown status count must not be labeled all transport failures")
                    self.assertEqual(phase["no_http_status_count"],
                                     sum(r["status"] is None for r in phase["rows"]))
                    self.assertTrue(all(key.isdigit() for key in phase["http_status_counts"]))
                    self.assertEqual(sum(phase["http_status_counts"].values()) + phase["no_http_status_count"],
                                     len(phase["rows"]))
                self.assertEqual(json.loads(output.read_text()), report)
                self.assertIn("Timeouts do not cancel", report["limitations"])

    def test_health_schema_and_identity_checked_before_and_after(self):
        for when in ("before", "after"):
            mutations = [lambda p: p.update(status="bad"),
                         lambda p: p["model_version"].update(manifest_sha256="wrong"),
                         lambda p: p["model_version"].update(base_model="wrong"),
                         lambda p: p["model_version"].update(revision="wrong"),
                         lambda p: p.update(device="mps"),
                         lambda p: p.update(quantized_modules=True)]
            for index, mutate in enumerate(mutations):
                def alter(kind, n, status, payload):
                    if kind == "health" and ((when == "before" and n == 1) or (when == "after" and n > 1)):
                        mutate(payload)
                    return status, payload
                output = self.root / f"health-{when}-{index}.json"
                with self.subTest(when=when, mutation=index), server(self.identity, alter) as (port, state):
                    if when == "before":
                        with self.assertRaisesRegex(ValueError, "health"):
                            self.client.run(self.store, port, 1, 1, output=output)
                        self.assertEqual(state["generated"], 0)
                        self.assertFalse(output.exists())
                    else:
                        report = self.client.run(self.store, port, 1, 1, output=output)
                        self.assertFalse(report["ok"])
                        self.assertFalse(report["health_after"]["ok"])
                        self.assertTrue(report["health_after"]["error"])
                        self.assertEqual(json.loads(output.read_text()), report)

    def test_probe_mismatch_or_warmup_failure_marks_run_failed_not_measured(self):
        cases = [(200, {"text": "wrong probe"}), (400, {}), (400, b"bad json"), (400, {"error": "blank"})]
        for index, (probe_status, probe_payload) in enumerate(cases):
            def alter(kind, n, status, payload):
                if kind == "probe":
                    return probe_status, probe_payload
                if index == 3 and kind == "generate" and n == 1:
                    return 500, {"error": "synthetic warmup failure"}
                return status, payload
            with self.subTest(case=index), server(self.identity, alter) as (port, state):
                report = self.client.run(self.store, port, 1, 1, output=self.root / f"probe-{index}.json")
                self.assertFalse(report["ok"])
                self.assertTrue(all(p["failure_count"] == 0 for p in report["phases"]))
                probe = report["invalid_prompt_probe"]
                self.assertEqual(probe["name"], "invalid_empty_prompt")
                self.assertEqual(probe["expected_status"], 400)
                self.assertTrue(probe["excluded_from_measurements"])
                self.assertEqual(probe["ok"], index == 3)
                self.assertTrue(report["warmup"][1]["ok"])
                self.assertEqual(state["generated"], 4)

    def test_cuda_allocator_observations_preserved_validated_and_maximized(self):
        mutations = [None, lambda m: m.update(peak_allocated_bytes=-1),
                     lambda m: m.update(peak_reserved_bytes=True),
                     lambda m: m.pop("reserved_before_bytes"),
                     lambda m: m.update(reserved_before_bytes=0),
                     lambda m: m.update(peak_allocated_bytes=0),
                     lambda m: m.update(peak_reserved_bytes=0)]
        for index, mutate in enumerate(mutations):
            def alter(kind, n, status, payload):
                if kind == "health":
                    payload["device"] = "cuda:0"
                if kind == "generate":
                    memory = {"allocated_before_bytes": 10 + n, "reserved_before_bytes": 20 + n,
                              "peak_allocated_bytes": 30 + n, "peak_reserved_bytes": 40 + n}
                    if n == 3 and mutate:
                        mutate(memory)
                    payload["metrics"]["cuda_memory"] = memory
                return status, payload
            with self.subTest(mutation=index), server(self.identity, alter) as (port, state):
                report = self.client.run(self.store, port, 2, 2, output=self.root / f"cuda-{index}.json")
                self.assertEqual(report["ok"], mutate is None)
                first, second = report["phases"]
                self.assertEqual(first["failure_count"], int(mutate is not None))
                self.assertEqual(first["max_observed_cuda_memory"],
                                 {"peak_allocated_bytes": 34, "peak_reserved_bytes": 44})
                self.assertEqual(second["max_observed_cuda_memory"],
                                 {"peak_allocated_bytes": 36, "peak_reserved_bytes": 46})
                self.assertEqual(first["rows"][1]["metrics"]["cuda_memory"]["allocated_before_bytes"], 14)

    def test_flat_cli_reports_summary_exit_status_and_keeps_source_protected(self):
        import shutil
        import subprocess
        import sys
        flat = self.root / "source"
        flat.mkdir()
        modules = {"smollm2_serve_selected.py": self.client.selected.__file__,
                   "smollm2_api_metrics.py": self.client.selected.api.__file__,
                   "smollm2_registry.py": self.client.selected.registry.__file__,
                   "smollm2_benchmark.py": self.client.bench.__file__,
                   "smollm2_observe_load.py": CLIENT}
        for name, original in modules.items():
            shutil.copyfile(original, flat / name)
        copied = load(flat / "smollm2_observe_load.py")
        self.assertEqual(Path(copied.selected.__file__).parent, flat)
        for mode in ("success", "failure", "health", "protected", "invalid"):
            def alter(kind, n, status, payload):
                if mode == "failure" and kind == "generate" and n == 3:
                    payload.pop("metrics")
                if mode == "health" and kind == "health":
                    payload["model_version"]["version"] = "wrong"
                return status, payload
            output = flat / "bad-report.json" if mode == "protected" else self.root / f"cli-{mode}.json"
            with self.subTest(mode=mode), server(self.identity, alter) as (port, state):
                command = [sys.executable, str(flat / "smollm2_observe_load.py"),
                           "--registry", str(self.store), "--port", str(port), "--output", str(output)]
                if mode == "invalid":
                    command += ["--requests", "25"]
                result = subprocess.run(command, cwd=flat, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0 if mode == "success" else 1, result.stderr)
                if mode in ("success", "failure"):
                    self.assertTrue(output.is_file(), result.stdout)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["ok"], mode == "success")
                    summaries = [json.loads(line.removeprefix("SUMMARY ")) for line in result.stdout.splitlines() if line.startswith("SUMMARY ")]
                    self.assertEqual(len(summaries), 2)
                    self.assertEqual([s["name"] for s in summaries], ["sequential", "concurrent"])
                    self.assertIn("SAVED " + str(output), result.stdout)
                    self.assertEqual(report["settings"], {"host": "127.0.0.1", "port": port,
                                     "requests_per_phase": 6, "workers": 3, "timeout_seconds": 30,
                                     "max_new_tokens": 8, "response_limit_bytes": 65536})
                else:
                    self.assertFalse(output.exists())
                    self.assertTrue(result.stderr.strip())
                    if mode != "health":
                        self.assertEqual(state["calls"], [])


if __name__ == "__main__":
    unittest.main()
