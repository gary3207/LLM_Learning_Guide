"""Small request-contract checks; --integration adds real NF4 + live HTTP, without training."""
import importlib.util
from pathlib import Path
import sys

script = Path(__file__).with_name("01-serve-qlora.py")
assert script.is_file(), "QLoRA API server is missing"
spec = importlib.util.spec_from_file_location("serve_qlora", script)
assert spec is not None and spec.loader is not None
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
assert api.parse_request(b'{"prompt":" hello "}') == (" hello ", 8)
assert api.parse_request(b'{"prompt":"hi","max_new_tokens":1}') == ("hi", 1)
assert api.parse_request(b'{"prompt":"hi","max_new_tokens":32}') == ("hi", 32)
for body in [b'[]', b'null', b'{}', b'bad json', b'\xff', b'{"prompt":"\\ud800"}', b'{"prompt":" "}',
             b'{"prompt":3}', b'{"prompt":"hi","max_new_tokens":true}',
             b'{"prompt":"hi","max_new_tokens":0}', b'{"prompt":"hi","max_new_tokens":33}',
             b'{"prompt":"hi","temperature":1}', b'{"prompt":"' + b'a' * 513 + b'"}']:
    try:
        api.parse_request(body)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid body accepted: {body[:80]!r}")
print("PASS: request JSON/type/length/token-limit contract")

measured = "--metrics" in sys.argv
registered = "--registered" in sys.argv
if measured:
    assert 'parser.add_argument("--metrics"' in script.read_text(encoding="utf-8"), "metrics option missing"

if "--integration" not in sys.argv:
    raise SystemExit(0)
assert hasattr(api, "load_model"), "saved QLoRA loader is missing"

import ast
import hashlib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


def hashes(directory):
    result = {}
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                result[str(path.relative_to(directory))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


# Read only constant assignments; never import the historical training program.
old = script.parents[1] / "20-qlora/01-qlora.py"
tree = ast.parse(old.read_text(encoding="utf-8"))
selected: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in {"model_id", "revision", "settings"} for t in node.targets)]
namespace = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(old), "exec"), namespace)
settings = namespace["settings"]
base = Path(snapshot_download(api.MODEL, revision=api.REVISION, allow_patterns=[
    "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]))
before_base = hashes(base)
torch.set_num_threads(1)
torch.manual_seed(0)
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    fixture = Path(temporary) / "zero_qlora_fixture"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        api.MODEL, revision=api.REVISION, trust_remote_code=False, use_safetensors=True,
        local_files_only=True, dtype=torch.float32, attn_implementation="eager", device_map={"": "cpu"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **settings["lora"],
                                           revision=api.REVISION)).eval()
    assert all(torch.count_nonzero(p).item() == 0 for n, p in model.named_parameters() if "lora_B" in n)
    model.save_pretrained(fixture, safe_serialization=True)
    tokenizer.save_pretrained(fixture)
    (fixture / "qlora_settings.json").write_text(json.dumps(settings), encoding="utf-8")
    before_fixture = hashes(fixture)
    server_script = script
    server_args = [str(fixture)]
    if registered:
        server_script = script.parents[1] / "24-registered-serving/01-serve-selected.py"
        selected_spec = importlib.util.spec_from_file_location("selected_server", server_script)
        assert selected_spec is not None and selected_spec.loader is not None
        selected_api = importlib.util.module_from_spec(selected_spec)
        selected_spec.loader.exec_module(selected_api)
        store = Path(temporary) / "registry"
        performance_report = Path(temporary) / "synthetic_report.json"
        performance_report.write_text('{"summary":{"fixture_only":true}}', encoding="utf-8")
        selected_api.registry.register(store, "zero-fixture-v1", fixture, performance_report)
        selected_api.registry.select(store, "zero-fixture-v1")
        before_store, before_report = hashes(store), performance_report.read_bytes()
        server_args = ["--registry", str(store)]
        # Exercise the documented VM's three flat filenames, not only the repository fallback.
        flat = Path(temporary) / "source"
        flat.mkdir()
        for source, name in [(server_script, "smollm2_serve_selected.py"), (script, "smollm2_api_metrics.py"),
                             (script.parents[1] / "23-model-registry/01-model-registry.py", "smollm2_registry.py")]:
            (flat / name).write_bytes(source.read_bytes())
        server_script = flat / "smollm2_serve_selected.py"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (Path(temporary) / "server.stderr").open("w+", encoding="utf-8") as errors:
        proc = subprocess.Popen([sys.executable, "-u", str(server_script), *server_args, "--device", "cpu", "--port", "0"]
                                + (["--metrics"] if measured else []),
                                stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8", env=env)
        try:
            ready = queue.Queue()
            assert proc.stdout is not None
            threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
            line = ready.get(timeout=180)
            match = re.match(r"READY (http://127\.0\.0\.1:\d+) \| device=cpu \| NF4=210", line)
            if not match:
                errors.flush(); errors.seek(0)
                raise AssertionError(line + errors.read())
            url = match[1]

            def call(path, data=None, headers=None):
                req = Request(url + path, data=data, headers=headers or {})
                try:
                    with urlopen(req, timeout=60) as response:
                        return response.status, json.load(response)
                except HTTPError as exc:
                    return exc.code, json.load(exc)

            status, health = call("/health")
            if registered:
                assert " | version=zero-fixture-v1" in line
                identity = health.pop("model_version")
                assert identity == {"version": "zero-fixture-v1", "base_model": api.MODEL,
                    "revision": api.REVISION, "manifest_sha256": selected_api.registry.fingerprint(
                        selected_api.registry.version_path(store, "zero-fixture-v1"))["sha256"]}
                # A live pointer change must not relabel an already-loaded model.
                selected_api.registry.register(store, "zero-fixture-v2", fixture, performance_report)
                selected_api.registry.select(store, "zero-fixture-v2")
                assert call("/health")[1]["model_version"] == identity
            assert status == 200 and health == {"status": "ok", "device": "cpu", "quantized_modules": 210}
            for prompt, limit in [("Device gpu status? Reply with one word.", 8),
                                  ("Device network status? Reply with one word.", 4)]:
                ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    tokenize=True, add_generation_prompt=True, return_tensors="pt")
                with torch.inference_mode():
                    generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                        generation_config=GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1,
                            use_cache=True, repetition_penalty=1.0, bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id))
                expected = {"text": tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True),
                            "input_tokens": ids.shape[1], "generated_tokens": generated.shape[1] - ids.shape[1]}
                status, response = call("/generate", json.dumps({"prompt": prompt, "max_new_tokens": limit}).encode(),
                                        {"Content-Type": "application/json"})
                if registered:
                    assert response.pop("model_version") == identity
                if measured:
                    metrics = response.pop("metrics")
                    assert metrics["generation_seconds"] > 0 and metrics["cuda_memory"] is None, metrics
                assert status == 200 and response == expected, response
                print("HTTP 200 real NF4 untrained fixture:", json.dumps(response, ensure_ascii=False))
            if registered:
                selected_api.registry.select(store, "zero-fixture-v1")
                before_store = hashes(store)
            for body, headers, expected_status in [
                (b'null', {"Content-Type": "application/json"}, 400),
                (b'{bad', {"Content-Type": "application/json"}, 400),
                (b'{"prompt":" "}', {"Content-Type": "application/json"}, 400),
                (b'{"prompt":"hi","max_new_tokens":true}', {"Content-Type": "application/json"}, 400),
                (b'{"prompt":"\\ud800"}', {"Content-Type": "application/json"}, 400),
                (json.dumps({"prompt": "x " * 240}).encode(), {"Content-Type": "application/json"}, 400),
                (b'x' * 4097, {"Content-Type": "application/json"}, 413),
                (b'{}', {"Content-Type": "text/plain"}, 415),
                (b'{}', {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
            ]:
                status, response = call("/generate", body, headers)
                assert status == expected_status and "error" in response, (status, response)
            assert call("/not-found")[0] == 404
            assert call("/not-found", b'{}', {"Content-Type": "application/json"})[0] == 404
            assert call("/health")[0] == 200
            if measured:
                benchmark_path = script.with_name("02-benchmark.py")
                benchmark_spec = importlib.util.spec_from_file_location("benchmark", benchmark_path)
                assert benchmark_spec is not None and benchmark_spec.loader is not None
                benchmark = importlib.util.module_from_spec(benchmark_spec)
                benchmark_spec.loader.exec_module(benchmark)
                output = Path(temporary) / "benchmark.json"
                report = benchmark.run(int(url.rsplit(":", 1)[1]), 2, output)
                assert report == json.loads(output.read_text(encoding="utf-8"))
                assert report["summary"]["requests"] == 4 and len(report["rows"]) == 6
                assert all(r["metrics"]["cuda_memory"] is None for r in report["rows"])
                saved = output.read_bytes()
                try:
                    benchmark.run(int(url.rsplit(":", 1)[1]), 2, output)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("existing report overwritten")
                assert output.read_bytes() == saved
                print("PASS: real CPU measured HTTP, warmup excluded, report readback and overwrite refusal")
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    assert hashes(fixture) == before_fixture, "server modified adapter/tokenizer/settings"
    if registered:
        assert hashes(store) == before_store and performance_report.read_bytes() == before_report
        print("PASS: selected registry real CPU NF4/live HTTP identity; startup snapshot survives pointer switch; registry/report unchanged by server")
    path = fixture / "qlora_settings.json"
    valid = path.read_bytes()
    data = json.loads(valid); data["revision"] = "wrong-revision"
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        api.load_model(fixture, "cpu")
    except ValueError as exc:
        assert "settings" in str(exc)
    else:
        raise AssertionError("wrong base revision accepted")
    path.write_bytes(valid)
    assert hashes(fixture) == before_fixture
    assert hashes(base) == before_base, "server changed base cache"
print("PASS: real CPU NF4/live HTTP equals direct generation; input errors survive; base/candidate unchanged; no training")
