# 22 · 서빙 성능: 응답 지연·생성 속도·GPU 메모리

[← 이전: HTTP API](21-api-serving.md) · [목차](README.md) · [다음: 버전 등록·선택 →](23-model-registry.md)

## 1. 이번에 완성할 결과물

21장의 저장 QLoRA를 그대로 사용해 **측정 서버 시작 → 두 질문 워밍업 → 순차 반복 호출 → 지연·토큰/초·메모리 기록 → JSON 저장 → 종료**를 한 번에 실행합니다. 재학습·튜닝·다른 모델과의 비교·동시 부하·외부 공개는 하지 않습니다. 새 패키지도 설치하지 않습니다.

## 2. 무엇을 재는가

| 지표 | 시작·끝과 해석 |
|---|---|
| `e2e_seconds` | 클라이언트가 HTTP 요청을 보내기 직전부터 응답 본문 바이트를 모두 읽은 직후까지. 연결·서버 처리·전송을 포함하고 클라이언트 JSON 해석은 제외 |
| `generation_seconds` | 서버의 `generate()` 주변 wall-clock 시간. attention mask 생성·Python/CPU 처리·prefill·전체 decode와 마지막 CUDA 동기화를 포함. 토큰화·입력 tensor 생성·응답 디코딩·HTTP 전송·모델 로딩은 제외 |
| `generation_tokens_per_second` | 측정 요청들의 **실제 생성 토큰 합 / generation_seconds 합**. 개별 속도의 단순 평균이 아니며 순수 decode 전용 속도도 아님 |
| `e2e_tokens_per_second` | 실제 생성 토큰 합 / HTTP 지연 합. 서버 최대 처리량·동시 사용자 수를 뜻하지 않음 |
| `allocated_before_bytes` | 생성 직전 해당 프로세스의 PyTorch CUDA 텐서 할당량. 모델·어댑터·입력 등을 포함 |
| `reserved_before_bytes` | 생성 직전 PyTorch allocator가 확보한 메모리. 현재 사용하지 않는 캐시도 포함 가능 |
| `peak_allocated_bytes` / `peak_reserved_bytes` | 이번 요청 직전 peak 통계를 초기화한 뒤 생성 완료까지의 최고치. 모델 로딩 중 peak나 추가 사용량만을 뜻하지 않음 |

GPU 작업은 CPU 호출과 비동기일 수 있으므로 CUDA에서는 타이머 시작 전·종료 전에 `torch.cuda.synchronize(model.device)`로 완료를 기다립니다. peak 초기화는 통계만 초기화하며 메모리를 비우지 않습니다. `empty_cache()`는 호출하지 않아 반복 요청의 캐시 재사용을 유지합니다. CPU 검사에서는 `cuda_memory`가 `null`입니다. CUDA 값은 PyTorch allocator 기준이며 드라이버·다른 프로세스까지 포함한 GPU 전체 사용량이나 `nvidia-smi` 값과 같지 않습니다. bytes를 MiB로 보려면 `1024 ** 2`로 나눕니다.

이 서버는 스트리밍하지 않습니다. **TTFT(첫 토큰 지연), 토큰 사이 지연, decode 전용 속도는 측정하지 않습니다.** 8토큰 이하의 짧은 응답이므로 오버헤드 비중이 큽니다. eos도 생성 토큰 수에 포함될 수 있습니다.

## 3. 터미널 A — 측정 서버 시작

**기존 Ubuntu A10 VM**, `~/llm-lab/source/.venv`에서 실행합니다. 20장의 `../smollm2_qlora_v1/`와 원본 캐시가 필요합니다. 21장 서버가 아직 실행 중이면 그 터미널에서 Ctrl+C로 종료합니다. 다른 프로세스를 임의로 중단하지 않습니다.

기존 `smollm2_api.py`는 보존하고 **새 코드 `smollm2_api_metrics.py`**를 만듭니다. 아래 `set -C`는 기존 파일 덮어쓰기를 막습니다. 이미 파일이 있다면 삭제하지 말고 새 이름을 사용하거나 확인된 파일을 실행합니다. 옵션 없는 서버의 응답 형식은 21장과 같고, `--metrics`일 때만 추가 측정값이 응답에 들어갑니다.

소스: [01-serve-qlora.py](source/21-api-serving/01-serve-qlora.py). 모델 복원·입력 제한·greedy 생성은 그대로 재사용합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
(set -C; cat > smollm2_api_metrics.py <<'PY'
"""Saved QLoRA HTTP demo. Loopback only; not a production server."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import traceback
from time import perf_counter

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def parse_request(body):
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict) or set(data) - {"prompt", "max_new_tokens"}:
        raise ValueError("Use a JSON object with prompt and optional max_new_tokens")
    prompt, limit = data.get("prompt"), data.get("max_new_tokens", 8)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise ValueError("prompt must be a nonblank string of at most 512 characters")
    prompt.encode("utf-8")  # Reject unpaired JSON surrogates before tokenization.
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("max_new_tokens must be an integer from 1 to 32")
    return prompt, limit


def load_model(directory, device):
    import bitsandbytes as bnb
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    settings = json.loads((directory / "qlora_settings.json").read_text(encoding="utf-8"))
    lora = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"], "bias": "none"}
    expected = {"base_model": MODEL, "revision": REVISION, "dtype": "float32", "attention": "eager",
                "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                 "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float32"},
                "lora": lora}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("QLoRA settings mismatch")
    adapter = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    adapter["target_modules"] = sorted(adapter.get("target_modules", []))
    expected_adapter = dict(lora, base_model_name_or_path=MODEL, revision=REVISION,
                            peft_type="LORA", task_type="CAUSAL_LM")
    if any(adapter.get(key) != value for key, value in expected_adapter.items()):
        raise ValueError("adapter config mismatch")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager", device_map={"": device},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, directory, is_trainable=False, local_files_only=True).eval()
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert quantized and all(m.weight.quant_state is not None and m.weight.quant_state.quant_type == "nf4"
                             and m.weight.quant_state.nested for m in quantized)
    assert all(p.device.type == device for p in model.parameters()), "자동 offload 금지"
    assert not any(p.requires_grad for p in model.parameters()), "추론 전용 모델이어야 합니다"
    return tokenizer, model, len(quantized)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "not found"})
        self.send_json(200, {"status": "ok", "device": str(self.server.model.device),
                             "quantized_modules": self.server.quantized_modules})

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "not found"})
        if self.headers.get_content_type() != "application/json":
            return self.send_json(415, {"error": "Content-Type must be application/json"})
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            return self.send_json(400, {"error": "one Content-Length and no Transfer-Encoding required"})
        try:
            length = int(lengths[0])
        except ValueError:
            return self.send_json(400, {"error": "invalid Content-Length"})
        if not 1 <= length <= 4096:
            return self.send_json(413, {"error": "body must contain 1 to 4096 bytes"})
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete request body")
            prompt, limit = parse_request(body)
        except (ValueError, TimeoutError, RecursionError) as exc:
            return self.send_json(400, {"error": str(exc)})
        import torch
        from transformers import GenerationConfig

        tokenizer, model = self.server.tokenizer, self.server.model
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=True, add_generation_prompt=True)
        if len(prefix) > 256:
            return self.send_json(400, {"error": "chat-formatted input exceeds 256 tokens"})
        try:
            inputs = torch.tensor([prefix], device=model.device)
            config = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1, use_cache=True,
                repetition_penalty=1.0, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            measured = getattr(self.server, "metrics", False)
            cuda = model.device.type == "cuda"
            memory = None
            metrics = None
            if measured:
                if cuda:
                    torch.cuda.synchronize(model.device)
                    torch.cuda.reset_peak_memory_stats(model.device)
                    memory = {"allocated_before_bytes": torch.cuda.memory_allocated(model.device),
                              "reserved_before_bytes": torch.cuda.memory_reserved(model.device)}
            start = perf_counter() if measured else None
            with torch.inference_mode():
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
            if measured:
                assert start is not None
                if cuda:
                    torch.cuda.synchronize(model.device)
                seconds = perf_counter() - start
                if memory is not None:
                    memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device))
                metrics = {"generation_seconds": seconds, "cuda_memory": memory}
            result = {"text": tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True),
                      "input_tokens": inputs.shape[1], "generated_tokens": generated.shape[1] - inputs.shape[1]}
            if metrics is not None:
                result["metrics"] = metrics
        except Exception:
            traceback.print_exc()
            return self.send_json(500, {"error": "inference failed; check server terminal"})
        self.send_json(200, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", nargs="?", type=Path, default=Path("../smollm2_qlora_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metrics", action="store_true", help="measure synchronized generation and CUDA allocator peaks")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be from 0 to 65535")
    tokenizer, model, count = load_model(args.adapter, args.device)
    # ponytail: loopback + one request at a time; production needs a serving stack, not external exposure of this demo.
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
        server.metrics = args.metrics
        print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("서버 종료", flush=True)
PY
) && python smollm2_api_metrics.py --device cuda --port 8000 --metrics
```

`READY http://127.0.0.1:8000 ...`는 준비 메시지의 **기대 형식**입니다. 준비될 때까지 기다린 뒤 터미널 A를 계속 켜둡니다. 셸 프롬프트가 돌아오지 않는 것이 정상입니다. 모델은 시작할 때 한 번만 복원하며, 아래 요청에는 다운로드·학습·저장이 없습니다.

## 4. 터미널 B — 워밍업·반복 호출·새 보고서 저장

**같은 VM의 다른 터미널**에서 전체 명령을 실행합니다. 클라이언트는 표준 라이브러리만 사용하며 `127.0.0.1`로만 연결합니다. 질문 두 개를 각각 한 번 워밍업한 다음, 두 질문을 번갈아 각각 5회 측정합니다. 따라서 생성 요청은 총12건, 요약의 `requests`는 워밍업을 제외한10건입니다. 워밍업 원문도 보고서에는 남습니다.

소스: [02-benchmark.py](source/21-api-serving/02-benchmark.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
(set -C; cat > smollm2_benchmark.py <<'PY'
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
PY
) && python smollm2_benchmark.py --repeats 5 --output ../serving_benchmark_v1.json
```

`SUMMARY` 뒤에 지연의 중앙값·최솟값·최댓값과 두 종류의 토큰/초가 나오고 `SAVED ../serving_benchmark_v1.json`으로 마무리됩니다. 이는 기대 출력 구조입니다. 별도로 확인한 CUDA 실행 예시는6절에 구분합니다. 보고서는 `~/llm-lab/serving_benchmark_v1.json`에 새로 만들고 기존 파일이 있으면 요청을 보내기 전에 중단합니다. 재측정은 `--output ../serving_benchmark_v2.json`처럼 새 이름을 사용합니다. 호출 실패·측정값 누락은 성공으로 집계하지 않고 오류로 중단합니다. 실패 전 요청 행은 터미널에 남지만 성공 보고서는 만들지 않습니다.

### 결과 다시 읽기

다음은 저장된 실제 값을 읽는 명령이지 예시 수치를 생성하는 코드가 아닙니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('../serving_benchmark_v1.json').read_text(encoding='utf-8'))
print('health:', report['health'])
print('summary:', report['summary'])
rows = [row for row in report['rows'] if row['phase'] == 'measured']
for name in ['allocated_before_bytes', 'reserved_before_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']:
    values = [row['metrics']['cuda_memory'][name] for row in rows if row['metrics']['cuda_memory'] is not None]
    print(name, 'max MiB:', max(values) / (1024 ** 2) if values else None)
PY
```

`max MiB`는 여러 측정 요청 중 가장 큰 값이며 평균이 아닙니다. `peak_allocated_bytes - allocated_before_bytes`는 그 요청의 기준점 대비 최고치 차이이지 모델 자체 크기·GPU 전체 추가 사용량을 정밀 분해한 값이 아닙니다. `reserved`가 반복 요청 후 높게 남는 것만으로 메모리 누수라고 판단하지 않습니다.

## 5. 코드 흐름과 결과 해석

- 서버의 `--metrics`가 측정을 켭니다. `reset_peak_memory_stats()`는 요청별 최고치 구간을 시작합니다. 동기화 후 `perf_counter()` 차이를 초 단위로 기록합니다. 입력·출력 토큰 수는 추정 글자 수가 아니라 실제 tensor 길이를 사용합니다.
- 클라이언트의 `perf_counter()`는 별도의 HTTP 왕복 구간을 잽니다. 서버/클라이언트의 절대 시각을 빼지 않고 각 프로세스 안의 경과시간만 사용합니다.
- `summarize()`는 워밍업을 뺀 행만 받습니다. 유한한 양수 시간·0 이상의 정수 토큰 수를 확인하고 토큰 합/시간 합으로 속도를 계산합니다. 총 생성량이0이면 속도도0입니다.
- JSON은 설정·health·모든 요청의 질문/생성문/토큰 수/시간/메모리와 요약을 포함합니다. `open('x')`가 기존 보고서 덮어쓰기를 막습니다. 모델 파일에는 쓰지 않습니다.
- 정답이 아닌 응답도 생성 시간 측정의 유효한 관찰입니다. **빠름과 정확함은 별개**입니다. 입력 길이·실제 출력 길이·GPU/드라이버·패키지·다른 작업 유무가 달라지면 직접 비교하기 어렵습니다.
- `generation_seconds`에는 prefill과 decode가 함께 들어가므로 길이가 다른 질문·출력을 비교할 때 조건을 함께 읽습니다. 작은 표본으로 p95·최대 동시 처리량·SLA를 주장하지 않습니다.

두 질문당 한 번 워밍업은 교육용 최소 구성입니다. 충분히 안정화됐음을 보장하지 않습니다. 반복 수는 `--repeats`로1~100까지 바꿀 수 있지만 기본 결과를 확인하기 전에 자동으로 설정 탐색이나 장시간 부하 실험을 추가하지 않습니다. 측정용 동기화·peak 조회 자체의 오버헤드도 있으므로 계측하지 않은 서버의 성능과 같다고 단정하지 않습니다.

호출 뒤 **터미널 A에서 Ctrl+C**, `서버 종료`와 셸 프롬프트 복귀를 확인합니다. `Connection refused`면 먼저 A의 파일 배치·시작 오류·READY·포트를 확인합니다. 측정값 누락 오류라면 21장의 기존 파일이 아니라 `smollm2_api_metrics.py ... --metrics`를 실행했는지 확인합니다. 재설치·재학습·포트 외부 공개로 우회하지 않습니다.

## 6. 검증된 범위와 아직 확인하지 않은 것

실제 Windows/Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2의 **미학습 zero-B NF4 fixture**로 서버 subprocess와 실제 HTTP 요청을 실행했습니다. 측정 없는 기존 API 응답/직접 생성 일치와 입력 거부를 재검사했고, 측정 모드에서 양수 생성 시간·CPU 메모리 `null`·워밍업2건 제외/측정4건 집계·JSON 읽기 일치·기존 보고서 덮어쓰기 거부·원본 캐시/fixture 불변을 확인했습니다. 집계의 독립 수치 검사도 통과했습니다. 이는 코드 경로 검사이며 학습 후보의 A10 성능 수치가 아닙니다.

개발자 재검사(실제 ML 의존성이 있는 저장소 루트):

```bash
python source/21-api-serving/test_benchmark.py
python source/21-api-serving/test_serving.py --integration
python source/21-api-serving/test_serving.py --integration --metrics
python source/check_guide.py
```

검사 소스: [test_benchmark.py](source/21-api-serving/test_benchmark.py), [test_serving.py](source/21-api-serving/test_serving.py). 위 CPU 코드 검증과 아래 CUDA VM 측정은 별도 근거입니다. 모델 로딩 시간·로딩 중 peak·TTFT·동시 부하·다른 양자화와 메모리 절감 비교·운영 안정성은 이번 측정 범위가 아닙니다.

### CUDA VM의 실제 순차 측정 예시

Ubuntu A10 실습 환경의 실행 로그에서 워밍업2건과 측정10건, `SAVED ../serving_benchmark_v1.json`을 확인했습니다. 두 질문의 입력은 각각41·40토큰이고 모든 응답은 새8토큰씩입니다. 측정10건의 실제 생성 토큰 합은80이며, 원문 요청별 값으로 요약을 다시 계산한 결과 아래 로그와 정확히 일치했습니다. CPU 미학습 fixture의 속도와 합치거나 환경 간 배율을 주장하지 않습니다.

```text
SUMMARY {"requests": 10, "e2e_median_seconds": 0.4477258074985002, "e2e_min_seconds": 0.44599267799640074, "e2e_max_seconds": 0.45072476199129596, "generation_tokens_per_second": 17.930662174202777, "e2e_tokens_per_second": 17.857251012975183}
SAVED ../serving_benchmark_v1.json
```

| 지표 | 실제 결과 |
|---|---|
| HTTP 지연 중앙값 | 447.73ms |
| HTTP 지연 최솟값~최댓값 | 445.99~450.72ms |
| 생성 구간 속도 | 17.93토큰/초 |
| HTTP 왕복 포함 속도 | 17.86토큰/초 |
| 생성 전 allocated | 측정10건 모두169.92MiB |
| 생성 구간 peak allocated | gpu질문176.04MiB / network질문175.96MiB |
| 생성 전 reserved / 생성 구간 peak reserved | 측정10건 모두316.00MiB /316.00MiB |

표는 초→ms와 bytes→MiB로 환산해 소수 둘째 자리까지 표시했습니다. 첫 워밍업의 HTTP 지연은714.11ms였지만 이는 요약에서 제외됩니다. 이번 짧은 반복 구간의 지연과 메모리 값은 비슷하게 유지됐습니다. 이것만으로 워밍업 차이의 원인, 장시간 안정성, 메모리 누수 부재나 최대 처리량을 입증하지는 않습니다.

생성문은 각각 `Device GPU status:\n\nDevice GPU`와 `Device network status:\n\nDevice network`로 반복됐으며, 한 단어 정답 `failed`·`healthy`는 여전히 생성하지 못했습니다. 성능 측정 성공을 품질 개선으로 읽지 않습니다. `reserved`316MiB는 모델 파일 크기·GPU 전체 사용량·다른 정밀도 대비 절감량이 아닙니다.

저장은 위 `SAVED` 로그로 확인한 범위이며 저장 파일을 다시 읽은 별도 출력은 이 로그에 포함하지 않습니다. 측정이 끝난 뒤에도 터미널 A의 서버는 계속 실행되므로5절의 Ctrl+C로 종료합니다. 이 숫자는 해당 입력·설정·실행 환경의 소규모 관찰이며 재실행 시 달라질 수 있습니다.

## 7. 복습

- [ ] HTTP 왕복 시간과 서버 generation 시간의 구간을 구분한다.
- [ ] 실제 생성 토큰 합/시간 합과 개별 속도의 평균이 다른 이유를 설명한다.
- [ ] 워밍업은 보고서에 남지만 요약에서 제외됨을 확인한다.
- [ ] allocated/reserved와 요청 구간 peak, GPU 전체 메모리를 구분한다.
- [ ] 성능 수치와 답변 품질을 구분하고 새 보고서로 저장한 뒤 서버를 종료한다.

## 공식 참고자료

- [Python3.12 perf_counter](https://docs.python.org/3.12/library/time.html#time.perf_counter)
- [PyTorch2.9 CUDA synchronize](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html)
- [PyTorch2.9 reset_peak_memory_stats](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.reset_peak_memory_stats.html)
- [PyTorch2.9 memory_allocated](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.memory_allocated.html)
- [PyTorch2.9 CUDA memory management](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-memory-management)

[← 이전: HTTP API](21-api-serving.md) · [목차](README.md) · [다음: 버전 등록·선택 →](23-model-registry.md)
