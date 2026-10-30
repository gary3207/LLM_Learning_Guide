# 21 · 저장한 QLoRA를 HTTP API로 호출하기

[← 이전: QLoRA](20-qlora.md) · [목차](README.md) · [다음: 서빙 성능 측정 →](22-serving-benchmark.md)

## 1. 이번에 완성할 결과물

20장은 명령을 실행할 때 모델을 복원하고 질문에 응답했습니다. 이번에는 **저장한 QLoRA를 한 번 복원 → 서버를 계속 실행 → 별도 터미널에서 HTTP 요청 → JSON 응답 확인 → 서버 종료**를 한 묶음으로 진행합니다.

재학습·어댑터 병합·모델 재저장은 없습니다. 기존 `smollm2_qlora_v1/`와 고정 원본 캐시를 읽기만 합니다. 응답 품질을 고치는 실습도 아니므로 이전의 한 단어 정답 생성 실패를 API가 해결한다고 기대하지 않습니다.

Python 표준 라이브러리의 `HTTPServer`를 사용합니다. FastAPI·Uvicorn·vLLM·컨테이너는 설치하지 않습니다. **외부에 공개하는 운영 서버가 아니라 같은 VM 안에서 HTTP 흐름을 배우는 예제**입니다. `127.0.0.1`에만 바인딩하며 외부 주소 옵션·인증·TLS·스트리밍·배칭은 제공하지 않습니다.

## 2. 서버와 클라이언트는 무엇이 다른가

```text
터미널 A: 원본 NF4 + 저장 어댑터 로딩 → HTTP 서버 대기
                                        ↑ 요청 / ↓ 응답
터미널 B: 질문을 JSON으로 POST → 생성 텍스트와 토큰 수 출력
```

- **서버:** 모델을 메모리에 유지하고 요청을 처리하는 Python 프로세스입니다. 요청마다 모델을 다시 다운로드하거나 복원하지 않습니다.
- **클라이언트:** 서버에 질문을 보내고 응답을 읽는 별도 프로그램입니다. 자체적으로 모델을 로딩하지 않습니다.
- **HTTP:** 주소·메서드·헤더·본문·상태코드로 요청과 응답을 주고받는 형식입니다.
- **JSON:** 질문과 결과를 문자열·숫자·객체로 표현하는 데이터 형식입니다. Python 코드 실행을 요청하는 것이 아닙니다.
- **localhost/127.0.0.1:** 명령을 실행하는 그 컴퓨터입니다. 따라서 아래 두 터미널은 모두 **같은 Ubuntu VM**에 접속해야 합니다. PC에서 같은 주소를 호출하면 PC 자신을 가리킵니다.

| 요청 | 용도 | 성공 결과 |
|---|---|---|
| `GET /health` | 모델 로딩이 끝난 서버의 응답 확인 | HTTP200, status/device/quantized_modules |
| `POST /generate` | 질문으로 자유 생성 | HTTP200, text/input_tokens/generated_tokens |

`/health` 성공은 서버가 준비됐다는 뜻이지 모델이 정답을 생성한다는 뜻이 아닙니다. OpenAI API 호환 규격이 아니라 이 장만의 작은 인터페이스입니다.

## 3. 터미널 A — 저장 모델을 읽고 서버 실행

기존 Ubuntu VM의 `~/llm-lab/source/.venv`를 사용합니다. 20장 실행을 마쳐 다음 파일과 원본 캐시가 있어야 합니다.

- `~/llm-lab/smollm2_qlora_v1/adapter_model.safetensors`
- 같은 폴더의 `adapter_config.json`, `qlora_settings.json`, 토크나이저 파일
- 고정 revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` 원본 캐시

새 코드는 `~/llm-lab/source/smollm2_api.py`입니다. **새 모델 출력 폴더는 만들지 않습니다.** 기존 PyTorch·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2를 사용하고 새 패키지를 추가하지 않습니다. 모든 모델 로딩은 `local_files_only=True`이며 캐시가 없으면 다운로드 대신 오류를 알립니다.

아래 전체 명령을 터미널 A에서 실행합니다. `set -C`가 기존 코드 파일 덮어쓰기를 막습니다. `&&`는 앞 단계가 실패하면 다음 단계를 실행하지 않습니다. `python -O`는 검사문을 끄므로 사용하지 않습니다.

소스: [01-serve-qlora.py](source/21-api-serving/01-serve-qlora.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
(set -C; cat > smollm2_api.py <<'PY'
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
) && python smollm2_api.py --device cuda --port 8000
```

모델 로딩 후 아래 형태의 준비 메시지가 나오는 것이 성공 기준입니다. **이 줄은 준비 메시지의 기대 형식이며, 별도로 확인한 실제 CUDA HTTP 응답 예시는6절에 구분합니다.**

```text
READY http://127.0.0.1:8000 | device=cuda:0 | NF4=210
```

이후 터미널에 셸 프롬프트가 돌아오지 않는 것이 정상입니다. 서버가 요청을 기다리는 중이므로 이 터미널을 그대로 두고 다음 터미널을 엽니다. bitsandbytes의 기존 `FutureWarning`이 나타나면 준비 메시지·요청 결과·마지막 오류를 함께 확인하며 임의로 패키지를 바꾸지 않습니다.

## 4. 터미널 B — health와 두 질문 호출

**같은 Ubuntu VM에 터미널을 하나 더 연결**하고 아래를 실행합니다. Python 표준 라이브러리만 사용하는 클라이언트이며 별도 파일을 만들지 않습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'PY'
import json
from urllib.request import Request, urlopen

base = 'http://127.0.0.1:8000'
with urlopen(base + '/health', timeout=60) as response:
    print('health:', response.status, json.load(response))

for prompt in ['Device gpu status? Reply with one word.',
               'Device network status? Reply with one word.']:
    body = json.dumps({'prompt': prompt, 'max_new_tokens': 8}).encode('utf-8')
    request = Request(base + '/generate', data=body,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        print('prompt:', prompt)
        print('response:', response.status, json.load(response))
PY
```

`urlopen`이 주소만 받으면 GET, `Request`에 `data`를 넣으면 여기서는 POST가 됩니다. `Content-Type`은 본문이 JSON임을 알리고, `json.dumps(...).encode('utf-8')`가 Python 사전을 전송할 바이트로 만듭니다. 응답은 `json.load`로 다시 Python 객체로 읽습니다.

### 요청·응답 읽기

요청은 `prompt`와 선택적 `max_new_tokens`만 받습니다. `max_new_tokens` 기본값은8이며1~32의 정수만 허용합니다. 출력 토큰 수 제한이지 영어 단어 수나 글자 수 제한이 아닙니다.

| 응답 필드 | 의미 |
|---|---|
| `text` | 입력을 제외한 새 생성 토큰을 디코딩한 텍스트 |
| `input_tokens` | 대화 템플릿을 포함해 실제 입력한 토큰 수 |
| `generated_tokens` | 실제 새 생성 토큰 수. eos가 생성되면 수에 포함되지만 text에서는 숨겨질 수 있음 |

정답 응답을 붙여 teacher-forced loss를 계산하지 않습니다. 질문만 넣고 greedy 생성을 하므로 이전 장의 생성과 연결되며, HTTP200 자체를 정답 판정으로 읽지 않습니다. 같은 모델·장치·설정에서 직접 생성과 API 생성을 비교하는 것이 서버 경로 확인의 핵심입니다.

## 5. 코드 흐름·입력 제한·종료

### 복원은 시작할 때 한 번만

`load_model()`은 저장 설정의 원본 ID/revision·float32·NF4/double quant·q/v rank4와 어댑터 설정을 검사합니다. 20장과 동일하게 4비트 원본을 준비하고 어댑터를 추론 전용으로 붙입니다. 실제 NF4 모듈·동일 장치 배치·모든 파라미터 동결을 확인한 뒤에만 서버 소켓을 엽니다. CUDA가 없으면 CPU로 자동 전환하지 않습니다.

이 함수는 `__main__`에서 한 번 호출합니다. 요청 처리 함수에는 복원·optimizer·backward·save가 없고, 메모리에 있는 모델의 `generate()`만 `torch.inference_mode()`에서 실행합니다. 학습용 `reference.pt`는 읽지 않습니다.

### HTTP 입력은 제한한다

- 본문1~4096bytes, `Content-Type: application/json`, 하나의 Content-Length를 요구합니다. chunked 요청은 받지 않습니다.
- `prompt`는 공백뿐인 문자열을 거부하고 최대512글자·UTF-8 표현 가능한 문자열만 허용합니다. 대화 템플릿 적용 뒤 입력은 최대256토큰입니다.
- 형식/입력 오류는400, 너무 큰 본문은413, 다른 Content-Type은415, 없는 경로는404로 응답합니다. bool을 정수 토큰 제한으로 받아들이지 않습니다.
- 생성 중 예외는500과 일반 오류 메시지로 알리고 자세한 오류는 서버 터미널에 남깁니다. 요청 본문을 별도 파일로 저장하지 않습니다.
- 소켓 읽기 제한10초는 불완전한 HTTP 입력의 대기를 제한하며, 모델 생성 전체의 강제 중단 시간은 아닙니다. 클라이언트의60초 timeout도 서버의 생성 작업을 취소하는 기능은 아닙니다.

**한 번에 요청 하나만 처리합니다.** 생성 중에는 health 요청도 기다릴 수 있습니다. 동시 처리·부하 테스트·큐·인증·운영용 HTTP 스택은 이번 범위가 아닙니다. 운영 서버로 쓰기 위해 이 예제의 바인딩만 `0.0.0.0`으로 바꾸지 않습니다. OCI 보안 목록·방화벽·포트 공개 설정을 변경할 필요가 없습니다.

### 서버 종료와 흔한 오류

호출을 마치면 **터미널 A에서 Ctrl+C**를 누릅니다. `서버 종료`가 출력되고 소켓이 닫히며 셸 프롬프트가 돌아오는 것이 종료 기준입니다. 종료·재실행해도 모델 파일을 다시 학습하거나 덮어쓰지 않습니다.

- `Connection refused`: 터미널 A의 준비 메시지와 실행 중 여부, 같은 VM/포트인지 확인합니다.
- `Address already in use`: 다른 서버가 해당 포트를 쓰고 있습니다. 이번에 직접 실행한 기존 서버라면 해당 터미널에서 Ctrl+C로 종료합니다. 다른 프로세스를 임의로 죽이지 않습니다.
- 원본 캐시/어댑터/설정 오류: 경로와 20장 결과를 확인합니다. 재학습이나 캐시 삭제로 자동 우회하지 않습니다.

## 6. 실제 로컬 검증과 한계

Windows·Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2 격리 환경에서 실제 NF4 원본과 **미학습 zero-B 어댑터 fixture**로 서버를 별도 프로세스에서 실행했습니다. 실제 HTTP 응답이 같은 fixture의 직접 생성과 일치하고, 잘못된 입력 뒤에도 서버가 살아 있는지, 원본 캐시·어댑터 파일이 바뀌지 않는지 검사했습니다. optimizer나 재학습은 없습니다.

아래는 실제 개발자 검사 출력입니다. **학습된 QLoRA의 응답 품질이나 A10 CUDA API 실행을 검증한 결과가 아닙니다.** 20장의 CUDA train/check 통과가 이 새 HTTP 서버의 CUDA 실행을 대신하지 않습니다.

```text
PASS: request JSON/type/length/token-limit contract
HTTP 200 real NF4 untrained fixture: {"text": "Device GPU status:\n\nDevice GPU", "input_tokens": 41, "generated_tokens": 8}
HTTP 200 real NF4 untrained fixture: {"text": "Device network status:", "input_tokens": 40, "generated_tokens": 4}
PASS: real CPU NF4/live HTTP equals direct generation; input errors survive; base/candidate unchanged; no training
```

개발자 환경의 저장소 루트에서 다음과 같이 재검사할 수 있습니다. 첫 검사는 표준 라이브러리만, 두 번째는 실제 ML 의존성과 원본 캐시를 사용합니다. 통합 검사의 fixture는 임시 폴더에 만들며 기존 학습 후보를 사용하거나 바꾸지 않습니다.

```bash
python source/check_guide.py
python source/21-api-serving/test_serving.py
python source/21-api-serving/test_serving.py --integration
```

검사 소스: [test_serving.py](source/21-api-serving/test_serving.py). VM의 학습 어댑터를 사용한 API 응답과 Ctrl+C 종료는 위 실습의 실제 출력으로 따로 확인합니다. 현재 검사는 운영 안정성·성능·GPU 메모리·모델 품질 평가가 아닙니다.

### Ubuntu VM의 CUDA HTTP 실행 예시

가이드 환경에서 실행한 제공 로그에서 `/health`와 두 `/generate` 요청 모두 HTTP200으로 응답한 것을 확인했습니다. health 응답은 `cuda:0`·NF4 모듈210개를 보고하며, 두 질문의 입력은 각각41·40토큰, 새 생성은 각각8토큰입니다. 위 CPU 미학습 fixture 검사와 별도 결과입니다.

```text
health: 200 {'status': 'ok', 'device': 'cuda:0', 'quantized_modules': 210}
prompt: Device gpu status? Reply with one word.
response: 200 {'text': 'Device GPU status:\n\nDevice GPU', 'input_tokens': 41, 'generated_tokens': 8}
prompt: Device network status? Reply with one word.
response: 200 {'text': 'Device network status:\n\nDevice network', 'input_tokens': 40, 'generated_tokens': 8}
```

HTTP 요청·응답 경로는 성공했지만 한 단어 정답 `failed`·`healthy`를 생성하지는 못했습니다. 서빙은 기존 모델을 호출하는 경로를 제공할 뿐 응답 품질을 자동 개선하지 않습니다. 이 응답은 지연시간·처리량·GPU 메모리·동시 처리·독립 품질 평가의 결과가 아닙니다. 호출 뒤 서버 종료는5절의 Ctrl+C 절차로 마무리합니다.

## 7. 복습

선택형 `--metrics`는 [22장](22-serving-benchmark.md)에서 사용합니다. 이 장의 실행 명령은 해당 옵션을 켜지 않아 기존 응답 형식과 생성 조건을 유지합니다.

- [ ] 모델을 메모리에 올리는 과정과 요청 한 건을 처리하는 과정을 구분한다.
- [ ] 서버와 클라이언트가 서로 다른 프로세스라는 점을 설명한다.
- [ ] HTTP200·모델 정답·서빙 성능이 별개임을 설명한다.
- [ ] 127.0.0.1이 그 명령을 실행한 컴퓨터 자신을 뜻함을 이해한다.
- [ ] 학습 없이 저장 모델을 호출하고 Ctrl+C로 서버를 종료한다.

## 공식 참고자료

- [Python3.12 http.server — 운영용이 아닌 기본 HTTP 서버](https://docs.python.org/3.12/library/http.server.html)
- [Python3.12 urllib.request — HTTP 요청](https://docs.python.org/3.12/library/urllib.request.html)
- [Transformers4.57.1 생성·GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)
- [PEFT 모델 복원](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model)

[← 이전: QLoRA](20-qlora.md) · [목차](README.md) · [다음: 서빙 성능 측정 →](22-serving-benchmark.md)
