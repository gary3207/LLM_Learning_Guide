# 29. 운영 관찰·작은 동시 부하 검사

[← 이전: 배포·롤백](28-deployment-rollback.md) · [목차](README.md) · [다음: Docker 실행환경 →](30-container-runtime.md)

## 1. 목표와 범위

LLMOps 여섯 묶음 중 **5/6 운영 관찰·부하 검사**입니다. 기존 등록 버전 API를 **측정용 loopback 서버**로 실행하고 순차 요청과 제한된 동시 요청을 비교합니다. 오류·응답 지연·처리량·생성 시간·CUDA allocator 메모리를 새 JSON 보고서에 남깁니다.

> **품질 미달 모델을 승인 배포하지 않습니다.** 27장의0/8 판정과28장의배포 거부는 유지합니다. 이번 실행은24장의 읽기 전용 진단 API를 활용하는 성능 실습이며,28장의service/active 상태·선택 포인터·평가 보고서를 변경하지 않습니다. 외부 사용자에게 제공하지 않고 시험이 끝나면 직접 종료합니다.

새 서버 프레임워크·데몬·모니터링 플랫폼·학습·튜닝·품질 우회 옵션은 추가하지 않습니다. Docker는 다음 묶음입니다. RAG는 포함하지 않습니다.

## 2. 이번에 비교하는 값

### 요청을 동시에 보내는 것과 모델이 동시에 실행되는 것은 다릅니다

24장의 서버는 표준 라이브러리 `HTTPServer`로 요청을 한 번에 하나씩 처리합니다. 새 클라이언트의 `ThreadPoolExecutor`는 요청을 보내는 쪽에만 있습니다. 서버를 다중 스레드로 바꾸거나 모델을 여러 개 로딩하지 않습니다.

- **순차 단계:** 요청 하나의 응답을 받은 다음 다음 요청을 보냅니다.
- **동시 단계:** 지정한 수 이하의 클라이언트 작업을 함께 실행합니다. 아직 시작하지 않은 작업은 executor에서 기다릴 수 있고, 연결한 요청도 서버 차례를 기다릴 수 있습니다.
- 두 단계 모두 같은 두 질문과 새 토큰 제한8을 사용합니다. 요청 순서·운영체제 스케줄링·GPU 상태 때문에 매번 동일한 지연은 보장하지 않습니다.

작업이 executor에서 대기하는 시간과 HTTP 호출을 시작한 뒤 기다리는 시간도 구분합니다. 여기서 **요청별 지연**은 HTTP 호출의 시작부터 응답 본문의JSON 해석·검사와 연결 정리까지입니다. 작업이 executor에서 시작하기 전의 대기는 이 요청별 값에 포함하지 않습니다. **단계 전체 시간**은 해당 요청 묶음을 실행하는 데 걸린 wall time입니다.

### 지연·속도·메모리의 분모

| 관찰값 | 해석 |
|---|---|
| 클라이언트 응답 시간의 중앙값·p95 | 성공한 요청의 분포. p95는 정렬 후 nearest-rank 방식이며 작은 표본은 안정적인 운영 SLO 추정이 아님 |
| 요청별 `generation_seconds` | 기존 서버가 측정한 생성 구간. CUDA에서는 동기화 뒤 측정하며 HTTP 대기와 다름 |
| 생성 토큰 수 / 생성 시간 합 |22장과 같은 생성 구간의 가중 평균 토큰 속도. 첫 토큰 지연이나 전체 단계 처리량이 아님 |
| 생성 토큰 수 / 클라이언트 시간 합 | 요청 지연 합을 분모로 한 값. 동시 호출의 시간이 겹치므로 전체 처리량이라고 해석하지 않음 |
| 성공 토큰 수 / 단계 wall time | 그 단계의 실제 관측 토큰 처리량. 준비·대기·실패가 소비한 단계 시간도 분모에 포함 |
| 성공 요청 수 / 단계 wall time | 해당 단계의 성공 요청 처리량. 시도 수와 성공 수를 혼동하지 않음 |
| CUDA peak allocated/reserved | 기존 서버가 요청마다 재설정해 관찰한 PyTorch allocator 최대값. 전체 GPU VRAM·GPU 사용률·CPU/RAM 사용률이 아님 |

클라이언트 지연에서 생성 시간을 빼도 정확한 큐 대기 시간은 아닙니다. 연결·HTTP 처리·토큰화·디코딩 등이 섞입니다. 비스트리밍 응답이므로 TTFT나 토큰 간 지연도 측정하지 않습니다. CPU 응답의 CUDA 메모리는 `null`이며0으로 꾸미지 않습니다.

### 의도한400과 실제 장애

빈 prompt를 한 번 보내400으로 거부되는지 확인합니다. 이것은 **의도한 입력 검증 probe**이며 순차/동시 측정의 장애율 분모에 넣지 않습니다. 이후 정상 요청으로 생성이 계속되는지 확인합니다.

반대로 측정 중500·timeout·잘못된JSON·모델 식별 정보 불일치·측정값 누락은 실패로 보관합니다. 성공한 행만으로 지연·토큰 속도를 계산하되 전체 시도 수와 실패 수를 함께 봅니다. 성공이 하나도 없으면 성공 지연 요약은 `null`이며 가짜0 지연을 만들지 않습니다. 자동 재시도는 하지 않습니다. 클라이언트 timeout이 서버의 모델 생성을 취소했다는 뜻도 아닙니다.

## 3. 실행 위치와 재사용 파일

모든 명령은 **같은 Ubuntu GPU VM의 `~/llm-lab/source`와 기존 `.venv`**에서 실행합니다. 두 터미널을 사용합니다.

| 파일 | 역할 |
|---|---|
| `smollm2_api_metrics.py` |22장의 기존 NF4 로더·HTTP handler·생성/메모리 측정 |
| `smollm2_registry.py` |23장의 기존 선택 버전·파일 지문 검증 |
| `smollm2_serve_selected.py` |24장의 기존 등록 버전 서버. 이미 `--metrics`를 지원 |
| `smollm2_benchmark.py` |22장의 기존 질문·정상 응답 집계 함수 |
| `smollm2_observe_load.py` | 이번에 추가하는 작은 부하 클라이언트 |
| `../model_registry_v1` | 기존 레지스트리. 읽기 전용 |
| `../serving_load_v1.json` | 이번에 새로 저장하는 보고서. 기존 파일 덮어쓰기 금지 |

기존 코드를 복제하거나 예전 API 파일을 덮어쓰지 않습니다.24장의 서버를 그대로 `--metrics`로 켜며,22장의 집계 함수를 새 클라이언트가 재사용합니다. `../serving_deployment_v1`의 부재를 실패로 취급하거나 가짜 승인 상태를 만들지 않습니다.

## 4. 터미널 A — 기존 진단 서버 실행

기존8000 수동 서버나8001 배포용 unit은 제어하지 않습니다. 이번 진단은 별도8002 포트를 씁니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python smollm2_serve_selected.py \
  --registry ../model_registry_v1 --device cuda --metrics --port 8002
```

- `--metrics`: 기존 생성 구간·CUDA allocator 관찰값을 응답에 포함합니다.
- `--port 8002`: 같은 VM에서만 접근하는 진단 포트입니다. 방화벽을 열지 않습니다.
- `READY http://127.0.0.1:8002`를 확인한 뒤 **이 터미널을 켜 둡니다.** foreground로 기다리는 것은 정상입니다.
- 이미 포트를 쓰고 있거나 시작 오류가 나면 다른 프로세스를 임의로 종료하지 않습니다. 출력으로 원인을 확인합니다.
- 이 서버는 품질 게이트를 통과한 운영 서비스가 아닙니다. 측정 실습이라는 제한을 유지합니다.

## 5. 터미널 B — 새 클라이언트와 제한된 요청 묶음

정본: [01-observe-load.py](source/29-operations-load/01-observe-load.py). 레지스트리와 모델, 소스 안을 출력 경로로 쓰지 않으며 기존 보고서도 보존합니다.

### 클라이언트 생성

새 파일 생성과 실행을 한 묶음으로 연결했습니다. 생성 실패 시 뒤의 측정도 실행하지 않습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path
path = Path('smollm2_observe_load.py')
if path.exists() or path.is_symlink():
    raise FileExistsError('기존 클라이언트를 덮어쓰지 않습니다.')
code = r'''"""Bounded loopback diagnostic client for chapter 24's --metrics server.

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
'''
with path.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(code)
print('CREATED', path, path.stat().st_size, 'bytes')
CREATE
{
python smollm2_observe_load.py \
  --registry ../model_registry_v1 --port 8002 \
  --requests 6 --workers 3 --timeout 30 \
  --output ../serving_load_v1.json
result=$?
printf 'LOAD_EXIT_CODE=%s\n' "$result"
}
```

### 실행 흐름과 종료 코드

위 명령은 생성 직후 부하 검사를 실행합니다. 터미널 A의READY를 먼저 확인하고 별도 터미널 B에서 실행하세요. 성공은종료0, 예상하지 못한 요청/관찰 오류는종료1입니다. 사전health·설정 오류는 보고서 생성 전에 종료할 수 있으며, 측정 중 실패는 가능한 행과 함께 새 보고서에 보관합니다.

`--requests`는 **단계마다** 보내는 요청 수이고, `--workers`는 동시 단계의 최대 클라이언트 작업 수입니다. 기본 흐름은health 확인→입력 검증probe→별도 warmup→순차 측정→동시 측정→마지막health 확인→새 보고서 저장입니다. `SUMMARY` 두 줄과 `SAVED` 뒤에 `LOAD_EXIT_CODE`를 확인합니다.

`--timeout`은 소켓 동작의 대기 제한이며 전체 실습의 wall time 제한이나 서버 작업 취소가 아닙니다.

워밍업·의도한400·health 요청은 측정 단계의 성공/실패 분모에서 제외합니다. 처음에는 이 작은 요청 수를 유지합니다. 높은 부하나 장시간 운전으로 확대하지 않습니다.

## 6. 결과 읽기와 종료

### 코드에서 따라갈 지점

- `exchange`: proxy나redirect 없이127.0.0.1의 지정 포트만 호출합니다. 응답은64KiB로 제한하고 소켓 오류·JSON 오류도 관찰 행으로 반환합니다.
- `check_health`/`validate_generation`: 레지스트리에서 고정한 전체 모델 식별 정보와 응답·측정값의 형식을 검사합니다. CPU의null 메모리와 실제 CUDA allocator 관찰을 구분합니다.
- `run`: 입력·출력 보호를 먼저 확인하고 probe·워밍업·두 측정 단계·마지막health를 실행합니다. `pool.map` 결과 행은 입력 순서이며 응답 완료 순서가 아닙니다.
- 단계 집계:22장의 `summarize`를 성공 행에만 적용하고 요청 지연 합과 wall time 기준 토큰 속도를 구분해 이름 붙입니다. 실패 행은 삭제하지 않습니다.
- `main`: 단계별SUMMARY와 저장 경로를 출력합니다. 전체 `ok`는 품질 승인이 아니라 probe·워밍업·측정·마지막health의 정상 여부입니다.

실행 성공 여부는 측정이 정상적으로 끝났는지를 말할 뿐 **질문에 정답을 생성했는지**나 **모델이 품질 승인을 받았는지**를 말하지 않습니다. 별도 CUDA VM 실행 예는 아래에 로컬 CPU 결과와 구분해 기록했습니다. 그 수치를 다른 환경의 보장값으로 사용하지 않습니다.

1. 순차·동시 단계의 시도/성공/실패 수가 의도한 요청 수와 맞는지 봅니다.
2. 의도한400 probe가 정상 거부됐는지, 측정 단계 오류와 분리됐는지 봅니다.
3. 동일한 `model_version`의 전체 식별 정보가health·모든 성공 생성 응답·마지막health에서 일치하는지 봅니다. 버전 이름만 같다고 판단하지 않습니다.
4. 지연 중앙값/p95와 단계 wall time 처리량을 함께 비교합니다. 동시 요청의 지연이 늘어도 모델이 여러 개 병렬 실행됐다고 해석하지 않습니다.
5. CUDA allocator peak를 요청 간 합산하지 않습니다. 요청별 값과 관찰된 최대값을 봅니다. 이는 장시간 메모리 누수 검사가 아닙니다.
6. 실패 행은 삭제하지 않습니다. timeout은 서버가 그 작업을 끝냈는지 모른다는 한계와 함께 읽습니다.

보고서의 `failure_count`는 HTTP 오류뿐 아니라 응답 검증 실패도 포함합니다. `no_http_status_count`는 HTTP 상태 자체를 받지 못한 시도 수입니다. 상태200을 받았더라도 본문 수신·JSON·식별 정보 검사가 실패하면 실패 행입니다. 따라서 HTTP200 수를 정상 생성 수로 단정하지 않습니다.

보고서를 받은 뒤 **터미널 A에서 Ctrl+C**로 이번 진단 서버를 종료합니다. 기존 수동 서버·배포 unit·WSL 전체를 종료하는 명령은 쓰지 않습니다. READY와 보고서 생성만으로 정상 종료까지 완료했다고 기록하지 않습니다.

## 7. 검증 범위와 한계

| 근거 | 확인할 범위 |
|---|---|
| 표준 라이브러리의 실제 loopback HTTP 검사 | 10개 검사 통과. 합성 transport의 요청 수·400 분리·실패 보관·순차 서버와 동시 클라이언트·요약 분모·입출력 보호. 동시 클라이언트 최대4와 서버 처리 최대1을 별도로 확인 |
| 실제 CPU NF4 검사 | 기존24장 서버/22장측정·실제 모델 생성과 새 클라이언트 연결 통과. 210개 NF4 모듈·워밍업2건·각 단계6건 성공/0실패·예상400·식별 정보 일치·기존 입력 보존·테스트 리스너 종료 재조회. 미학습 fixture이며 학습 모델 품질·GPU 성능이 아님 |
| 별도 CUDA VM | 순차/동시 각6건 성공·0실패·보고서 저장·클라이언트 종료0 확인. CUDA allocator 관찰값을 실제 출력으로 보존. 서버READY/정상 종료 로그는 별도 미확인 |

테스트 정본: [test_load.py](source/29-operations-load/test_load.py). 표준 라이브러리 검사 명령입니다. 합성 응답을 검증하는 테스트이며 모델 추론 결과가 아닙니다.

```bash
python source/29-operations-load/test_load.py
python source/21-api-serving/test_benchmark.py
python source/check_guide.py
```

### 로컬 CPU NF4 실제 관찰 예

Windows/Python3.12·CPU·미학습 어댑터에서 실제 생성한 한 실행의 결과입니다. 양쪽 단계 모두 요청6건·새 토큰 합48이며, 지연은 소수 셋째 자리까지 표시했습니다.

| 단계 | 클라이언트 작업 수 | 성공/실패 | 지연 중앙값(초) | p95(초) | 단계 wall time 기준 토큰/초 |
|---|---:|---:|---:|---:|---:|
| 순차 | 1 | 6/0 | 3.269 | 3.384 | 2.436 |
| 동시 | 3 | 6/0 | 12.154 | 14.166 | 1.975 |

이 실행에서는 동시 요청의 지연이 늘었고 처리량은 개선되지 않았습니다. 단일 요청 서버에 클라이언트 수만 늘린 결과이며 일반적인 용량 결론은 아닙니다. CPU의CUDA 메모리 값은 `null`입니다. 같은 수치의 재현·A10 GPU 속도·학습 모델 품질을 보장하지 않습니다. 테스트 프로세스는 별도로 정리하고 포트 종료를 재조회했지만, 이는 VM에서Ctrl+C로 종료했다는 증거가 아닙니다.

### 별도 CUDA VM의 실제 관찰 예

아래는 별도 CUDA VM의 실제 실행 출력입니다. 로컬 CPU 미학습 fixture 수치와 구분합니다. 두 단계 모두6건 성공·0건 실패이고 보고서 저장 뒤 클라이언트가 종료0으로 셸에 복귀했습니다.

```text
CREATED smollm2_observe_load.py 12880 bytes
SUMMARY {"name": "sequential", "workers": 1, "phase_wall_seconds": 2.7231570290168747, "max_observed_cuda_memory": {"peak_allocated_bytes": 184592384, "peak_reserved_bytes": 331350016}, "no_http_status_count": 0, "failure_count": 0, "http_status_counts": {"200": 6}, "summary": {"requests": 6, "e2e_median_seconds": 0.4539858559874119, "e2e_min_seconds": 0.45260598900495097, "e2e_max_seconds": 0.45434278799802996, "generation_tokens_per_second": 17.699766372505735, "tokens_per_sum_client_seconds": 17.63114398717225, "tokens_per_phase_wall_second": 17.626600114694508, "successful_requests_per_phase_wall_second": 2.2033250143368135, "e2e_p95_seconds": 0.45434278799802996, "mean_client_minus_generation_seconds": 0.001759169841534458}}
SUMMARY {"name": "concurrent", "workers": 3, "phase_wall_seconds": 2.720901683002012, "max_observed_cuda_memory": {"peak_allocated_bytes": 184592384, "peak_reserved_bytes": 331350016}, "no_http_status_count": 0, "failure_count": 0, "http_status_counts": {"200": 6}, "summary": {"requests": 6, "e2e_median_seconds": 1.3595823969808407, "e2e_min_seconds": 0.4542930930037983, "e2e_max_seconds": 1.3602752500155475, "generation_tokens_per_second": 17.702867275402696, "tokens_per_sum_client_seconds": 7.058120381845497, "tokens_per_phase_wall_second": 17.64121074269794, "successful_requests_per_phase_wall_second": 2.2051513428372425, "e2e_p95_seconds": 1.3602752500155475, "mean_client_minus_generation_seconds": 0.6815420429969284}}
SAVED ../serving_load_v1.json
LOAD_EXIT_CODE=0
```

| 단계 | 클라이언트 작업 수 | 성공/실패 | 지연 중앙값(ms) | p95(ms) | 단계 wall time 기준 토큰/초 |
|---|---:|---:|---:|---:|---:|
| 순차 | 1 | 6/0 | 453.99 | 454.34 | 17.63 |
| 동시 | 3 | 6/0 | 1359.58 | 1360.28 | 17.64 |

표는 지연·속도를 소수 둘째 자리로 표시했습니다. 두 단계의 peak allocated는176.04MiB, peak reserved는316.00MiB로 같았습니다. 이 값은 관찰된 CUDA allocator 최대값이며 전체 GPU VRAM이나GPU 사용률이 아닙니다.

이 표본에서 동시 요청의 중앙 지연은 약3배가 됐지만 단계 처리량은 거의 같았습니다. 한 번에 하나를 처리하는 서버에 요청이 겹쳤다는 설명과 맞으며, GPU 병렬 추론의 처리량 개선이나 최대 용량을 증명하지 않습니다. `mean_client_minus_generation_seconds` 역시 정확한 큐 대기 시간이 아닙니다.

종료0은 진단 검사가 정상 완료됐다는 뜻이지 응답의 정답 품질·승인 배포가 아닙니다. 이 로그에는 서버의READY나Ctrl+C 종료 메시지가 없으므로 서버 종료는 별도 확인해야 합니다. 저장된 전체 JSON을 직접 읽은 결과나 장시간 안정성 검사로 확대하지 않습니다.

이번 보고서는 관찰 시점의 작은 요청 묶음입니다. 모니터링 서비스·알림·대시보드·연속 자원 수집·운영 SLO·최대 처리 용량 검증·권한 통제·실제 승인 배포·모델 개선을 대신하지 않습니다. 레지스트리와 파일을 다른 작성자가 실행 중 함께 바꾸지 않는 로컬 실습 범위입니다.

## 8. 복습과 다음 묶음

- [ ] 동시 클라이언트와 단일 요청 서버를 구분한다.
- [ ] 의도한400과 예상하지 못한 실패를 분리해 집계한다.
- [ ] 요청 지연 합과 단계 wall time을 분모로 한 속도를 구분한다.
- [ ] 생성 구간 속도·HTTP 지연·정답 품질을 구분한다.
- [ ] CUDA allocator peak와 전체 GPU 사용률을 구분한다.
- [ ] 보고서 저장과 진단 서버 종료를 별도로 확인한다.

다음은 **6/6 Docker 실행환경 패키징**입니다. 이번 단계에서는 아직 Dockerfile·이미지·컨테이너를 만들지 않습니다.

[← 이전: 배포·롤백](28-deployment-rollback.md) · [목차](README.md)
