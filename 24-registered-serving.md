# 24 · 선택한 등록 버전으로 API를 시작하고 응답에서 확인하기

[← 이전: 버전 등록·선택](23-model-registry.md) · [목차](README.md) · [다음: 품질 게이트 →](25-quality-gate.md)

## 1. 이번에 완성할 결과물

23장은 `qlora-v1`을 등록하고 `selected.json`에 선택 정보를 기록했습니다. 선택만으로 서버가 바뀌지는 않습니다. 이번에는 **선택한 등록 정보 검증 → 해당 모델 복원 → 다시 검증 → API 시작 → health·생성 응답의 버전 대조 → 종료**를 한 묶음으로 연결합니다.

기존22장 측정 서버의 로딩 함수와 HTTP Handler, 23장의 `show()` 검증을 재사용하는 작은 실행 파일을 추가합니다. 모델 경로를 다시 손으로 지정하는 대신 레지스트리가 가리키는 폴더를 읽습니다. 새 학습·어댑터 병합·모델/보고서 저장·레지스트리 수정·새 패키지 설치는 없습니다. 기존 서버의 기본 동작도 바꾸지 않습니다.

**같은 VM 안의 로컬 실습**입니다. `127.0.0.1`에만 바인딩하고 한 번에 요청 하나를 처리합니다. 외부 공개·자동 배포·인증·동시 처리·품질 승인 시스템을 만드는 장이 아닙니다.

## 2. 선택 정보와 실행 중인 모델은 다르다

```text
selected.json → 선택한 receipt(manifest)와 현재 모델·보고서 검사
                                      ↓
                          기존 API 로더로 모델 복원
                                      ↓
                    선택·receipt·현재 입력을 다시 검사
                                      ↓
                   127.0.0.1에서 요청 대기, READY 출력
                                      ↓
          /health와 /generate에 같은 model_version 포함
```

여기서 **manifest는23장에서 만든 `versions/qlora-v1.json` 등록 receipt**입니다. 모델 가중치 자체가 아니라 모델/설정/보고서를 참조하는 JSON입니다. 그 파일의 SHA256이 `manifest_sha256`입니다. 어댑터 파일 한 개의 SHA256이나 전체 원본 모델의 지문과 혼동하지 않습니다.

| 구분 | 의미 |
|---|---|
| 레지스트리의 선택 | 다음 서버 시작 때 읽을 로컬 메타데이터 |
| 시작 시점 스냅샷 | 검증하고 실제로 로딩한 버전의 식별 정보. 이 프로세스에서는 고정 |
| 응답의 `model_version` | 이 서버가 시작할 때 로딩한 모델의 식별 정보 |

서버를 켠 뒤 다른 버전을 `select`해도 **hot reload되지 않습니다.** 실행 중인 서버는 기존 모델과 식별 정보를 유지합니다. 의도적으로 선택을 바꾼 뒤에는 기존 서버를 Ctrl+C로 종료하고 새로 시작해야 합니다. 이 장에서는 동작 확인을 위해 불필요한 새 버전을 등록하거나 선택을 바꾸지 않습니다.

`/health`는 **메모리에 로딩된 시작 스냅샷의 식별 정보**를 알려 줍니다. 요청할 때마다 디스크 파일이나 현재 `selected.json`을 재검증하는 무결성 감시 기능이 아닙니다. 두 응답의 버전 일치는 같은 서버의 식별 정보 확인이지 생성 내용의 정답 보장이 아닙니다.

## 3. 실행 위치·필요한 파일

기존 Ubuntu VM의 `~/llm-lab/source/.venv`를 활성화합니다. PyTorch와 기존 고정 의존성 Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2를 그대로 사용합니다. CUDA를 명시했는데 사용할 수 없으면 CPU로 조용히 전환하지 않습니다. 기존 원본 캐시가 필요하며 모델 로딩은 계속 `local_files_only=True`입니다.

| VM 경로 | 역할·취급 |
|---|---|
| `~/llm-lab/source/smollm2_api_metrics.py` | 22장의 기존 API 로더/Handler. 읽기 전용 모듈 |
| `~/llm-lab/source/smollm2_registry.py` | 23장의 등록 정보 검증 모듈. 읽기 전용 |
| `~/llm-lab/source/smollm2_serve_selected.py` | 이번에 새로 만들 실행 파일 |
| `~/llm-lab/model_registry_v1/selected.json` | 기존 `qlora-v1` 선택 정보. 읽기만 함 |
| `~/llm-lab/model_registry_v1/versions/qlora-v1.json` | 기존 버전 receipt. 읽기만 함 |
| `~/llm-lab/smollm2_qlora_v1/` | receipt가 참조하는 기존 모델·설정·토크나이저 |
| `~/llm-lab/serving_benchmark_v1.json` | receipt가 참조하는 기존 성능 보고서 |

저장소의 장별 경로와 VM의 평평한 파일 배치는 다릅니다. 새 파일은 기존 두 모듈과 **같은 `~/llm-lab/source` 폴더**에 둡니다. 기존 서버·레지스트리 코드를 다시 만들거나 덮어쓰지 않습니다.

22장은 측정 서버를 `smollm2_api_metrics.py`라는 별도 이름으로 만들었습니다. 이번 실행 파일은 **이22장 모듈**을 읽습니다. 21장의 이전 `smollm2_api.py`를 덮어쓰거나 대신 사용하지 않습니다. 필요한 파일이 없으면22장의 원문과 파일 배치를 먼저 확인하고, 임의로 기존 파일을 덮어쓰거나 패키지를 재설치하지 않습니다.

21·22장 서버가 같은 포트에서 실행 중이라면 **그 서버의 터미널에서 Ctrl+C로 종료**한 뒤 시작합니다. 무관한 프로세스를 강제 종료하지 않습니다. 시작 중에는 레지스트리·모델·보고서를 변경하는 다른 작업을 실행하지 않습니다. 단일 로컬 작성자를 전제로 한 실습입니다.

## 4. 터미널 A — 새 실행 파일 생성·서버 시작

소스: [01-serve-selected.py](source/24-registered-serving/01-serve-selected.py). 아래 생성 명령의 본문은 이 정본 소스 전체와 같아야 합니다.


아래 전체 명령은 **VM 터미널 A**에서 실행합니다. `open('x')`는 기존 파일이 있으면 덮어쓰지 않고 실패합니다. 이미 파일이 있다면 삭제하지 말고 정본과 내용을 확인합니다. `&&`는 생성 실패 시 서버 시작을 막습니다. `python -O`는 검사문을 끄므로 사용하지 않습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path

code = r'''"""Serve a verified selected registry version. Local, single-user demo only."""
import argparse
import importlib.util
from pathlib import Path


def dependency(name, relative):
    here = Path(__file__).resolve().parent
    path = here / (name + ".py")
    if not path.is_file():
        path = here.parent / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = dependency("smollm2_api_metrics", "21-api-serving/01-serve-qlora.py")
registry = dependency("smollm2_registry", "23-model-registry/01-model-registry.py")


def load_selected(store, device):
    record = registry.show(store)
    if record["base_model"] != api.MODEL or record["revision"] != api.REVISION:
        raise ValueError("selected base/revision is not supported by this QLoRA loader")
    manifest = registry.version_path(store, record["version"])
    digest = registry.fingerprint(manifest)["sha256"]
    tokenizer, model, count = api.load_model(Path(record["model_directory"]), device)
    # ponytail: trusted single writer; no file/selection changes during startup, not an atomic artifact snapshot.
    if registry.show(store) != record or registry.fingerprint(manifest)["sha256"] != digest:
        raise ValueError("selection changed during model loading; restart with stable inputs")
    identity = {"version": record["version"], "manifest_sha256": digest,
                "base_model": record["base_model"], "revision": record["revision"]}
    return tokenizer, model, count, identity


class Handler(api.Handler):
    def send_json(self, status, data):
        if status == 200:
            data = dict(data, model_version=self.server.identity)
        super().send_json(status, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be from 0 to 65535")
    tokenizer, model, count, identity = load_selected(args.registry, args.device)
    with api.HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
        server.metrics, server.identity = args.metrics, identity
        print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}"
              f" | version={identity['version']}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("서버 종료", flush=True)


if __name__ == "__main__":
    main()
'''
path = Path('smollm2_serve_selected.py')
with path.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(code)
print('CREATED', path.name, path.stat().st_size, 'bytes')
CREATE
python smollm2_serve_selected.py --registry ../model_registry_v1 --device cuda --port 8000
```

아래는 **이미 정본 파일을 생성한 뒤 별도로 시작할 때**의 전체 명령입니다. 생성에 실패했다면 이 시작 명령만 실행하지 말고 원인부터 확인합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python smollm2_serve_selected.py --registry ../model_registry_v1 --device cuda --port 8000
```

`--registry`는 기존 선택 정보가 있는 폴더, `--device`는 계산 장치, `--port`는 같은 VM에서 사용할 포트입니다. 선택 버전과 모델 폴더는 레지스트리에서 읽으므로 모델 경로를 별도 인자로 전달하지 않습니다. 측정이 필요할 때만 같은 명령 끝에 `--metrics`를 추가합니다. 측정 필드의 구간·한계는[22장](22-serving-benchmark.md)과 같으며 이번 버전 대조에 필수는 아닙니다.

준비 메시지는 기존 형식 뒤에 ` | version=qlora-v1`이 붙습니다. 아래는 **기대 형식일 뿐 실제 실행 로그가 아닙니다.** 장치 표기와 NF4 개수는 실행 결과를 읽습니다.

```text
READY http://127.0.0.1:8000 | device=<실제 장치> | NF4=<실제 개수> | version=qlora-v1
```

READY가 나온 뒤에는 **터미널 A를 켜 둡니다.** 셸 프롬프트가 돌아오지 않는 것이 정상입니다. 지금 Ctrl+C를 누르면 다음 클라이언트가 연결할 서버가 없어집니다.

## 5. 터미널 B — health·생성 응답의 버전 대조

**같은 VM의 두 번째 터미널**에서 실행합니다. `127.0.0.1`은 이 명령을 실행하는 컴퓨터 자신이므로 다른 PC에서 호출하지 않습니다. 클라이언트는 Python 표준 라이브러리만 사용하며 curl 설치가 필요 없습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'PY'
import json
import re
from urllib.request import Request, urlopen

base = 'http://127.0.0.1:8000'
with urlopen(base + '/health', timeout=60) as response:
    assert response.status == 200
    health = json.load(response)
print('health:', json.dumps(health, ensure_ascii=False, indent=2))

identity = health['model_version']
assert set(identity) == {'version', 'manifest_sha256', 'base_model', 'revision'}
assert identity['version'] == 'qlora-v1'
assert re.fullmatch(r'[0-9a-f]{64}', identity['manifest_sha256'])
assert identity['base_model'] == 'HuggingFaceTB/SmolLM2-135M-Instruct'
assert identity['revision'] == '12fd25f77366fa6b3b4b768ec3050bf629380bac'

prompt = 'Device gpu status? Reply with one word.'
body = json.dumps({'prompt': prompt, 'max_new_tokens': 8}).encode('utf-8')
request = Request(base + '/generate', data=body,
                  headers={'Content-Type': 'application/json'})
with urlopen(request, timeout=60) as response:
    assert response.status == 200
    generated = json.load(response)
print('generate:', json.dumps(generated, ensure_ascii=False, indent=2))
assert generated['model_version']['version'] == 'qlora-v1'
assert generated['model_version']['manifest_sha256'] == identity['manifest_sha256']
assert generated['model_version'] == identity
print('검사 통과: health와 generate의 시작 버전·manifest 식별 정보 일치')
PY
```

이 명령은 실제 받은 JSON을 그대로 출력합니다. 미리 정한 생성문·토큰 수·해시를 출력 예시처럼 만들지 않습니다. `urlopen`으로 GET을 보내고, `Request`의 `data`와 JSON Content-Type으로 POST를 보냅니다. `json.load`는 응답 본문을 Python 객체로 읽습니다. timeout은 클라이언트의 대기 제한이며 서버의 생성 작업을 취소하는 기능은 아닙니다.

| 응답의 `model_version` 필드 | 확인하는 것 |
|---|---|
| `version` | 이번 실행에서 로딩한 등록 이름 `qlora-v1` |
| `manifest_sha256` | 로딩한 시작 스냅샷의 등록 receipt 지문 |
| `base_model` | 어댑터와 결합한 원본 모델 ID |
| `revision` | 고정 원본 revision |

두 응답 모두 `version == 'qlora-v1'`인지 확인하고, manifest 지문뿐 아니라 식별 객체 전체가 같은지도 검사합니다. 형식만 맞는 해시는 인증서가 아니며 두 응답의 일치만으로 디스크의 현재 상태까지 보장하지 않습니다. 질문의 정답 여부도 이 assert의 대상이 아닙니다. JSON에 나타난 생성문을 그대로 관찰하며 정답 실패를 서버 오류로 바꾸어 해석하지 않습니다.

클라이언트 명령이 끝난 **다음에 터미널 A에서 Ctrl+C**를 누릅니다. `서버 종료`와 셸 프롬프트 복귀를 확인합니다. 이후 선택을 바꾸더라도 새 선택을 적용하려면 다시 서버를 시작합니다.

## 6. 코드 흐름과 실패 시 확인할 것

1. `registry.show()`로 선택 receipt와 현재 모델 파일·보고서를 검증합니다. 선택 정보가 없거나 파일이 등록 뒤 바뀌었다면 서버를 열지 않습니다.
2. 검증된 등록 정보의 모델 폴더를 기존 API `load_model()`에 전달합니다. 설정·원본 revision·NF4·어댑터·추론 전용 조건은 기존 로더가 확인합니다.
3. 모델 로딩 뒤 선택과 입력을 다시 검증합니다. 시작 시 읽은 식별 정보와 달라졌다면 READY/바인딩 전에 중단합니다.
4. 검증된 시작 식별 정보를 서버에 고정하고 기존 Handler를 확장합니다. **성공한 `/health`와 `/generate` JSON**에만 `model_version`을 덧붙입니다. 기존 오류 응답이나 모델의 생성 규칙을 버전 정보로 대체하지 않습니다.
5. 이후 요청은 이미 메모리에 있는 모델을 사용합니다. 요청마다 `show()`·모델 복원·저장·학습을 실행하지 않습니다. 이전 `smollm2_api.py`·`smollm2_api_metrics.py`를 단독 실행할 때의 기본 응답은 그대로입니다.

앞뒤로 검사해도 **검사와 로딩 사이의 작은 틈이 없어지는 것은 아닙니다.** 파일 잠금·원자적 artifact 스냅샷·동시 변경 방지 기능이 아닙니다. 단일 작성자 로컬 실습으로 사용하고 시작 중 선택/모델/보고서를 수정하지 않습니다. 읽는 도중 바꾸었다가 되돌리는 경우까지 보호한다고 주장하지 않습니다.

레지스트리는 **신뢰하는 로컬 메타데이터**입니다. 파일 지문 검사는 전자서명·접근 제어·암호학적 인증이 아닙니다. 누군가 모델과 receipt·선택 정보를 함께 고칠 수 있다면 이 실습만으로 신뢰를 확보할 수 없습니다. 기존 보고서의 과거 실행 모델 짝을 소급 증명하지 못하는23장의 `report_linkage` 한계도 유지합니다. 성능 보고서 연결은 품질 보증이 아닙니다.

| 증상 | 먼저 확인할 것 |
|---|---|
| 모듈 경로의 `FileNotFoundError` | 기존 `smollm2_api_metrics.py`·`smollm2_registry.py`가 새 파일과 같은 VM 폴더에 있는지 확인 |
| 선택 정보/모델/보고서 없음 | 기존23장의 실제 경로·파일을 확인. 재학습·가짜 보고서·무조건 재등록으로 우회하지 않음 |
| manifest/모델/보고서 변경 오류 | 시작 중 다른 작업 유무와 실제 변경 원인 확인. 해시를 수동 편집해 맞추지 않음 |
| CUDA 사용 불가 | 기존 가상환경·GPU 상태 확인. CPU로 자동 전환됐다고 해석하지 않음 |
| 포트 사용 중 | 기존 실습 서버의 터미널 확인 후 정상 종료. 외부 포트 공개로 우회하지 않음 |
| `Connection refused` | A의 파일 생성·시작 오류·READY·포트 순서로 확인. A가 실행 중인지 먼저 확인 |
| 응답에 `model_version` 없음 | 이전 일반 API가 아니라 새 실행 파일을 시작했는지 확인 |
| 선택을 바꿨는데 이전 버전 응답 | hot reload가 아님. 정상 종료 후 재시작해야 새 선택을 읽음 |

## 7. 검증 상태와 개발자 재검사

이 장의 새 경로는 **로컬 CPU 검사와 별도 VM 실행을 구분**합니다. 다음 로컬 검사에서는 실제 NF4 원본과 학습하지 않은 zero-B 어댑터를 사용했습니다. 새 실행 파일의 경로·HTTP 동작 검사이지 기존 학습 후보의 품질이나 CUDA 검증이 아닙니다.

| 근거 | 현재 구분 |
|---|---|
| 로컬 선택형 서빙 검사 | 선택 경로·식별 정보·읽기 전용·시작 전 모델/receipt 변경 거부·로딩 중 보고서 변경 거부 통과. 합성 파일이며 추론 검사와 구분 |
| 로컬 실제 모델·HTTP 통합 검사 | 실제 CPU NF4/미학습 zero-B 어댑터로 health·생성 응답과 직접 생성 일치·식별 정보 고정·잘못된 입력 거부·원본/후보/보고서 보존 통과. VM의 평평한 파일 배치도 검사 |
| 별도 CUDA VM의 health/생성 식별 정보 일치 | `cuda:0`·NF4모듈210·`qlora-v1`·동일 manifest 지문·입력41/새8토큰 응답과 클라이언트 검사 통과. 아래 실행 예시와 로컬 검사를 구분 |
| 별도 VM의 READY 원문·Ctrl+C 종료 | READY 원문은 미제공. health/생성 성공으로 서버 동작은 확인했으며, 종료는 아직 미확인 |
| 품질 개선·동시 변경 안전성·운영 준비 | 이번 검증 대상 아님 |

개발자 재검사 위치는 **저장소 루트**이며 VM의 평평한 파일 배치와 다릅니다. 첫 검사는 선택/검증/시작 경로를 검사하고, 두 번째는 실제 ML 의존성이 있는 환경에서 서버와 HTTP를 확인하기 위한 명령입니다. 합성 메타데이터 검사와 실제 모델 추론을 구분하고, 미학습 fixture를 사용한다면 학습 모델의 품질 결과라고 부르지 않습니다.

```bash
python source/24-registered-serving/test_selected.py
python source/21-api-serving/test_serving.py --registered --integration
python source/check_guide.py
```

고정 의존성 조합은 기존 장과 동일하게 유지합니다. 로컬 CPU 검사 환경을 마련할 때는 기존의 Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2 조합을 재사용하고, VM의 이미 설치된 CUDA 환경을 이 CPU 조합으로 교체하지 않습니다. 로컬 검사 통과가 CUDA 실행·학습 후보 품질·VM 정상 종료의 증거는 아닙니다.

검사 소스: [test_selected.py](source/24-registered-serving/test_selected.py), [test_serving.py](source/21-api-serving/test_serving.py). 기존 API의 실제 CPU HTTP 회귀 검사와 저장소 배치의 선택형 `--metrics` 검사도 통과했습니다. 이 검사에서 서버 프로세스를 종료한 것은 별도 VM에서의 Ctrl+C 정상 종료 증거가 아닙니다.

### 별도 CUDA VM 실행 예시

다음은 기존 등록 QLoRA를 사용한 실제 응답입니다. 로컬 CPU의 미학습 fixture 검사와 별도 근거이며, manifest 지문은 각 환경의 등록 receipt 내용에 따라 달라집니다.

```text
health: {
  "status": "ok",
  "device": "cuda:0",
  "quantized_modules": 210,
  "model_version": {
    "version": "qlora-v1",
    "manifest_sha256": "bbeeeacbef0f07944c52a1596d5342feecbb09246d8d25868f1c7ee8fc65cad0",
    "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac"
  }
}
generate: {
  "text": "Device GPU status:\n\nDevice GPU",
  "input_tokens": 41,
  "generated_tokens": 8,
  "model_version": {
    "version": "qlora-v1",
    "manifest_sha256": "bbeeeacbef0f07944c52a1596d5342feecbb09246d8d25868f1c7ee8fc65cad0",
    "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac"
  }
}
검사 통과: health와 generate의 시작 버전·manifest 식별 정보 일치
```

두 응답의 `model_version` 전체가 같으므로 **실행 중인 서버가 보고하는 시작 버전과 receipt 식별 정보의 일치**를 확인했습니다. 생성은 입력41토큰에서 새8토큰이며, 출력문은 요청한 한 단어 정답이 아닙니다. 버전 연결 성공과 응답 품질은 별개입니다. 이 로그에는 지연·GPU메모리·시작 READY 원문·정상 종료 결과가 없으므로 이를 추가로 검증했다고 해석하지 않습니다. 호출 뒤에는 터미널 A에서 Ctrl+C로 종료합니다.

## 8. 복습

- [ ] 등록 이름·manifest 지문·원본 revision이 각각 무엇을 식별하는지 설명한다.
- [ ] 선택 기록만으로 실행 중인 모델이 바뀌지 않음을 이해한다.
- [ ] READY 뒤 같은 VM의 두 번째 터미널에서 호출하고 마지막에 서버를 종료한다.
- [ ] health·generate의 버전과 manifest가 같은지 실제 JSON으로 확인한다.
- [ ] health의 시작 스냅샷 정보와 실시간 파일 검증을 구분한다.
- [ ] 시작 중 파일을 변경하지 않고 동시 변경 보호·인증·품질 보증으로 확대하지 않는다.

[← 이전: 버전 등록·선택](23-model-registry.md) · [목차](README.md) · [다음: 품질 게이트 →](25-quality-gate.md)
