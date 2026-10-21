# 30. Docker로 실행환경 패키징

[← 운영 관찰·부하 검사](29-operations-load.md) · [전체 로드맵](00-roadmap.md) · [이어가기](07-resume.md)

## 목표와 현재 범위

기존 등록 모델 API를 **코드와 라이브러리가 들어 있는 이미지**로 만들고, 기존 모델을 컨테이너에서 불러온다. 재학습하거나 레지스트리를 다시 만들지 않는다.

| 단계 | 확인할 내용 |
|---|---|
| 1/4 환경 준비 | Docker 실행, NVIDIA runtime 등록, 작은 컨테이너에서 GPU 확인 |
| 2/4 이미지 구성·빌드 | 좁은 빌드 context, 패키지 설치, 이미지 안의 PyTorch CUDA 확인 |
| 3/4 모델 실행 | 기존 저장물을 읽기 전용으로 연결하고 API 버전·생성·종료 확인 |
| 4/4 결과 정리 | 실제 출력과 검증 한계를 기록하고 남은 학습으로 연결 |

**25~28장의 품질 결과는 여전히 `BLOCKED 0/8`이다.** 이번에는 24장의 수동 진단 API를 패키징한다. 컨테이너 실행 성공은 품질 승인이나 28장의 승인 배포·롤백 성공이 아니다.

## 1. 환경 준비 결과

Linux GPU VM에 Docker Engine과 NVIDIA Container Toolkit이 준비돼 있어야 한다. 설치가 필요한 경우 공식 [Docker Ubuntu 설치](https://docs.docker.com/engine/install/ubuntu/)와 [NVIDIA Toolkit 설치·runtime 구성](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)을 따른다. 설치는 시스템 서비스·네트워크 설정을 변경하므로 기존 환경에 무작정 반복하지 않는다. 기존 GPU 드라이버나 학습용 `.venv`를 교체하는 단계가 아니다.

다음은 준비 단계에서 확인된 **실행 예시**다. 최신 버전 권장표가 아니며, 환경에 따라 버전은 다를 수 있다.

```text
Docker Client: 29.9.2
Docker Server: 29.9.2
NVIDIA runtime: nvidia-container-runtime
NVIDIA Container Toolkit CLI version 1.20.1
ubuntu:24.04 컨테이너 내부: NVIDIA A10, 595.91.07
DOCKER_SETUP_EXIT_CODE=0
```

이 결과는 Docker와 GPU 접근 경로를 확인한 것이다. 아직 모델 실행 이미지를 빌드하거나 컨테이너 안에서 PyTorch·NF4 추론을 확인한 것은 아니다. 준비를 완료했다면 아래 단계부터 진행한다.

## 2. 이미지에 넣는 것과 넣지 않는 것

| 이미지에 포함 | 호스트에 남기고 실행 때 연결 |
|---|---|
| Python 3.12 계열 기반 이미지 | 기존 `model_registry_v1`과 등록 모델·평가 파일 |
| CUDA용 PyTorch와 추론 라이브러리 | Hugging Face Hub 모델 캐시 |
| 21·23·24장의 기존 코드 세 파일 | 학습용 `.venv`, 다른 체크포인트·보고서 |

기존 VM에서 확인한 주요 패키지 버전을 그대로 지정한다.

```text
torch==2.11.0+cu128
transformers==4.57.1
peft==0.17.1
bitsandbytes==0.48.2
accelerate==1.15.0
huggingface_hub==0.36.2
```

Python 기반 이미지는 digest로 고정한다. 다만 **호스트의 Python 3.12.3 및 모든 하위 의존성까지 비트 단위로 복제한다는 뜻은 아니다.** 실제 Python 패치 버전은 아래 실행 결과로 확인하고, 설치된 패키지 목록은 이미지의 `/opt/llm/image-packages.txt`에 보관한다. 같은 이미지 ID를 재사용하면 빌드된 결과를 재사용하는 것이지만, 모든 하위 의존성·APT 패키지까지 잠그지 않은 미래의 재빌드는 달라질 수 있다.

소스 정본:
- [Dockerfile](source/30-container-runtime/Dockerfile): 기반 이미지, 설치, 실행 명령
- [requirements.txt](source/30-container-runtime/requirements.txt): 주요 패키지 버전
- [.dockerignore](source/30-container-runtime/.dockerignore): 기본 제외 후 필요한 파일만 허용
- [context 준비 코드](source/30-container-runtime/01-prepare-context.py): 원본을 수정하지 않고 새 폴더에 여섯 파일 복사
- [실행 가능한 검사](source/30-container-runtime/test_container.py): 파일 보존·정본·거부 경로 검사

API 코드는 새로 만들지 않는다. `smollm2_api_metrics.py`는 [21장 정본](source/21-api-serving/01-serve-qlora.py), `smollm2_registry.py`는 [23장 정본](source/23-model-registry/01-model-registry.py), `smollm2_serve_selected.py`는 [24장 정본](source/24-registered-serving/01-serve-selected.py)을 사용한다.

## 3. 레시피와 빌드 context 준비

VM의 `~/llm-lab/source`에서 실행한다. 레시피 네 파일은 고정 커밋에서 내려받아 전부 SHA-256을 확인한 뒤 **새 폴더에만** 저장한다. 실제 API 코드는 이미 VM에 있는 세 파일을 복사한다. 기존 폴더를 덮어쓰거나 모델·캐시를 빌드 context로 보내지 않는다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'RECIPE'
import hashlib
from pathlib import Path
from urllib.request import urlopen

ref = "8a3904fbbe0570a3fbbd59be432b1cd53cccfc57"
proto = "https"
base = f"{proto}://raw.githubusercontent.com/gary3207/LLM_Learning_Guide/{ref}/source/30-container-runtime"
expected = {
    "Dockerfile": "85af25ed83506628d652d27ed91e0ef9b7a8c68ed48d3935a5ec7a753afce2e5",
    "requirements.txt": "91f7507a630ba295b18c960187f23dbf7629eefd8fcb7711f01b55d0682e7a1b",
    ".dockerignore": "845870dad5cf2c346f46832b62b790e363a371acf1f7bb59bfa92f38d45c841c",
    "01-prepare-context.py": "504a7f810d1556dbea0facd63c8aadeb5e7a4d25a637c55456331aacd93d0749",
}
folder = Path("container_recipe_v1")
if folder.exists() or folder.is_symlink():
    raise FileExistsError("기존 레시피 폴더를 덮어쓰지 않습니다.")
files = {}
for name, digest in expected.items():
    with urlopen(base + "/" + name, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(name + ": SHA256 불일치. 저장하거나 실행하지 않습니다.")
    files[name] = data
folder.mkdir()
for name, data in files.items():
    with (folder / name).open("xb") as stream:
        stream.write(data)
print("RECIPE_FILES", len(files))
RECIPE
```

이어서 context를 준비한다.

```bash
cd ~/llm-lab/source &&
python container_recipe_v1/01-prepare-context.py \
    --source . --output container_context_v1
```

출력의 `FILES 6`은 레시피 세 파일과 API 코드 세 파일이다. 이어지는 SHA-256은 실제 복사된 API 코드의 식별자다. 원본 코드를 실행하거나 모델을 로드하지 않는다. 원본 누락·잘못된 Python 문법·심볼릭 링크·기존 출력 폴더는 덮어쓰는 대신 오류로 중단한다. 폴더를 재귀 복사하지 않으므로 `.venv`나 모델 저장물은 포함되지 않는다.

## 4. 이미지 빌드와 PyTorch CUDA 확인

처음 빌드할 때 CUDA 라이브러리를 내려받으므로 시간과 디스크 공간이 필요하다. 모델 가중치를 새로 내려받는 단계는 아니다. 다음 명령은 컨테이너 내부의 라이브러리를 설치할 뿐 호스트 `.venv`에 `pip install`하지 않는다.

```bash
cd ~/llm-lab/source &&
bash <<'BUILD'
set -euo pipefail
image=smollm2-runtime:lab-v1
if sudo -n docker image inspect "$image" >/dev/null 2>&1; then
    printf '이미 존재하는 이미지 태그를 덮어쓰지 않습니다.\n' >&2
    exit 1
fi
sudo -n docker build --platform linux/amd64 \
    --tag "$image" --iidfile container_context_v1/image-id.txt \
    container_context_v1
sudo -n docker image inspect "$image" --format 'IMAGE_ID={{.Id}}'
BUILD
result=$?
printf 'IMAGE_BUILD_EXIT_CODE=%s\n' "$result"
```

`IMAGE_BUILD_EXIT_CODE=0` 이후에만 다음을 실행한다. 실제 이미지의 주요 패키지와 CUDA tensor를 확인하는 검사이며 **모델 추론 검사는 아직 아니다.** 이 검사에는 호스트 모델·캐시를 연결하지 않는다.

```bash
sudo -n docker run --rm -i --runtime=nvidia --gpus all \
    --user "$(id -u):$(id -g)" --entrypoint python \
    smollm2-runtime:lab-v1 - <<'IMAGE_CHECK'
import json
import platform
from importlib import metadata
import torch

expected = {
    "torch": "2.11.0+cu128", "transformers": "4.57.1", "peft": "0.17.1",
    "bitsandbytes": "0.48.2", "accelerate": "1.15.0", "huggingface_hub": "0.36.2",
}
actual = {name: metadata.version(name) for name in expected}
print("IMAGE_ENV", json.dumps({"python": platform.python_version(), "packages": actual,
      "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}))
assert actual == expected, "주요 패키지 버전이 다릅니다."
assert torch.cuda.is_available(), "컨테이너 PyTorch에서 CUDA를 사용할 수 없습니다."
tensor = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("CUDA_TENSOR", tensor.device, tensor.item(), torch.cuda.get_device_name(0))
IMAGE_CHECK
result=$?
printf 'IMAGE_CHECK_EXIT_CODE=%s\n' "$result"
```

오류가 나면 해당 오류와 종료 코드를 먼저 확인한다. 실패한 빌드의 context를 지우거나 준비 명령으로 덮어쓰지 않는다. context는 남겨 두고 원인을 확인한 뒤 빌드 명령만 다시 실행할 수 있다. 이미 태그가 만들어졌다면 임의로 제거·재지정하지 않는다.

### 실제 이미지 빌드·CUDA 실행 예시

```text
IMAGE_ID=sha256:ac474c3aa23c467f4aebb6e23dc47a040cd0e52c94c057625692fda1ef5f48fc
IMAGE_BUILD_EXIT_CODE=0
IMAGE_ENV {"python": "3.12.15", "packages": {"torch": "2.11.0+cu128", "transformers": "4.57.1", "peft": "0.17.1", "bitsandbytes": "0.48.2", "accelerate": "1.15.0", "huggingface_hub": "0.36.2"}, "torch_cuda": "12.8", "cuda_available": true}
CUDA_TENSOR cuda:0 1.0 NVIDIA A10
IMAGE_CHECK_EXIT_CODE=0
```

이 실행 예시에서는 이미지 빌드와 이미지 내부 PyTorch CUDA tensor가 성공했다. 컨테이너 Python은 **3.12.15**로 호스트의 **3.12.3**과 다르며, 지정한 주요 패키지 여섯 개는 일치한다. 다른 환경의 이미지 ID나 패치 버전이 같을 것이라고 가정하지 않는다. 모델·NF4 추론·API·정상 종료는 이 출력만으로 확인되지 않는다.

## 5. 모델 연결과 API 확인

이미지 검사까지 성공한 뒤 진행한다. 같은 VM의 터미널 두 개를 사용하며, 기존24장 서버와29장 클라이언트를 재사용한다. 새 모델 서버·HTTP 클라이언트나 품질 우회를 추가하지 않는다.

- **코드·라이브러리**는 이미지의 `/opt/llm`에서 실행한다. 호스트 `.venv`를 연결하지 않는다.
- **모델·레지스트리·평가 파일**은 `~/llm-lab`을 기존과 같은 절대 경로에 읽기 전용으로 연결한다. 기존 레지스트리의 절대 경로를 수정하지 않기 위해서다.
- **Hub 캐시**는 별도 읽기 전용 mount로 `/cache/hub`에 연결한다. `HF_HUB_OFFLINE=1`과 `TRANSFORMERS_OFFLINE=1`로 기존 캐시를 사용한다. 캐시가 부족하면 중단하며 몰래 온라인 다운로드로 바꾸지 않는다.
- `--user "$(id -u):$(id -g)"`는 호스트 파일 소유권에 맞춘 실제 실행 UID/GID다. 이미지의 `USER=llm` 환경변수는 이름 조회용이며 권한을 부여하지 않는다.
- 기존 API는 `127.0.0.1`에 bind한다. Linux의 `--network host`로 그 동작을 유지하며 `-p` 포트 공개는 사용하지 않는다. 호스트 네트워크를 공유하므로 네트워크 격리라고 주장하지 않는다.
- 새 진단 포트는 `8003`이다. 기존 `8000`·`8002` 서비스를 임의로 종료하지 않는다.

실행 후에는 이번 컨테이너의 READY·이미지 ID·실행 상태·읽기 전용 mount를 확인하고, `/health`와 `/generate`의 전체 모델 식별자가 호스트 선택 기록과 같은지 검사한다. 생성된 글이 정답인지 여부는 품질 검사와 별개다. 사용 후 해당 컨테이너만 종료하고 자동 제거를 확인한다.

### 터미널 A — 모델 서버

빌드가 기록한 이미지 ID를 사용한다. 새 이름이 이미 사용 중이면 Docker가 거부하며 기존 컨테이너를 삭제하거나 재사용하지 않는다. `--rm`은 이번 컨테이너 종료 후 컨테이너를 제거하는 옵션이지 모델 파일이나 이미지를 지우는 명령이 아니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
lab=$(cd .. && pwd -P) &&
cache=$(python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)') &&
image=$(python -c 'from pathlib import Path; print(Path("container_context_v1/image-id.txt").read_text().strip())') &&
sudo -n docker run --rm -it --name smollm2-runtime-v1 \
    --runtime=nvidia --gpus all --network host \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$lab,dst=$lab,readonly" \
    --mount "type=bind,src=$cache,dst=/cache/hub,readonly" \
    "$image" --registry "$lab/model_registry_v1" \
    --device cuda --metrics --port 8003
```

이번 명령의 `READY`가 나온 뒤 A는 그대로 두고 B로 이동한다. **포그라운드 서버가 기다리는 상태는 정상이다.** 로딩 오류·캐시 누락·포트 충돌로 종료되면 먼저 A의 오류를 확인하고 B를 실행하지 않는다. 다른 리스너를 임의로 종료하지 않는다.

### 터미널 B — 이번 컨테이너와 기존 클라이언트 확인

실행 중인 이미지와 이번 컨테이너의 bind 후 READY 로그를 먼저 확인한다. 단순히 같은 버전의 API가 응답한다는 이유만으로 새 컨테이너가 준비됐다고 판단하지 않는다. mount 출력의 `RW`는 두 항목 모두 `false`, 네트워크는 `host`여야 한다.

```bash
bash <<'CHECK'
set -euo pipefail
cd ~/llm-lab/source
source .venv/bin/activate
image=$(python -c 'from pathlib import Path; print(Path("container_context_v1/image-id.txt").read_text().strip())')
state=$(sudo -n docker container inspect smollm2-runtime-v1 --format '{{.State.Running}} {{.Image}}')
if [ "$state" != "true $image" ]; then
    printf '이번 이미지의 컨테이너가 실행 중이 아닙니다: %s\n' "$state" >&2
    exit 1
fi
logs=$(sudo -n docker logs smollm2-runtime-v1 2>&1)
proto=http
ready="READY ${proto}://127.0.0.1:8003 |"
case "$logs" in
    *"$ready"*) ;;
    *) printf '%s\n컨테이너 READY가 아직 없습니다. 터미널 A를 확인하세요.\n' "$logs" >&2; exit 1 ;;
esac
printf 'CONTAINER_STATE %s\n' "$state"
sudo -n docker container inspect smollm2-runtime-v1 \
    --format 'USER={{.Config.User}} NETWORK={{.HostConfig.NetworkMode}} MOUNTS={{json .Mounts}}'
python smollm2_observe_load.py --registry ../model_registry_v1 \
    --port 8003 --requests 2 --workers 2 --timeout 30 \
    --output ../container_load_v1.json
python - <<'REPORT'
import json
from pathlib import Path
report = json.loads(Path("../container_load_v1.json").read_text())
print("CONTAINER_REPORT", json.dumps({
    "ok": report["ok"], "expected_identity": report["expected_identity"],
    "health": report["health_before"]["payload"],
}, ensure_ascii=False))
REPORT
CHECK
result=$?
printf 'CONTAINER_CHECK_EXIT_CODE=%s\n' "$result"
```

기존 클라이언트가 health와 생성 응답의 전체 모델 identity, 예상400 probe, 워밍업, 순차·동시 각2개 요청을 검사하고 새 보고서만 저장한다. 이번 작은 묶음의 목적은 컨테이너 연결 검사다. 29장의 각6개 측정과 직접적인 속도 우열 비교나 품질 승인으로 해석하지 않는다. 오류·기존 보고서 충돌 시 삭제·덮어쓰기하지 말고 해당 출력을 확인한다.

### 검사 후 종료

B가 끝난 뒤 **A에서 Ctrl+C**를 누른다. A의 `서버 종료` 출력과 함께, B에서 아래 명령이 성공하고 대상 행이 없는지 확인한다. 이번 컨테이너만 조회하며 이미지·모델이나 다른 서비스를 삭제하지 않는다.

```bash
sudo -n docker ps -a --filter 'name=^/smollm2-runtime-v1$'
```

READY·보고서 성공과 정상 종료는 각각의 실제 출력으로 확인한다. 아직 위 모델 실행·종료 출력이 제공되지 않았으므로 아래 표에서도 미확인으로 둔다.

## 6. 지금 확인된 것과 미확인인 것

| 항목 | 확인 범위 |
|---|---|
| Docker/NVIDIA runtime 설치 및 GPU visibility | 위 VM 실행 예시로 확인 |
| context·원본 보존·기존 출력 거부 | Windows stdlib 검사 5개 통과·심볼릭 링크 2개 권한상 건너뜀; Linux WSL 7개 모두 통과 |
| 레시피와 패키지 고정·공식 CUDA wheel 존재 | 정본/index 확인과 별도 VM 이미지 내부 주요 여섯 패키지 버전 일치 확인 |
| 이미지 빌드·PyTorch CUDA tensor | 위 VM 실행 예시에서 빌드0·검사0, Python3.12.15·CUDA12.8·cuda:0·A10 확인 |
| NF4 모델의 컨테이너 HTTP 추론·종료 | 아직 실행 출력 대기 |
| 품질 승인·승인 배포·롤백 | 이번 진단 이미지로 수행한 것이 아님 |

Windows/WSL 검사 환경에는 Docker 엔진이 없어, 파일 검사 결과를 실제 Docker 빌드 성공으로 대신하지 않았다.

## 7. 이 묶음 이후 남은 학습

Docker의 실제 모델 실행까지 확인하면 LLMOps 입문의 **품질 → 추적 → 파이프라인 → 배포·복구 경로 → 관찰 → 실행환경 패키징** 연결을 정리할 수 있다. 모든 운영 문제가 해결되거나 학습 모델이 승인됐다는 뜻은 아니다.

그 다음은 다음 순서가 적합하다. 지금 자동으로 재학습을 시작하지 않는다.

1. **독립 평가 데이터**: 학습·개발·최종 평가를 나누고, 이미 본 여덟 문항만으로 일반화를 주장하지 않기.
2. **모델 품질 개선**: 한 단어 답변용 학습 데이터·형식·학습량을 점검하고, 조건을 하나씩 바꿔 원본/SFT/LoRA/QLoRA를 다시 비교하기.
3. **학습 재현·재개**: 추론용 어댑터 복원과 optimizer·scheduler·RNG까지 포함한 학습 재개를 구분하고 검증하기.
4. **작은 모델 구조 확장**: 현재까지 배운 Attention·Transformer를 바탕으로 더 큰 문맥·여러 층·여러 head의 효과를 같은 평가 조건에서 비교하기.

RAG는 이 경로에서 제외한다. Kubernetes·새 서빙 프레임워크·별도 추적 플랫폼을 이번 완료 조건으로 추가하지 않는다.

## 부록: 코드 읽기

아래는 내려받은 정본의 내용이다. **실행은 위의 Bash 명령으로 한다.** Dockerfile이나 `.dockerignore` 내용을 셸 명령처럼 터미널에 붙이지 않는다.

### Dockerfile

```dockerfile
FROM python:3.12-slim-bookworm@sha256:5d9e6adc9abd558bc0ce05f9ab16cef71328916fc9728f6b81e337af86a737e2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/cache/hub \
    HOME=/tmp USER=llm LOGNAME=llm PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/llm
COPY requirements.txt ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128' \
    && python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/llm/image-packages.txt
COPY app/smollm2_serve_selected.py ./
COPY app/smollm2_api_metrics.py ./
COPY app/smollm2_registry.py ./
USER 65532:65532
ENTRYPOINT ["python", "-B", "smollm2_serve_selected.py"]
CMD ["--device", "cuda", "--metrics", "--port", "8003"]
```

`FROM`은 고정한 기반 이미지, `RUN`은 빌드 중 실행할 명령, `COPY`는 context에서 가져올 파일이다. `WORKDIR`은 이미지 안의 작업 위치다. `ENTRYPOINT`는 실행 프로그램이며 `CMD`는 기본 인자다. 실제 실행 때 인자를 추가하면 기본 인자가 교체되므로 필요한 옵션을 모두 전달한다.

### requirements.txt

```text
torch==2.11.0+cu128
transformers==4.57.1
peft==0.17.1
bitsandbytes==0.48.2
accelerate==1.15.0
huggingface_hub==0.36.2
```

### .dockerignore

```text
**
!Dockerfile
!requirements.txt
!app/
!app/smollm2_serve_selected.py
!app/smollm2_api_metrics.py
!app/smollm2_registry.py
```

`**`로 기본 제외하고 `!`로 필요한 항목만 허용한다. 빌드 결과인 `image-id.txt` 등 추가 파일은 포함하지 않는다.

### 01-prepare-context.py

```python
"""Copy only runtime recipes and existing app code into a NEW Docker context."""
import argparse
import ast
import hashlib
from pathlib import Path


def prepare(source, output):
    source, output = Path(source), Path(output)
    recipes = Path(__file__).resolve().parent
    apps = {
        "smollm2_serve_selected.py": "24-registered-serving/01-serve-selected.py",
        "smollm2_api_metrics.py": "21-api-serving/01-serve-qlora.py",
        "smollm2_registry.py": "23-model-registry/01-model-registry.py",
    }
    inputs = {name: recipes / name
              for name in ("Dockerfile", "requirements.txt", ".dockerignore")}
    for name, relative in apps.items():
        flat = source / name
        inputs[f"app/{name}"] = flat if flat.exists() or flat.is_symlink() else source / relative
    files = {}
    for name, path in inputs.items():
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("symbolic links are not supported: " + str(path))
        files[name] = path.read_bytes()
    for name, data in files.items():
        if name.startswith("app/"):
            ast.parse(data, filename=name)
    output.mkdir()
    (output / "app").mkdir()
    for name, data in files.items():
        (output / name).write_bytes(data)
    print("CONTEXT", output.absolute())
    print("FILES", len(files))
    for name, data in files.items():
        if name.startswith("app/"):
            print(hashlib.sha256(data).hexdigest() + "  " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
```

먼저 필요한 바이트를 모두 읽고 Python 문법을 확인한 다음 새 폴더를 만든다. `mkdir()`에 덮어쓰기 옵션을 주지 않아 기존 출력은 거부한다. 이는 파일 보존 검사이지 실제 Docker 빌드나 GPU 실행 검사가 아니다.

## 참고

- [Docker bind mount와 readonly](https://docs.docker.com/engine/storage/bind-mounts/)
- [Docker Linux host networking](https://docs.docker.com/engine/network/drivers/host/)
- [PyTorch 공식 CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/torch/)
