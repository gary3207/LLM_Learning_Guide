# 23 · MLOps 입문: 모델·설정·보고서의 버전 등록과 선택

[← 이전: 서빙 성능](22-serving-benchmark.md) · [목차](README.md)

## 1. 이번에 완성할 결과물

22장까지는 파일을 저장하고 호출·측정했습니다. 이번에는 **기존 QLoRA 폴더와 성능 보고서 확인 → 버전 이름으로 등록 → 파일 무결성 검사 → 사용할 버전의 선택 기록 → 다시 조회**를 한 묶음으로 진행합니다.

여기서 레지스트리는 **한 VM에서 사용하는 작은 메타데이터 목록**입니다. 모델 가중치를 복사하거나 학습하지 않고, 경로·설정·파일별 SHA256·성능 요약을 새 JSON에 묶습니다. MLflow·데이터베이스·클라우드 서비스·새 패키지를 설치하지 않습니다. 실제 서버 배포·자동 모델 선택·품질 합격 판정도 하지 않습니다.

## 2. 버전 등록과 선택은 다르다

| 개념 | 이번 실습에서 하는 일 | 하지 않는 일 |
|---|---|---|
| 모델 산출물 | 기존 QLoRA 어댑터·토크나이저·설정 파일을 참조 | 전체 원본 모델 복사, 새 학습 |
| 버전 | `qlora-v1`이라는 이름과 경로·파일 해시·등록 시각을 새 receipt에 기록 | 모델 가중치 자동 변경, 같은 버전 덮어쓰기 |
| 설정 | `qlora_settings.json` 전체를 해시 목록에 포함하고 원본 모델 ID/revision을 추출 | 전체 학습 데이터·실행 환경의 완전한 재현 |
| 평가 근거 | 기존22장 성능 보고서를 경로·해시·summary로 연결 | 성능 보고서를 정답률 평가로 취급 |
| 선택 | `selected.json`이 등록된 버전을 가리키도록 기록 | 서버 재시작·트래픽 전환·운영 승인 |

**등록은 보관 목록에 이름을 붙이는 것, 선택은 그 목록에서 하나를 지정하는 것**입니다. 이번에는 기존 QLoRA 하나를 명시적으로 선택합니다. 응답 품질이 좋다는 뜻이나 SFT보다 우수하다는 뜻이 아닙니다. 버전 이름의 `v1`은 이 레지스트리 안의 이름이며 학습 횟수·모델 크기를 나타내지 않습니다.

SHA256은 파일 내용으로 계산하는 지문입니다. 같은 파일명이라도 바이트가 바뀌면 검사에서 달라진 값이 나옵니다. `show`와 `select`는 파일 목록·크기·해시를 다시 확인하고 변경·추가·삭제를 거부합니다. 이는 **등록 시점 이후의 바이트 일치 검사**이지, 악의적인 사용자가 파일과 메타데이터를 함께 고칠 수 없도록 하는 전자서명·접근 제어가 아닙니다.

### 성능 보고서를 붙일 때의 한계

기존22장 보고서는 실행 당시의 어댑터 해시를 기록하지 않았습니다. 따라서 지금 선택한 폴더와 보고서를 함께 등록하더라도 **과거 측정이 바로 이 바이트의 모델로 실행됐음을 암호학적으로 소급 증명할 수 없습니다.** 코드의 `report_linkage`에 이 한계를 명시합니다. 등록은 현재 파일들을 함께 참조하는 기록이며 완전한 학습/평가 lineage를 새로 만들어내지 않습니다. 성능 summary는 그대로 첨부할 뿐 재측정·재채점하지 않습니다.

## 3. 실행 위치·필요한 파일·생기는 파일

기존 Ubuntu VM의 `~/llm-lab/source/.venv`에서 실행합니다. GPU는 사용하지 않으며 Python3.12 이상의 표준 라이브러리만 있으면 됩니다. 이전 HTTP 서버가 켜져 있다면 해당 터미널에서 Ctrl+C로 종료합니다. 이 실습은 서버를 호출하지 않으며 종료 보고를 새로운 실행 허가 조건으로 삼지 않습니다.

| 경로 | 취급 |
|---|---|
| `~/llm-lab/smollm2_qlora_v1/` | 읽기 전용 입력. 20장에서 저장한 전용 모델 폴더 |
| `~/llm-lab/serving_benchmark_v1.json` | 읽기 전용 입력. 22장의 실제 성능 보고서 |
| `~/llm-lab/source/smollm2_registry.py` | 이번에 새로 만들 코드 |
| `~/llm-lab/model_registry_v1/versions/qlora-v1.json` | 새 버전 등록 receipt. 기존 이름 덮어쓰기 금지 |
| `~/llm-lab/model_registry_v1/selected.json` | 선택한 버전과 receipt의 SHA256. `select`에서만 교체 |

원본 폴더의 모든 일반 파일을 해시로 기록하므로 `reference.pt`도 존재하면 목록에 포함됩니다. **파일을 바이트로 해시할 뿐 `torch.load()`나 Python 코드 실행은 하지 않습니다.** 현재 SmolLM2 저장 형식에 필요한 어댑터·설정·토크나이저 파일을 검사하고, 어댑터 설정의 원본 모델 ID/revision이 QLoRA 설정과 같은지 확인합니다. 파일 내용이 실제 추론 가능한 모델인지 새로 로딩해 검사하는 단계는 아닙니다.

모델·보고서는 복사하지 않습니다. 원본을 이동·수정·삭제하면 이후 검사가 실패합니다. 레지스트리 안의 절대 경로는 같은 VM에서 사용하기 위한 것이며 백업·다른 VM으로 이동 가능한 배포 패키지가 아닙니다. 등록 중에는 해당 파일을 바꾸는 다른 작업을 함께 실행하지 않습니다.

## 4. 코드 준비·등록·선택·조회

아래 전체 명령을 **VM 터미널 한 곳**에서 실행합니다. 기존 모델·보고서가 있어야 하며, 없다고 재학습하거나 가짜 보고서를 생성하지 않습니다. `set -C`는 기존 코드 덮어쓰기를 막습니다. `&&`로 연결해 앞 단계가 실패하면 다음 단계는 실행하지 않습니다.

소스: [01-model-registry.py](source/23-model-registry/01-model-registry.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
(set -C; cat > smollm2_registry.py <<'PY'
"""Single-user local QLoRA metadata registry. No training, copying weights or deployment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

REQUIRED = {"adapter_model.safetensors", "adapter_config.json", "qlora_settings.json",
            "tokenizer.json", "tokenizer_config.json"}


def fingerprint(path):
    with path.open("rb") as stream:
        return {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(), "bytes": path.stat().st_size}


def model_files(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("model must be a real directory, not a symbolic link")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symbolic links are not supported: " + str(path))
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = fingerprint(path)
    if not REQUIRED <= files.keys():
        raise ValueError("missing QLoRA/tokenizer files: " + str(sorted(REQUIRED - files.keys())))
    return files


def version_path(store, version):
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", version):
        raise ValueError("version must be 1..64 ASCII letters/digits/underscores/hyphens, starting with a letter/digit")
    return store / "versions" / (version + ".json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_locations(store, model, report):
    if model.is_symlink() or report.is_symlink():
        raise ValueError("input symbolic links are not supported")
    model, report = model.resolve(), report.resolve()
    if store.resolve().is_relative_to(model):
        raise ValueError("registry must be outside the model directory")
    if report == (store / "selected.json").resolve():
        raise ValueError("performance report must not be the registry selection output")
    return model, report


def register(store, version, model, report):
    target = version_path(store, version)
    if target.exists():
        raise FileExistsError("version already registered: " + version)
    model, report = input_locations(store, model, report)
    files = model_files(model)
    settings = read_json(model / "qlora_settings.json")
    adapter = read_json(model / "adapter_config.json")
    base, revision = settings["base_model"], settings["revision"]
    if not isinstance(base, str) or not base.strip() or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("base_model and pinned 40-hex revision required")
    if adapter.get("base_model_name_or_path") != base or adapter.get("revision") != revision:
        raise ValueError("adapter and QLoRA base/revision mismatch")
    performance = read_json(report)
    if not isinstance(performance, dict) or not isinstance(performance.get("summary"), dict):
        raise ValueError("performance report must be a JSON object with summary")
    record = {"schema": 1, "version": version, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "base_model": base, "revision": revision, "model_directory": str(model), "model_files": files,
              "report_path": str(report), "report_file": fingerprint(report), "benchmark_summary": performance["summary"],
              "report_linkage": "attached_at_registration; historical model/report pairing not proven"}
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return record


def verify(store, version):
    record = read_json(version_path(store, version))
    if record.get("schema") != 1 or record.get("version") != version:
        raise ValueError("registry schema/version mismatch")
    model, report = input_locations(store, Path(record["model_directory"]), Path(record["report_path"]))
    if model_files(model) != record["model_files"]:
        raise ValueError("model file set/content changed since registration")
    if fingerprint(report) != record["report_file"]:
        raise ValueError("performance report changed since registration")
    return record


def select(store, version):
    verify(store, version)
    choice = {"version": version, "manifest_sha256": fingerprint(version_path(store, version))["sha256"]}
    # ponytail: one local writer; add coordination only when multiple writers are required.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=store, prefix=".selected-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(choice, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, store / "selected.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return choice


def show(store):
    choice = read_json(store / "selected.json")
    path = version_path(store, choice["version"])
    if fingerprint(path)["sha256"] != choice["manifest_sha256"]:
        raise ValueError("selected version manifest changed")
    return verify(store, choice["version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("version")
    registration.add_argument("--model", type=Path, default=Path("../smollm2_qlora_v1"))
    registration.add_argument("--report", type=Path, default=Path("../serving_benchmark_v1.json"))
    selection = commands.add_parser("select")
    selection.add_argument("version")
    commands.add_parser("show")
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.registry, args.version, args.model, args.report)
        print("REGISTERED", result["version"], "model files:", len(result["model_files"]))
    elif args.command == "select":
        result = select(args.registry, args.version)
        print("SELECTED", result["version"])
    else:
        print(json.dumps(show(args.registry), ensure_ascii=False, indent=2, allow_nan=False))
PY
) &&
python smollm2_registry.py register qlora-v1 &&
python smollm2_registry.py select qlora-v1 &&
python smollm2_registry.py show
```

처음에는 아래 형태로 출력된 뒤, 선택 버전의 전체 JSON이 나옵니다. **기대 형식이며 별도로 확인한 실제 VM 실행 예시는6절에 구분합니다.** 모델 파일 개수·절대 경로·등록 시각·해시는 실제 파일에 따라 표시됩니다.

```text
REGISTERED qlora-v1 model files: <실제 파일 개수>
SELECTED qlora-v1
```

조회 JSON에서 다음을 읽습니다.

| 필드 | 의미 |
|---|---|
| `version` | 선택된 등록 이름. 이번에는 `qlora-v1` |
| `registered_at_utc` | 등록 시각. UTC이며 한국 시각 표시와 구분 |
| `base_model`, `revision` | 어댑터를 결합할 고정 원본 ID·revision |
| `model_directory`, `model_files` | 기존 폴더의 절대 경로, 상대 파일명별 SHA256·bytes |
| `report_path`, `report_file` | 실제 성능 보고서 경로와 그 파일의 SHA256·bytes |
| `benchmark_summary` | 보고서에서 읽은 성능 요약. 새 측정값이 아님 |
| `report_linkage` | 등록 시 연결이며 과거 실행의 모델/보고서 짝을 소급 증명하지 않는다는 표시 |

QLoRA 어댑터는 원본 가중치를 전부 포함하지 않습니다. 레지스트리 JSON에도 원본 가중치·캐시·Python 패키지가 들어가지 않습니다. 같은 원본 revision과 양자화 설정이 필요한 이유는20장과 같습니다.

### 재조회와 선택 정보 확인

등록을 반복하지 않고 조회만 하려면 다음을 실행합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python smollm2_registry.py show
```

`show`는 저장된 JSON을 보여주기만 하는 명령이 아닙니다. 선택한 receipt의 해시, 기존 모델 폴더의 파일 목록/해시, 성능 보고서 해시를 다시 검사한 뒤 출력합니다. `selected.json` 자체를 확인하려면 다음과 같이 읽을 수 있습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'PY'
import json
from pathlib import Path
choice = json.loads(Path('../model_registry_v1/selected.json').read_text(encoding='utf-8'))
print(json.dumps(choice, ensure_ascii=False, indent=2))
PY
```

서로 다른 등록 버전 중 하나를 선택할 때도 같은 `select <버전이름>`을 사용합니다. 이번 실습을 위해 가짜 두 번째 모델이나 새 학습 후보는 만들지 않습니다. 같은 파일을 다른 이름으로 등록하는 것만으로 다른 모델을 얻는 것은 아닙니다.

## 5. 코드 흐름과 실패 시 읽는 방법

1. **`register`**: 버전 이름/기존 receipt 확인 → 모델 폴더의 파일 지문 계산 → 설정 ID/revision 대조 → 실제 보고서 JSON의 summary 확인 → 새 receipt 생성. 학습·추론·모델 저장은 없습니다.
2. **`select`**: 버전 receipt와 현재 입력 파일들을 검증 → receipt의 SHA256을 포함한 선택 정보를 임시 파일에 기록 → `os.replace()`로 `selected.json` 교체. 입력이 바뀐 상태에서는 선택 정보를 갱신하지 않습니다.
3. **`show`**: `selected.json`이 가리키는 receipt의 해시와 현재 모델/보고서를 검증 → 등록 정보를 출력. 저장된 summary의 정답률이나 수치를 자동 평가하지 않습니다.

`open('x')`는 같은 버전을 덮어쓰지 못하게 합니다. 선택 포인터만 의도적으로 바꿀 수 있습니다. 선택 파일의 임시 쓰기와 교체는 불완전한 JSON으로 기존 선택을 덮는 위험을 줄이지만, 여러 작성자 간 잠금·이력 저장·디스크 장애까지 포함한 트랜잭션 보장은 아닙니다. 버전 receipt를 쓰는 도중 디스크 오류나 강제 중단이 나면 새 이름에 불완전한 파일이 남을 수 있으므로 오류를 확인합니다. 기존 정상 receipt나 모델을 자동 삭제하지 않습니다.

| 오류 | 의미·대응 |
|---|---|
| `version already registered` | 같은 이름의 등록은 보존됨. 조회는 `show`, 의도적인 새 등록은 새 버전 이름 사용 |
| 모델/성능 보고서 `FileNotFoundError` | 앞 장의 실제 파일 경로를 확인. 자동 재학습·가짜 보고서 생성으로 우회하지 않음 |
| `model file set/content changed` | 등록 뒤 모델 폴더의 파일 추가·삭제·내용 변경. 원인을 확인하고 의도한 변경이면 새 버전으로 등록 |
| `performance report changed` | 등록 뒤 보고서가 바뀜. 기존 기록을 고쳐 맞추지 말고 원인/새 보고서 확인 |
| `performance report must not be the registry selection output` | 입력 보고서 경로가 선택 출력인 `selected.json`과 겹침. 입력을 덮지 않도록 등록·검증 모두 거부 |
| `selected version manifest changed` | 선택 뒤 버전 receipt가 바뀜. 메타데이터를 임의 편집하지 말고 원인 확인 |
| 설정 ID/revision 불일치 | 다른 원본에 대한 어댑터/설정이 섞였을 수 있음. 토큰을 임의로 수정해 통과시키지 않음 |

레지스트리는 모델 폴더 밖에 두고 입력 심볼릭 링크는 지원하지 않습니다. 버전 이름은 영문/숫자로 시작하는1~64자의 영문·숫자·밑줄·하이픈만 받으며 경로 탈출 문자를 거부합니다. 단일 사용자·단일 작성자용이며 권한 관리·다중 작성자·서명·원격 artifact 보관·완전한 학습/평가 lineage는 후속 범위입니다. 파일 수/크기에 비례해 해시를 다시 읽으므로 대규모 저장소용 카탈로그로 설계한 것이 아닙니다.

## 6. 실제 로컬 검증과 아직 확인하지 않은 것

Python 표준 라이브러리로 **합성 파일 fixture**를 만들고 등록·선택·조회·별도 CLI 조회를 실제 실행했습니다. fixture의 가중치 파일은 검사 전용 바이트이며 모델로 로딩하지 않았습니다. 원본 파일/보고서 불변, 기존 버전 덮어쓰기 거부, 잘못된 버전 경로/없는 버전, 모델 폴더 안 레지스트리 거부, 가중치·파일 목록·보고서·선택 receipt 변경 감지를 확인했습니다.
입력 보고서와 선택 출력의 경로 충돌도 등록·선택에서 거부하고, 충돌 거부 뒤 보고서 바이트가 보존되는지 검사했습니다.

아래는 실제 개발자 검사 출력입니다. **실제 학습 모델의 추론·품질·VM 등록 성공을 의미하지 않습니다.**

```text
PASS: register/select/show; source unchanged; duplicate/path/missing-version/input-change/receipt-change rejected
```

개발자 재검사(저장소 루트, ML 패키지 불필요):

```bash
python source/23-model-registry/test_registry.py
python source/check_guide.py
```

검사 소스: [test_registry.py](source/23-model-registry/test_registry.py). 위 합성 파일 검증과 아래 실제 QLoRA 폴더·성능 보고서의 VM 등록은 별도 근거입니다. 이번에는 파일 관리 단계만 실행하며,22장의 서버 종료 여부는 그 단계의 상태와 구분합니다.

### Ubuntu VM의 실제 등록·선택·조회 예시

다음은 실제 VM 출력 중 파일 생성·등록·선택 부분입니다. 이어서 `show`가 전체 등록 JSON을 출력하고 셸 프롬프트로 돌아온 것을 확인했습니다.

```text
CREATED smollm2_registry.py 6449 bytes
REGISTERED qlora-v1 model files: 11
SELECTED qlora-v1
```

| 조회 항목 | 실제 결과 |
|---|---|
| schema / version | 1 / `qlora-v1` |
| 원본 모델 | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| 고정 revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| 등록한 모델 폴더 파일 수 | 11개. 어댑터·설정·토크나이저·reference 등 |
| 어댑터 파일 크기 | 937,056bytes |
| 성능 보고서 파일 크기 | 7,260bytes |
| 연결된 성능 요약 | 측정10건·HTTP 중앙값447.73ms·생성17.93토큰/초 |

모델 파일11개와 보고서의 지문은64자리 소문자16진수 SHA256 형식이며, 성능 summary는22장의 측정 결과와 일치했습니다. `show` 성공은 선택한 receipt의 지문, 현재 모델 폴더의 파일 목록/크기/지문, 현재 보고서의 크기/지문을 코드가 다시 검사하고 통과했다는 뜻입니다. 원격 파일을 별도 도구로 다시 읽은 검사나 새 추론 결과는 아닙니다.

이제 같은 이름의 재등록 대신 `python smollm2_registry.py show`로 조회할 수 있습니다. 기존 모델과 보고서는 원래 위치에 남아 있고 선택 정보만 레지스트리에 기록합니다. 선택한 모델이 자동으로 서버에 배포된 것은 아니며, `report_linkage`의 과거 실행 짝 증명 한계와 기존 정답 생성 실패도 그대로 유지합니다.

## 7. 복습

- [ ] 모델 파일·설정·성능 근거를 하나의 버전 이름으로 찾을 수 있다.
- [ ] 등록은 새 모델 학습이 아니고 선택은 배포가 아님을 설명한다.
- [ ] 경로 참조·해시 검사가 복사본/백업/서명과 다름을 이해한다.
- [ ] 성능 보고서를 붙이는 것과 품질 평가·과거 실행 lineage를 구분한다.
- [ ] 원본을 바꾸지 않고 등록·선택·재조회를 실행한다.

## 공식 참고자료

- [Python3.12 hashlib.file_digest](https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest)
- [Python3.12 pathlib의 파일 열기 모드](https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.open)
- [Python3.12 os.replace](https://docs.python.org/3.12/library/os.html#os.replace)
- [MLflow 모델 레지스트리 개념](https://mlflow.org/docs/latest/ml/model-registry/): 버전·metadata·lineage·alias의 확장 사례. 이번 코드는 MLflow 구현/호환 API가 아니며 MLflow를 설치하지 않습니다.

[← 이전: 서빙 성능](22-serving-benchmark.md) · [목차](README.md)
