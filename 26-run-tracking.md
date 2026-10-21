# 26 · 평가 실행을 ID로 추적하고 재현 조건 남기기

[← 이전: 품질 게이트](25-quality-gate.md) · [목차](README.md)

## 1. 이번에 완성할 결과물

25장은 기존 QLoRA가 기준 미달일 때 평가를 마치고 보고서를 저장한 뒤 서버 시작을 차단했습니다. 이제 질문은 **어떤 코드·데이터·설정·모델로 그 평가를 실행했는지 나중에 찾을 수 있는가?**입니다.

이번에는 새 평가 한 번을 실행할 때 고유한 `run_id`를 만들고, 실행 시점의 입력 지문·시작/종료·실행환경·평가 보고서를 연결합니다. **기존25장 실행에 코드 지문을 소급해서 붙이는 것이 아닙니다.** 기존 보고서를 보존하고, 추적이 적용된 새 평가를 한 번 실행합니다. 모델 재학습·서버 시작·선택 버전 변경은 없습니다.

```text
run_id 생성 → 실행 전 코드/질문/선택 모델 기록
                          ↓
                기존25장 gate.run으로 새 평가
                          ↓
             실행 후 입력 지문·결과 연결 확인
                          ↓
              quality.json + run.json → 목록 조회
```

LLMOps 여섯 묶음 중 **2/6 실험 추적·재현성**입니다. 파이프라인·배포/롤백·운영 관찰/부하·Docker는 후속 범위이고 RAG는 제외합니다.

## 2. 모델 버전과 실행 ID는 다르다

| 이름 | 무엇을 식별하는가? | 예시의 의미 |
|---|---|---|
| 모델 버전 | 어떤 모델 폴더·설정·보고서를 등록했는가? | 기존 `qlora-v1` |
| manifest SHA256 | 선택한 등록 receipt의 바이트 | 23·24·25장에서 사용한 등록 지문 |
| run_id | 언제 어떤 조건으로 실행한 평가인가? | 매번 새로 생성하는 32자리 UUID hex 문자열 |
| 보고서 SHA256 | 해당 실행이 남긴 평가 보고서의 바이트 | 같은 모델을 여러 번 평가해도 실행별로 보존 |

같은 모델·질문으로 평가를 반복해도 실행 시각과 실행 ID는 다릅니다. 점수가 같다고 실행 두 개를 하나로 합치지 않습니다. `blocked`는 평가가 끝났으나 품질 기준 미달이라는 뜻이며 추적 실패가 아닙니다.

## 3. 기록하는 것과 보장하지 않는 것

| 기록 | 쓰임 |
|---|---|
| 실행 ID·UTC 시작/종료·경과 시간 | 실행 구분과 조사 시작점 |
| 실제 사용한 추적/게이트/선택 로더/API/registry/scorer 파일의 경로·SHA256·크기 | 어떤 코드 바이트를 관찰했는지 식별 |
| 고정 질문·정답의 데이터 지문 | 평가 입력 변경 여부 구분 |
| 모델 버전·receipt 지문·등록 모델 설정 | 모델과 평가 조건 연결 |
| 실제 생성 설정·패키지 버전·장치 | 실행환경 비교에 필요한 조건 |
| 새 평가 보고서 지문·요약·승인 여부·실행 상태 | 실행 기록과 실제 결과 파일 연결 |

추적 전후로 코드·질문·선택 입력이 달라지면 유효한 완료 실행으로 표시하지 않습니다. 등록 모델 폴더·기존 성능 보고서는25장의 검증 경로도 재사용합니다.

의존 모듈을 가져올 때는 기존 `.pyc` 캐시 대신 소스를 읽도록 Python의 캐시 경로를 일시적으로 격리하고, 새 캐시는 쓰지 않습니다. 파일 크기와 수정 시각이 같은 낡은 캐시가 다른 코드를 실행하는 경우도 검사합니다. 기존 캐시를 삭제하지 않으며 모듈 로딩 뒤에는 인터프리터 설정을 복원합니다. 이 초기화는 단일 스레드 CLI를 전제로 합니다.

**재현 조건을 기록하는 것과 비트 단위 동일 출력을 보장하는 것은 다릅니다.** 모델·코드·패키지를 모두 보관하거나 자동 복구하지 않으며, 장치·의존성·연산 조건에 따른 차이가 남습니다. 이 실습의 지문은 서명·인증·접근 제어가 아닙니다. 단일 로컬 작성자 전제이며 시작·평가 중 입력을 수정하지 않습니다.

이전25장 보고서에는 그때 기록되지 않은 코드 지문을 나중에 붙여 과거 실행을 증명하지 않습니다. 이번 새 실행에서만 실행 전후에 관찰한 지문과 새 결과를 연결합니다.

## 4. 실행 위치와 새 파일

기존 Ubuntu VM의 `~/llm-lab/source`에서 `.venv`를 활성화합니다. 기존 ML 패키지를 사용하고 새 설치는 없습니다.

| VM 경로 | 역할 |
|---|---|
| `source/smollm2_track_run.py` | 이번 새 추적 실행 파일 |
| `source/smollm2_quality_gate.py` | 25장의 평가·판정. `serve=False`로 호출 |
| `source/smollm2_serve_selected.py` | 24장의 선택 검증·NF4 로딩 |
| `source/smollm2_api_metrics.py`·`smollm2_registry.py`·`smollm2_evaluate.py` | 기존 API·등록 검사·고정 질문/채점 |
| `model_registry_v1/`·`smollm2_qlora_v1/` | 기존 선택 정보·모델. 읽기 전용 |
| `qlora_quality_gate_v1.json` | 기존25장 보고서. 이번 실행에서 수정하지 않음 |
| `llmops_runs_v1/<run_id>/quality.json` | 이번 새 실행의 평가 보고서 |
| `llmops_runs_v1/<run_id>/run.json` | 이번 새 실행의 추적 기록 |

`source/`는 `~/llm-lab/source/`, 나머지는 `~/llm-lab/` 기준입니다. 기존 모듈들은 같은 VM `source/` 폴더에 둡니다. 실행 폴더는 새 이름으로 독점 생성하며 기존 실행을 덮어쓰지 않습니다. 추적 루트는 모델·레지스트리 밖에 둡니다.

## 5. VM에서 새 추적 실행과 조회

소스: [01-track-run.py](source/26-run-tracking/01-track-run.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path
code = r'''"""New read-only evaluations with local receipts; no historical-run reconstruction."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

here = Path(__file__).resolve().parent
path = here / "smollm2_quality_gate.py"
if not path.is_file():
    path = here.parent / "25-quality-gate/01-quality-gate.py"
spec = importlib.util.spec_from_file_location("smollm2_quality_gate", path)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
# Ignore existing .pyc without deleting inputs or writing a replacement cache.
cache_settings = sys.pycache_prefix, sys.dont_write_bytecode
try:
    sys.pycache_prefix = str(here / (".uncached-" + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    spec.loader.exec_module(gate)
finally:
    sys.pycache_prefix, sys.dont_write_bytecode = cache_settings
registry = gate.selected.registry


def sources():
    modules = {"gate": gate, "selected": gate.selected, "api": gate.selected.api,
               "registry": registry, "scorer": gate.scorer}
    paths = dict(tracker=Path(__file__), **{name: Path(module.__file__) for name, module in modules.items()})
    return {name: dict(path=str(path.resolve()), **registry.fingerprint(path)) for name, path in paths.items()}


def inputs(store):
    record = registry.show(store)
    return {"sources": sources(), "registry_record": record,
            "selection_file": registry.fingerprint(store / "selected.json"),
            "selection": dict(version=record["version"], base_model=record["base_model"],
                revision=record["revision"], manifest_sha256=registry.fingerprint(
                    registry.version_path(store, record["version"]))["sha256"]),
            "dataset_sha256": hashlib.sha256(json.dumps(gate.scorer.ROWS, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            "generation": dict(gate.GENERATION)}


LOADED_SOURCES = sources()


def validate_report(report, before):
    if (not isinstance(report, dict) or type(report.get("schema")) is not int or report["schema"] != 1 or
            type(report.get("approved")) is not bool or report.get("training_executed") is not False or
            report.get("model_version") != before["selection"] or
            report.get("policy", {}).get("dataset_sha256") != before["dataset_sha256"] or
            not isinstance(report.get("runtime"), dict) or not isinstance(report.get("generation"), dict) or
            any(report["generation"].get(key) != value for key, value in before["generation"].items())):
        raise ValueError("quality report schema/identity/dataset/config mismatch")
    summary, approved = gate.decide(report.get("rows"))
    if report.get("summary") != summary or report["approved"] is not approved:
        raise ValueError("quality report decision mismatch")


def run(runs, store, device):
    runs, store = Path(runs), Path(store)
    before = inputs(store)
    if before["sources"] != LOADED_SOURCES:
        raise ValueError("source files changed after import; restart with stable code")
    target, record = runs.resolve(), before["registry_record"]
    performance = Path(record["report_path"]).resolve()
    if (device not in ("cpu", "cuda") or runs.is_symlink() or
            target.is_relative_to(Path(record["model_directory"]).resolve()) or
            target.is_relative_to(store.resolve()) or target == performance or
            target.is_relative_to(performance)):
        raise ValueError("runs must be outside model/registry/performance report; device must be cpu/cuda")
    directory = runs / uuid.uuid4().hex
    directory.mkdir(parents=True)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    receipt = {"schema": 1, "run_id": directory.name, "started_at_utc": started,
               "inputs": before, "requested_device": device,
               "scope": "new evaluation; trusted single local writer; hashes are not backups/signatures or a bit-exact replay guarantee"}
    try:
        report = gate.run(store, device, directory / "quality.json", serve=False)
        # ponytail: before/after checks assume one trusted writer, not an atomic snapshot.
        if inputs(store) != before:
            raise ValueError("evaluation inputs changed during run")
        retained = registry.read_json(directory / "quality.json")
        if retained != report:
            raise ValueError("returned and retained quality reports differ")
        validate_report(retained, before)
        receipt.update(status="approved" if report["approved"] else "blocked", approved=report["approved"],
                       runtime=report["runtime"], generation=report["generation"],
                       quality_file=registry.fingerprint(directory / "quality.json"))
    except Exception as exc:
        receipt.update(status="error", error_type=type(exc).__name__)
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    receipt.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), duration_seconds=time.monotonic() - clock)
    with (directory / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def retained_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty retained string required")
    return value


def retained_hex(value, length=64):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError("invalid retained hexadecimal identity")


def retained_fingerprint(value):
    retained_hex(value["sha256"])
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("invalid retained byte count")


def retained_utc(value):
    stamp = datetime.fromisoformat(retained_text(value))
    offset = stamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("aware UTC timestamp required")


def retained_generation(value):
    import math
    for key in ("max_new_tokens", "num_beams"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("positive integer generation limit required")
    for key in ("do_sample", "use_cache"):
        if type(value[key]) is not bool:
            raise ValueError("boolean generation option required")
    penalty = value["repetition_penalty"]
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("positive finite repetition penalty required")


def validate_retained_receipt(receipt, run_id):
    import math
    import re
    if (type(receipt["schema"]) is not int or receipt["schema"] != 1 or
            receipt["run_id"] != run_id or receipt["status"] not in ("approved", "blocked", "error")):
        raise ValueError("run receipt schema/identity/status mismatch")
    retained_hex(run_id, 32)
    retained_utc(receipt["started_at_utc"])
    retained_utc(receipt["ended_at_utc"])
    duration = receipt["duration_seconds"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid retained duration")
    if receipt["requested_device"] not in ("cpu", "cuda"):
        raise ValueError("invalid requested device")
    before = receipt["inputs"]
    retained_hex(before["dataset_sha256"])
    retained_fingerprint(before["selection_file"])
    selection = before["selection"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", retained_text(selection["version"])):
        raise ValueError("invalid retained version")
    retained_text(selection["base_model"])
    retained_hex(selection["revision"], 40)
    retained_hex(selection["manifest_sha256"])
    for name in ("tracker", "gate", "selected", "api", "registry", "scorer"):
        meta = before["sources"][name]
        retained_text(meta["path"])
        retained_fingerprint(meta)
    model = before["registry_record"]
    if (type(model["schema"]) is not int or model["schema"] != 1 or
            any(model[key] != selection[key] for key in ("version", "base_model", "revision"))):
        raise ValueError("invalid retained registry identity")
    retained_utc(model["registered_at_utc"])
    for key in ("model_directory", "report_path"):
        retained_text(model[key])
    retained_fingerprint(model["report_file"])
    files = model["model_files"]
    if not isinstance(files, dict) or not {"adapter_model.safetensors", "adapter_config.json",
            "qlora_settings.json", "tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise ValueError("missing retained model files")
    for name, meta in files.items():
        name = retained_text(name)
        if "\\" in name or name.startswith("/") or ":" in name or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("invalid retained model filename")
        retained_fingerprint(meta)
    if not isinstance(model["benchmark_summary"], dict):
        raise ValueError("invalid retained benchmark summary")
    retained_generation(before["generation"])
    if receipt["status"] == "error":
        retained_text(receipt["error_type"])
        if any(key in receipt for key in ("approved", "runtime", "generation", "quality_file")):
            raise ValueError("error receipt contains fabricated completion")
    else:
        if (type(receipt["approved"]) is not bool or
                receipt["status"] != ("approved" if receipt["approved"] else "blocked") or "error_type" in receipt):
            raise ValueError("invalid completion status")
        retained_fingerprint(receipt["quality_file"])
        retained_generation(receipt["generation"])
        if (not isinstance(receipt["runtime"], dict) or
                any(receipt["generation"].get(k) != v for k, v in before["generation"].items())):
            raise ValueError("invalid retained runtime/generation")
        device = retained_text(receipt["runtime"]["device"])
        if device != receipt["requested_device"] and not (receipt["requested_device"] == "cuda" and
                re.fullmatch(r"cuda:[0-9]+", device)):
            raise ValueError("retained runtime device mismatch")


def validate_retained_report(report, receipt):
    before = receipt["inputs"]
    retained_generation(report["generation"])
    if (type(report["schema"]) is not int or report["schema"] != 1 or
            type(report["approved"]) is not bool or report["approved"] is not receipt["approved"] or
            report["model_version"] != before["selection"] or report["training_executed"] is not False or
            report["runtime"] != receipt["runtime"] or report["generation"] != receipt["generation"]):
        raise ValueError("retained report/receipt linkage mismatch")
    policy = report["policy"]
    if (policy["dataset_sha256"] != before["dataset_sha256"] or
            policy.get("scoring", "strip-only, case-sensitive exact match") != "strip-only, case-sensitive exact match" or
            policy.get("criterion", "all 8 correct; pedagogical strict threshold, not a production criterion") !=
            "all 8 correct; pedagogical strict threshold, not a production criterion"):
        raise ValueError("invalid retained scoring policy")
    rows = report["rows"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("eight retained rows required")
    identities, groups = [], {}
    for row in rows:
        identity = tuple(retained_text(row[key]) for key in ("group", "prompt", "answer"))
        if not isinstance(row["text"], str) or identity in identities:
            raise ValueError("invalid or duplicate retained row")
        identities.append(identity)
        group = groups.setdefault(identity[0], {"correct": 0, "total": 0})
        group["total"] += 1
        group["correct"] += row["text"].strip() == identity[2]
    digest = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    summary = report["summary"]
    for counts in (summary, *summary["groups"].values()):
        if any(type(counts[key]) is not int for key in ("correct", "total")):
            raise ValueError("integer summary counts required")
    correct = sum(group["correct"] for group in groups.values())
    if (digest != before["dataset_sha256"] or
            summary != {"correct": correct, "total": 8, "groups": groups} or report["approved"] is not (correct == 8)):
        raise ValueError("retained rows/dataset/decision mismatch")


def list_runs(runs):
    runs = Path(runs)
    if not runs.exists():
        return []
    results = []
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir():
            continue
        item = {"run_id": directory.name, "status": "incomplete"}
        try:
            if directory.is_symlink():
                raise ValueError("run directory is a symbolic link")
            manifest = directory / "run.json"
            if manifest.exists():
                receipt = registry.read_json(manifest)
                validate_retained_receipt(receipt, directory.name)
                item = {key: receipt[key] for key in ("run_id", "status", "started_at_utc", "duration_seconds")}
                if receipt["status"] == "error":
                    item["error_type"] = receipt["error_type"]
                else:
                    quality = directory / "quality.json"
                    if registry.fingerprint(quality) != receipt["quality_file"]:
                        raise ValueError("quality fingerprint mismatch")
                    report = registry.read_json(quality)
                    # Historical query uses retained metadata, never current source/dataset/registry.
                    validate_retained_report(report, receipt)
                    item.update(approved=receipt["approved"], summary=report["summary"],
                                model_version=report["model_version"], quality_file=receipt["quality_file"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            item = {"run_id": directory.name, "status": "invalid", "error_type": type(exc).__name__}
        results.append(item)
    return results


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("run", help="execute a NEW evaluation, never import an old report")
    evaluation.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    evaluation.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    listing = commands.add_parser("list", help="query retained receipts and verify report fingerprints")
    listing.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        if args.command == "list":
            result = list_runs(args.runs)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 1 if any(row["status"] in ("invalid", "incomplete") for row in result) else 0
        receipt = run(args.runs, args.registry, args.device)
        print(json.dumps({"run_id": receipt["run_id"], "status": receipt["status"],
                          "run_directory": str((args.runs / receipt["run_id"]).resolve())}, ensure_ascii=False))
        return {"approved": 0, "blocked": 2, "error": 1}[receipt["status"]]
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
path = Path('smollm2_track_run.py')
with path.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(code)
print('CREATED', path.name, path.stat().st_size, 'bytes')
CREATE
python smollm2_track_run.py run --runs ../llmops_runs_v1 --registry ../model_registry_v1 --device cuda
result=$?
printf 'EXIT_CODE=%s\n' "$result"
python smollm2_track_run.py list --runs ../llmops_runs_v1
```

- `run`: 기존 선택 모델을 새로 평가하고 실행 ID·보고서를 남깁니다. 학습하거나 서버를 열지 않습니다.
- `--runs`: 실행별 기록을 추가할 루트입니다. 기존 모델·등록 정보 밖의 기본값을 사용합니다.
- `--registry`: 기존 선택 버전을 읽는 위치입니다. 선택을 바꾸지 않습니다.
- `list`: 보관한 실행을 조회합니다. 모델을 다시 로딩하거나 평가하지 않습니다.
- 종료2는 완료된 평가의 품질 미달이고, 실행/기록 오류는 따로 구분합니다. 목록 조회 성공을 모델 품질 합격으로 해석하지 않습니다.

이미 새 코드를 만들었다면 생성 블록을 다시 실행하지 말고 기존 파일의 내용·오류를 확인합니다. `run`을 다시 호출하면 새로운 실행 ID와 폴더가 생기며 기존25장 보고서나 앞선 실행을 덮어쓰지 않습니다. 이번에는 기록 한 개를 확인하면 충분하며, 같아 보이는 점수를 만들기 위해 반복 실행하지 않습니다.

## 6. 실행 기록을 읽는 방법

출력에 나온 **실제 run_id**로 해당 폴더의 `run.json`과 `quality.json`을 연결해 봅니다. 미리 정한 ID·시각·지문을 성공 예시처럼 만들지 않습니다. `list`는 보관 보고서의 지문을 확인해 결과 파일이 바뀌었는지도 구분합니다.

목록 조회는 지문 일치뿐 아니라 기록 스키마·시각·기간·모델/데이터/코드 지문 형식과 보관 문항·집계의 구조도 검사합니다. 이때 현재 질문 목록을 가져와 과거 점수를 바꾸지 않고, 보관 문항에서 데이터 지문과 요약을 확인합니다. 오류를 기록할 때는 `error_type`을 남기고 원인 메시지는 터미널의 표준 오류에 출력합니다. 오류를 공유할 때는 계정·접속 정보 등 불필요한 내용을 제외합니다.

기록 상태에서 다음을 나눕니다.

| 상태 | 해석 |
|---|---|
| 승인된 평가 | 평가가 끝났고25장 기준을 통과. 추적 완료와 품질 판정을 함께 보되 운영 배포는 아님 |
| 차단된 평가 | 평가가 끝났지만 품질 미달. 실패 결과도 조사 가능한 정상 기록 |
| 오류 | 평가·입력 검증 등 실행 중 예외. 정상 완료로 꾸미지 않음 |
| 불완전한 실행 | 중단 등으로 폴더는 있으나 완료 기록이 없음. 기존 파일을 보존하고 원인 확인 |
| 결과 파일 변경/손상 | 보관 보고서의 지문이나 기록 형식이 맞지 않음. 완료·합격으로 취급하지 않음 |

조회의 종료0은 보관 기록 조회가 정상이라는 뜻입니다. 그 안에 `error` 상태인 과거 실행이 있더라도 그 실행을 성공으로 바꾸지는 않습니다. `invalid`·`incomplete`가 있으면 조회 종료1로 구분합니다.

과거 실행 뒤 현재 소스가 바뀌었다고 그 실행의 점수를 자동으로 삭제하거나 실패로 바꾸지는 않습니다. 기록 당시 코드 지문과 현재 파일이 달라졌다는 사실은 재현 조건 비교 대상이고, **보관한 당시 결과 파일이 손상됐는지와는 다른 문제**입니다.

파일 지문만으로 과거 코드/모델을 복원하지 못합니다. 재현하려면 기록에 대응하는 코드·모델·환경을 별도로 확보해야 합니다. Docker 패키징은 후속 묶음에서 다룹니다.

## 7. 검증 상태와 재검사

| 근거 | 구분 |
|---|---|
| 로컬 합성 경계의 기록·오류·손상·경로 검사 | 한 번의 평가 호출·입력 지문/시각/보고서 연결·경로 충돌/덮어쓰기 거부·오류 기록·손상/불완전 표시·조회 시 현재 소스와 독립 통과. 가로챈 평가 경계이며 실제 품질과 구분 |
| 로컬 실제 CPU NF4·미학습 zero-B 새 평가/실행 기록 | 실제8문항·BLOCKED 0/8·실행 종료2·새 quality/run 보고서 읽기·6소스/데이터/모델/환경 연결·조회 종료0·보고서 변경 시 invalid/조회1 통과. 원본/어댑터/등록/기존 보고서/소스 보존 확인. 학습 후보 CUDA 품질과 구분 |
| 별도 VM의 추적 적용 새 평가·목록 조회 | 아직 미확인 |
| 기존25장 CUDA 결과 | 이미 확인한0/8·종료2이며26장 실행 완료 증거는 아님 |

개발자 실행 위치는 저장소 루트입니다. ML 의존성이 있는 검사는 기존 분리된 CPU 환경을 사용하며 VM의 CUDA 패키지를 바꾸지 않습니다.

```bash
python source/26-run-tracking/test_tracking.py
python source/26-run-tracking/test_tracking.py --integration
python source/check_guide.py
```

## 8. 복습과 다음 묶음

- [ ] 모델 버전과 실행 ID가 서로 다름을 설명한다.
- [ ] 과거 보고서의 사후 등록과 실행 시점 입력·결과 연결을 구분한다.
- [ ] 품질 미달·실행 오류·불완전한 기록을 구분한다.
- [ ] 보관 보고서의 지문과 실행 당시 코드 지문이 각각 무엇을 확인하는지 설명한다.
- [ ] 새 실행이 이전 보고서·모델·레지스트리를 덮어쓰지 않음을 확인한다.
- [ ] 추적 기록을 동일 출력 보장·전체 아티팩트 백업·서명으로 과장하지 않는다.

다음 묶음은 **파이프라인**입니다. 현재 장은 평가 실행을 기록하고 조회하며 자동 등록·배포·롤백은 수행하지 않습니다.

[← 이전: 품질 게이트](25-quality-gate.md) · [목차](README.md)
