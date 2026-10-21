# 27. 평가 파이프라인: 실행 → 검증 → 다음 단계 판정

[← 이전: 실행 추적](26-run-tracking.md) · [목차](README.md) · [다음: 배포·롤백 →](28-deployment-rollback.md)

## 1. 목표와 범위

기존 등록·선택 모델을 **한 번 새로 평가하고, 저장 결과를 검증한 뒤, 다음 단계로 넘길 수 있는지 판정**합니다. LLMOps 여섯 묶음 중 **3/6 파이프라인**입니다.

26장은 평가 실행과 기록 조회 명령을 따로 사용했습니다. 이 장은 그 추적기를 재사용해 `Evaluate → Verify → Decision`을 한 명령으로 연결합니다. 단계 순서·중단 조건·종료 코드·연결된 결과 파일이 이번 결과물입니다.

이미 등록된 선택 모델을 대상으로 하므로 다시 등록하거나 선택을 바꾸지 않습니다. 모델 학습·서버 시작·배포·롤백은 수행하지 않습니다. 별도 스케줄러·CI 서비스·MLflow·새 의존성을 설치하지 않으며 RAG는 제외합니다.

## 2. 세 단계의 의미

| 단계 | 하는 일 | 실패·미달 처리 |
|---|---|---|
| Evaluate |26장 추적기로 새 평가1회. 기존 등록 모델 검증과25장 채점·새 실행 기록을 재사용 | 실행 오류면 결과 검증·다음 단계 판정을 건너뜀. 품질 미달은 평가 완료이므로 기록 검증은 수행 |
| Verify | 반환값과 저장한 run.json 일치·스키마·quality.json 지문·보관 문항/집계·현재 입력·소스 재검증 | 불일치면 오류로 중단하고 다음 단계 판정을 건너뜀 |
| Decision | 검증이 끝난 결과만 품질 승인/미달로 분기 | 승인: ready_for_next_step/종료0. 미달: halted_quality/종료2 |

**품질 미달은 실행 오류와 다릅니다.** 0/8이어도 결과 기록이 올바르면 Verify는 passed이고 Decision에서 중단합니다. Verify=passed는 모델 품질 합격이라는 뜻이 아닙니다.

`ready_for_next_step`도 실제 배포 완료가 아닙니다. 이 실행기는 후속 명령을 실행하지 않아 모든 경로에서 `followup=skipped`, `deployed=false`입니다. 다음 묶음의 배포 로직과 연결할 수 있는 판정·종료 코드를 제공할 뿐입니다.

## 3. 입력과 생성 파일

기존 모델·레지스트리·성능 보고서·25장 보고서·26장 실행 이력을 보존합니다. 실행할 때마다 새 UUID 폴더를 만듭니다.

```text
llmops_runs_v1/
  <이번에 생성된 run_id>/
    quality.json    새 평가의 문항·점수·환경
    run.json        26장 실행 조건과 평가 보고서 지문
    pipeline.json   이번 파이프라인의 단계·판정·연결 지문
```

pipeline.json에는 같은 run_id, run.json과quality.json의 SHA256/크기, 선택 모델 식별 정보, 파이프라인 자체 소스 지문, 단계 상태와 종료 코드를 연결합니다. 기존 추적기의 여섯 소스 기록은 그대로 두고 일곱 번째인 파이프라인 소스는 별도 필드에 기록합니다.

이전 실행의 기록을 사후에 파이프라인 실행으로 바꾸지 않습니다. 이번에 새로 생성한 실행만 검증하며, 관련 없는 과거 손상 기록 때문에 새 실행을 일괄 중단하지 않습니다. `list`는 여전히 전체 실행 기록의 별도 조회 명령입니다.

## 4. 실행 위치와 준비

Ubuntu VM의 `~/llm-lab/source`, 기존 `.venv`에서 실행합니다.26장 완료 상태의 아래 여섯 파일이 같은 폴더에 있어야 합니다.

- smollm2_track_run.py
- smollm2_quality_gate.py
- smollm2_serve_selected.py
- smollm2_registry.py
- smollm2_api_metrics.py
- smollm2_evaluate.py

기존 `../model_registry_v1`의 선택 모델과 캐시된 원본을 사용합니다. CUDA 환경을 새로 설치하거나 변경하지 않습니다. 새 파일이 이미 있으면 생성 블록을 반복하지 말고 기존 파일과 오류를 먼저 확인합니다.

## 5. 전체 코드와 실행

정본: [01-pipeline.py](source/27-evaluation-pipeline/01-pipeline.py). 새 파일 생성은 exclusive 모드이므로 덮어쓰지 않습니다. 등록·평가·생성 코드를 복제하지 않고 기존 추적기와 검증 함수를 호출합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path
code = r'''"""One new evaluation -> verification -> CLI handoff. Never deploys."""
import importlib.util
import json
from pathlib import Path
import sys
import uuid

here = Path(__file__).resolve().parent
path = here / "smollm2_track_run.py"
if not path.is_file():
    path = here.parent / "26-run-tracking/01-track-run.py"
spec = importlib.util.spec_from_file_location("smollm2_track_run", path)
assert spec is not None and spec.loader is not None
tracker = importlib.util.module_from_spec(spec)
# Ignore stale root AND dependency bytecode without changing existing caches.
cache_settings = sys.pycache_prefix, sys.dont_write_bytecode
try:
    sys.pycache_prefix = str(here / (".uncached-" + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    spec.loader.exec_module(tracker)
finally:
    sys.pycache_prefix, sys.dont_write_bytecode = cache_settings
registry = tracker.registry
LOADED_PIPELINE = registry.fingerprint(Path(__file__))


def run(runs, store, device):
    runs, store = Path(runs), Path(store)
    source = dict(path=str(Path(__file__).resolve()), **registry.fingerprint(Path(__file__)))
    if {key: source[key] for key in ("sha256", "bytes")} != LOADED_PIPELINE:
        raise ValueError("pipeline source changed after import; restart with stable code")
    # Names only: no historical receipt/report is consulted or validated.
    existing = {entry.name for entry in runs.iterdir()} if runs.exists() else set()
    receipt = tracker.run(runs, store, device)
    tracker.retained_hex(receipt["run_id"], 32)
    directory = runs / receipt["run_id"]
    if (receipt["run_id"] in existing or runs.is_symlink() or directory.is_symlink() or
            not directory.is_dir() or directory.resolve().parent != runs.resolve() or
            (directory / "run.json").is_symlink()):
        raise ValueError("receipt must identify this new real run directory")
    result = {"schema": 1, "run_id": directory.name, "decision": "error", "exit_code": 1,
              "stages": {"Evaluate": receipt["status"], "Verify": "error", "Decision": "skipped"},
              "run_file": registry.fingerprint(directory / "run.json"),
              "pipeline_source": source, "followup": "skipped", "deployed": False}
    try:
        if registry.fingerprint(Path(__file__)) != LOADED_PIPELINE:
            raise ValueError("pipeline source changed during evaluation")
        retained = registry.read_json(directory / "run.json")
        if retained != receipt:
            raise ValueError("returned and retained run receipts differ")
        tracker.validate_retained_receipt(retained, directory.name)
        if retained["status"] == "error":
            result["stages"]["Verify"] = "skipped"
            result["error_type"] = retained["error_type"]
        else:
            quality = directory / "quality.json"
            if registry.fingerprint(quality) != retained["quality_file"]:
                raise ValueError("quality fingerprint mismatch")
            report = registry.read_json(quality)
            tracker.validate_retained_report(report, retained)
            # ponytail: one trusted writer; before/after checks are not an atomic snapshot.
            if (tracker.inputs(store) != retained["inputs"] or
                    registry.fingerprint(Path(__file__)) != LOADED_PIPELINE or
                    registry.fingerprint(directory / "run.json") != result["run_file"] or
                    registry.fingerprint(quality) != retained["quality_file"]):
                raise ValueError("evaluation inputs/evidence/source changed before handoff")
            decision, code = ("ready_for_next_step", 0) if retained["approved"] else ("halted_quality", 2)
            result.update(decision=decision, exit_code=code, quality_file=retained["quality_file"],
                          selection=retained["inputs"]["selection"])
            result["stages"].update(Verify="passed", Decision=decision)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    with (directory / "pipeline.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("../llmops_runs_v1"))
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        result = run(args.runs, args.registry, args.device)
        print(json.dumps(result, ensure_ascii=False))
        return result["exit_code"]
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
path = Path('smollm2_pipeline.py')
with path.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(code)
print('CREATED', path.name, path.stat().st_size, 'bytes')
CREATE
{
python smollm2_pipeline.py --runs ../llmops_runs_v1 --registry ../model_registry_v1 --device cuda
result=$?
printf 'PIPELINE_EXIT_CODE=%s\n' "$result"
}
```

## 6. 코드와 결과를 읽는 방법

- 초기화: 저장소 배치 또는 VM의 평평한 파일 배치에서26장 추적기를 불러옵니다. 기존 캐시 대신 소스를 읽도록 임시 캐시 경로·쓰기 금지를 사용하고 인터프리터 설정을 복원합니다.
- `tracker.run(...)`: 새 평가를 정확히 한 번 수행합니다. 이미 있던 실행 ID를 재사용하지 않습니다.
- 검증: 저장한 run.json을 다시 읽어 반환값과 비교하고,26장의 보관 기록 검증을 재사용합니다. 새 평가의 현재 모델/선택/질문/소스가 바뀌었는지도 확인합니다.
- 결정: 저장 결과가 일관된 경우에만 승인 여부로0/2를 선택합니다. 평가·검증·기록 오류는1입니다. 잘못된 CLI 인자도1이므로 품질 미달2와 혼동하지 않습니다.
- 저장: 새 실행 폴더의pipeline.json을 exclusive 모드로 기록합니다. 예외 원인은 stderr에 표시하며 기존 증거를 삭제하지 않습니다. 실행 폴더 생성 전 오류나 파일 쓰기 실패라면pipeline.json이 없거나 불완전할 수 있습니다.

| 결과 | 단계·종료 코드 | 해석 |
|---|---|---|
| 승인 | Evaluate=approved → Verify=passed → Decision=ready_for_next_step / 0 | 다음 단계 허용 판정. 실제 배포는 하지 않음 |
| 품질 미달 | Evaluate=blocked → Verify=passed → Decision=halted_quality / 2 | 평가·기록 검증 완료, 품질 때문에 다음 단계 중단 |
| 평가 오류 | Evaluate=error → Verify=skipped → Decision=skipped / 1 | 오류 기록 보존. 없는 품질 결과를 만들어내지 않음 |
| 검증 오류 | Verify=error → Decision=skipped / 1 | 파일/입력 불일치 등. 다음 단계로 진행하지 않음 |

전체 기록을 나중에 조회하려면 기존26장 명령을 사용합니다. 이는 재평가가 아닙니다. 이 조회는run/quality 연결을 검사하며pipeline.json 전체를 감사하는 전용 조회 기능은 아닙니다.

```bash
python smollm2_track_run.py list --runs ../llmops_runs_v1
```

## 7. 검증 근거와 한계

| 근거 | 확인 범위 |
|---|---|
| 로컬 합성 평가 경계 | 승인0/미달2/평가 오류1·단계 건너뛰기·결과/입력/소스 변경 차단·새 실행만 사용·기존 파일 보존·exclusive 쓰기·평평한 파일 배치 검사 통과. 모델 품질 검증이 아님 |
| 로컬 실제 CPU NF4·미학습 zero-B | 실제8문항 생성0/8. Evaluate=blocked → Verify=passed → Decision=halted_quality, CLI2. 새pipeline/run/quality 연결 및 기존 원본/모델/등록/보고서 보존 검사 통과 |
| 별도 CUDA VM의 파이프라인 실행 | `qlora-v1`·8문항0/8·Evaluate=blocked → Verify=passed → Decision=halted_quality·종료2·followup=skipped/deployed=false·새 실행 ID·결과 지문·셸 복귀 확인 |

실제 CPU 검사의 판정 필드 발췌입니다. 실행 ID·경로·해시는 실행마다 달라질 수 있습니다.

```json
{
  "decision": "halted_quality",
  "exit_code": 2,
  "stages": {
    "Evaluate": "blocked",
    "Verify": "passed",
    "Decision": "halted_quality"
  },
  "followup": "skipped",
  "deployed": false
}
```

이 검사는 미학습 fixture의 실제 추론과 연결 검증이며 학습 후보의 CUDA 품질·합격 모델 배포·운영 안정성 증거가 아닙니다. 기준은 기존에 본 합성 진단8문항 그대로이며 품질 합격을 만들기 위해 기준을 낮추거나 재학습하지 않습니다.

### 별도 CUDA VM 실행 예시

기존 선택 `qlora-v1`을 새로 평가한 실행에서도 위의 `halted_quality`·단계 상태·`followup=skipped`·`deployed=false`를 확인했습니다. 여덟 생성문은26장 CUDA 예시와 같았고, 모델 품질은0/8로 미달이었습니다.

```text
CREATED smollm2_pipeline.py 5047 bytes
BLOCKED 0/8
PIPELINE_EXIT_CODE=2
```

출력 JSON의 파일 지문 발췌입니다. 실행 ID와 경로는 실행마다 달라집니다.

```json
{
  "run_file": {
    "sha256": "57c63bdfc35966eb784370ca230edd32e9143eacdcdab4807103d429610b4e9c",
    "bytes": 5540
  },
  "quality_file": {
    "sha256": "136c94b319936f725e03e7ba786cf680ea6d427b424f7890d4040db7141a0d5f",
    "bytes": 3293
  }
}
```

선택 식별 정보는 기존 `qlora-v1`·manifest `bbeeeacbef0f07944c52a1596d5342feecbb09246d8d25868f1c7ee8fc65cad0`·고정 원본/revision과 일치했습니다. 파이프라인 소스는5,047bytes·SHA256 `7b11a6ed7db2415381619df61fbf518324b75bcfbf5071f2d5d76da0b5252c16`이었습니다.

26장과quality_file 지문이 같아도 실행 ID는 새로 생성됐습니다. 같은 보고서 내용이면 지문이 같을 수 있으며, **결과 내용의 지문과 실행 ID는 다른 개념**입니다. 이번 출력은 새run.json 지문과 단계 결과를 함께 보여줍니다.

`bitsandbytes`의 `_check_is_size` FutureWarning은 있었지만 평가·검증·판정과 셸 복귀는 완료됐습니다. 로그와 출력 JSON을 확인한 예시이며, 원격 파일 직접 읽기·경고 해결·실제 배포·이전 버전 롤백 증거는 아닙니다.

한 명의 신뢰된 로컬 작성자를 전제로 합니다. 전후 지문 확인은 동시 변경에 대한 원자적 스냅샷·잠금·서명이 아닙니다. 지문만으로 코드나 모델을 복원하지 못하고 동일 출력도 보장하지 않습니다. 실행 기록과 승인 판정은 실제 운영 배포·접근 제어와 구분합니다.

개발자 실행 위치는 저장소 루트입니다. 실제 모델 검사는 별도의 검증 환경에서 실행하고 VM 패키지를 바꾸지 않습니다.

```bash
python source/27-evaluation-pipeline/test_pipeline.py
python source/27-evaluation-pipeline/test_pipeline.py --integration
python source/check_guide.py
```

## 8. 복습과 다음 묶음

- [ ] 품질 미달이어도 Verify가passed일 수 있는 이유를 설명한다.
- [ ] 추적기의run.json과 실행기의pipeline.json이 각각 무엇을 연결하는지 설명한다.
- [ ] 현재 새 실행의 검증과 과거 전체 실행 목록 조회를 구분한다.
- [ ] 종료0을 배포 완료로 해석하지 않는다.
- [ ] 오류 때 후속 단계를 건너뛰고 기존 증거를 보존하는 이유를 설명한다.

다음 묶음은 [28장 배포·롤백](28-deployment-rollback.md)입니다. 이 장에서는 승인 판정까지만 연결하며 서버 전환과 이전 버전 복구를 수행하지 않습니다.

[← 이전: 실행 추적](26-run-tracking.md) · [목차](README.md) · [다음: 배포·롤백 →](28-deployment-rollback.md)
