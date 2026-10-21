# 25 · 품질 기준에 미달하면 서버 시작을 막기

[← 이전: 등록 버전 API](24-registered-serving.md) · [목차](README.md)

## 1. 이번에 완성할 결과물

24장은 선택한 `qlora-v1`을 CUDA에서 호출하고 응답의 버전을 확인했습니다. 그러나 요청한 한 단어 정답 대신 `Device GPU status: ...`가 생성되었습니다. **모델을 정확히 찾아 실행하는 것과, 그 모델을 서비스에 승인하는 것은 다릅니다.**

이번에는 **기존 등록 버전 검증 → 같은 QLoRA 복원 → 고정 질문 생성·채점 → 그룹별 품질 판정 → 새 JSON 저장 → 합격한 경우에만 서버 시작**을 한 실행으로 연결합니다. 재학습하거나 정답을 맞히도록 설정을 반복 조정하지 않습니다.

- 입력: 기존 레지스트리·모델·원본 캐시·19장 채점 코드·22장 API·24장 선택형 로더.
- 새 코드: `~/llm-lab/source/smollm2_quality_gate.py`.
- 새 보고서: `~/llm-lab/qlora_quality_gate_v1.json`. 기존 보고서는 덮어쓰지 않습니다.
- 기준 미달이면 보고서를 남기고 **종료 코드2**로 돌아옵니다. `--serve`를 지정했어도 포트를 열지 않습니다.
- 합격하면 평가한 **동일한 메모리 모델**로 로컬 서버를 시작합니다. 판정 후 다른 모델을 다시 로딩하지 않습니다.

이 장은 LLMOps 후속 여섯 묶음 중 품질 검증·승인을 다룹니다. 추적·파이프라인·배포/롤백·관찰/부하·Docker는 후속 묶음이며 RAG는 제외합니다.

## 2. 품질 게이트의 의미와 이번 기준

게이트는 **이미 정한 조건을 통과해야 다음 작업을 허용하는 분기**입니다. `loss` 감소, 파일 해시 검사, `/health` 정상 응답만으로 응답 품질에 합격했다고 보지 않습니다.

```text
검증한 선택 모델 → 고정 질문에 greedy 생성 → 문자열 채점 → 판정·보고서
                                                      ├─ 미달: BLOCKED, 종료2, 서버 없음
                                                      └─ 합격: --serve일 때만 로컬 서버 시작
```

19장의 문항과 `classify()` 채점을 재사용합니다. **합성 규칙은 `gpu → failed`, `network → healthy`**입니다. 실제 장비 상태를 진단하는 문제가 아닙니다. 19장의 float32 모델 로딩 경로는 사용하지 않습니다. QLoRA 로딩은24장의 NF4 로더를 그대로 사용합니다.

| 그룹 | 문항 수 | 합격 조건 | 해석 |
|---|---:|---|---|
| train | 2 | 두 문항 모두 정답 | 이전 학습 질문 재현 |
| probe | 2 | 두 문항 모두 정답 | 이미 관찰한 표현 변경 질문 |
| diagnostic | 4 | 네 문항 모두 정답 | 같은 합성 규칙의 추가 표현 진단 |

각 그룹의 모든 정답이 필요하며 어느 그룹의 실패도 다른 그룹의 점수로 상쇄하지 않습니다. 빈 결과·누락·중복·바뀐 질문/정답은 합격으로 취급하지 않습니다.

**이 엄격한 기준은 게이트 동작을 배우기 위한 교육용 조건입니다.** 이미 본 작은 진단 문항을 사용하므로 독립 최종 test가 아니며, 모두 맞혀도 일반화·실서비스 적합성·안전성을 승인한 것은 아닙니다. 현재 생성 실패를 본 상태에서 만든 개발용 기준이라는 점도 유지합니다. 합격시키기 위해 질문·정답·기준을 실행 뒤 바꾸지 않습니다.

### 채점과 생성 조건

앞뒤 공백만 제거한 뒤 정답과 대소문자를 구분해 완전히 일치해야 합니다. `failed`는 정답이고 `FAILED`, `failed.`, `The status is failed`는 오답입니다. 정답 단어가 일부 포함되어도 통과하지 않습니다.

생성은 기존 서버와 같은 greedy·beam1·최대 새8토큰·반복 페널티1.0 조건입니다. 모델의 저장된 임의 생성 기본값에 맡기지 않습니다. 이번에는 **실제 생성문 채점만** 사용하며 teacher-forcing loss는 계산하지 않습니다.

## 3. 실행 위치·기존 파일·쓰기 범위

Ubuntu VM의 `~/llm-lab/source`에서 기존 `.venv`를 사용합니다. 새 패키지는 설치하지 않습니다.

| VM 파일·폴더 | 용도 |
|---|---|
| `source/smollm2_quality_gate.py` | 이번 새 실행 파일 |
| `source/smollm2_serve_selected.py` | 24장의 선택 버전 검증·QLoRA 로딩·Handler |
| `source/smollm2_api_metrics.py` | 22장의 실제 NF4 로더·제한된 HTTP API |
| `source/smollm2_registry.py` | 23장의 receipt·모델·보고서 무결성 검사 |
| `source/smollm2_evaluate.py` | 19장의 고정 질문·문자열 채점. main 가드가 있으므로 가져오기로 학습/평가를 시작하지 않음 |
| `model_registry_v1/` | 기존 선택 정보·receipt. 읽기 전용 |
| `smollm2_qlora_v1/` | 기존 QLoRA 어댑터·설정·토크나이저. 읽기 전용 |
| `serving_benchmark_v1.json` | 기존 등록 성능 보고서. 읽기 전용 |
| `qlora_quality_gate_v1.json` | 이번 새 품질 보고서 |

표의 `source/`는 `~/llm-lab/source/`, 나머지는 `~/llm-lab/` 기준입니다. 이전 파일이 없으면 해당 장의 파일 배치를 확인합니다. 모델을 다시 학습하거나 기존 파일을 삭제해 해결하지 않습니다.

새 보고서는 모델 폴더나 레지스트리 안에 두지 않습니다. 기존 등록 성능 보고서와 같은 경로도 사용하지 않습니다. 시작 전 기존 출력이 있으면 모델 로딩보다 먼저 거부하며, 마지막 쓰기도 독점 생성으로 기존 파일을 보호합니다. 쓰기 도중 장애가 나면 새 보고서가 일부만 남을 수 있으므로 파일과 오류를 먼저 확인합니다.

시작·평가 중 선택 정보·모델·성능 보고서를 바꾸지 않습니다. 단일 로컬 작성자 실습이며, 재검사만으로 동시 변경의 원자적 스냅샷을 보장하지 않습니다.

## 4. VM에서 생성·평가·조건부 시작

소스: [01-quality-gate.py](source/25-quality-gate/01-quality-gate.py).


```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path
code = r'''"""Read-only diagnostic quality gate; optionally serve the same approved in-memory model."""
import importlib.util
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

GENERATION = {"max_new_tokens": 8, "do_sample": False, "num_beams": 1,
              "use_cache": True, "repetition_penalty": 1.0}

here = Path(__file__).resolve().parent
path = here / "smollm2_serve_selected.py"
if not path.is_file():
    path = here.parent / "24-registered-serving/01-serve-selected.py"
spec = importlib.util.spec_from_file_location("smollm2_serve_selected", path)
assert spec is not None and spec.loader is not None
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)
scorer = selected.dependency("smollm2_evaluate", "19-model-evaluation/01-evaluate-models.py")


def decide(records):
    if not isinstance(records, list) or len(records) != len(scorer.ROWS):
        raise ValueError("exactly the fixed eight rows are required")
    for row, expected in zip(records, scorer.ROWS):
        if (not isinstance(row, dict) or
                tuple(row.get(key) for key in ("group", "prompt", "answer")) != expected or
                not isinstance(row.get("text"), str)):
            raise ValueError("fixed row identity/order and generated text required")
    groups = {}
    for row in records:
        group = groups.setdefault(row["group"], {"correct": 0, "total": 0})
        group["total"] += 1
        group["correct"] += scorer.classify(row["text"], row["answer"]) == "correct"
    correct = sum(group["correct"] for group in groups.values())
    return {"correct": correct, "total": len(records), "groups": groups}, correct == 8


def evaluate(tokenizer, model):
    import torch
    from transformers import GenerationConfig

    config = GenerationConfig(**GENERATION, bos_token_id=tokenizer.bos_token_id,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    records = []
    with torch.inference_mode():
        for group, prompt, answer in scorer.ROWS:
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True)
            inputs = torch.tensor([prefix], device=model.device)
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
            text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
            records.append(dict(group=group, prompt=prompt, answer=answer, text=text,
                                input_tokens=inputs.shape[1], generated_tokens=generated.shape[1] - inputs.shape[1]))
            print(f"[{group}/{len(records)}] {scorer.classify(text, answer)} | answer={answer} | text={text!r}",
                  flush=True)
    return records


def run(store, device, output, serve=False, port=8000):
    store, output = Path(store), Path(output)
    if device not in ("cpu", "cuda") or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("device must be cpu/cuda and port from 0 to 65535")
    if output.is_symlink() or output.exists():
        raise FileExistsError("quality report overwrite refused: " + str(output))
    record = selected.registry.show(store)
    target = output.resolve()
    if (target.is_relative_to(Path(record["model_directory"]).resolve()) or
            target.is_relative_to(store.resolve()) or target == Path(record["report_path"]).resolve()):
        raise ValueError("quality report must be outside model/registry and not the performance report")
    tokenizer, model, count, identity = selected.load_selected(store, device)
    records = evaluate(tokenizer, model)
    summary, approved = decide(records)
    report = {"schema": 1, "model_version": identity,
              "policy": {"criterion": "all 8 correct; pedagogical strict threshold, not a production criterion",
                         "scoring": "strip-only, case-sensitive exact match",
                         "dataset_sha256": hashlib.sha256(json.dumps(scorer.ROWS, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()},
              "runtime": dict(device=str(model.device), quantized_modules=count,
                              **{name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}),
              "generation": dict(GENERATION, bos_token_id=tokenizer.bos_token_id,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id),
              "rows": records, "summary": summary, "approved": approved, "training_executed": False,
              "scope": "diagnostic DEV examples already seen; synthetic gpu=failed/network=healthy; not independent final test; no teacher-forcing CE; this launcher only"}
    # ponytail: trusted single local writer, not an atomic snapshot or concurrent-mutation guarantee.
    if (selected.registry.show(store) != record or
            identity["version"] != record["version"] or
            selected.registry.fingerprint(selected.registry.version_path(store, record["version"]))["sha256"]
            != identity["manifest_sha256"]):
        raise ValueError("selection/artifacts changed during quality gate; restart with stable inputs")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{'APPROVED' if approved else 'BLOCKED'} {summary['correct']}/{summary['total']}", flush=True)
    if approved and serve:
        with selected.api.HTTPServer(("127.0.0.1", port), selected.Handler) as server:
            server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
            server.identity, server.metrics = dict(identity), False
            print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}"
                  f" | version={identity['version']}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("Server stopped", flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("../qlora_quality_gate_v1.json"))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        raise SystemExit(1 if exc.code else 0) from None
    if not 0 <= args.port <= 65535:
        parser.exit(1, "port must be from 0 to 65535\n")
    result = run(args.registry, args.device, args.output, args.serve, args.port)
    raise SystemExit(0 if result["approved"] else 2)
'''
path = Path('smollm2_quality_gate.py')
with path.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(code)
print('CREATED', path.name, path.stat().st_size, 'bytes')
CREATE
python smollm2_quality_gate.py --registry ../model_registry_v1 --device cuda --output ../qlora_quality_gate_v1.json --serve --port 8000
result=$?
printf 'EXIT_CODE=%s\n' "$result"
```

- `--registry`: 기존 선택 버전이 있는 폴더입니다. 새 버전 등록이나 선택 변경을 하지 않습니다.
- `--output`: 이번 실행의 새 보고서입니다. 이미 있다면 삭제하지 말고 새 이름을 사용합니다.
- `--serve`: 합격한 경우에만 평가한 동일 모델로 서버를 시작합니다. 미달이면 서버를 시작하지 않습니다.
- `--port`: 합격 시 열 포트이며 `127.0.0.1`에서만 사용합니다.
- `result=$?`: 바로 앞 실행의 종료 코드를 보관합니다. `|| true`처럼 오류를 감추지 않고 실제 코드를 읽습니다.

이미 파일을 만든 다음 재실행할 때는 가상환경을 활성화하고 실행 파일을 호출합니다. 보고서를 보존하려면 `--output ../qlora_quality_gate_v2.json`처럼 **새 이름**을 사용합니다. 첫 실행을 성공으로 꾸미기 위한 재평가·튜닝은 필요하지 않습니다.

### 종료 코드 읽기

| 결과 | 뜻 | 다음 행동 |
|---|---|---|
| `BLOCKED`, 종료2 | 평가가 끝났고 기준 미달로 시작을 거부 | 실패 문항·그룹과 새 보고서를 읽음. 정상적인 게이트 결과 |
| 종료0, `--serve` 없음 | 합격하고 보고서를 저장한 평가 전용 실행 | 실제 서버를 시작한 것은 아님 |
| 합격 후 READY | `--serve`가 있어 같은 모델로 서버 시작 | 같은 VM 다른 터미널에서24장 health/생성 호출 후 Ctrl+C |
| 파일·모듈·CUDA·설정 오류 등 | 평가/환경 실패 | 오류 원문을 확인. 품질 미달과 구분 |

현재 모델의 이전 한 단어 생성 실패를 보면 차단될 가능성이 있지만, **이번 여덟 문항의 정답 수와 실제 판정은 실행 출력으로 확인**합니다. 예상 수치를 실행 결과처럼 쓰지 않습니다.

이전에24장 서버가 켜져 있다면 그 터미널에서 Ctrl+C로 종료합니다. 이번 실행이 차단되어도 **이미 실행 중이던 다른 서버를 종료하지는 않습니다.** 예전 서버가 계속 응답하는 것을 게이트가 시작한 서버라고 해석하지 않습니다. 이전 종료를 증명하기 위해 서버를 다시 실행할 필요는 없습니다.

## 5. 보고서와 판정 해석

보고서는 선택 모델의 버전/manifest 식별 정보, 고정 평가 정책·질문, 생성 조건, 실제 생성문, 그룹별 판정과 승인 여부를 연결합니다. 파일 저장 성공 자체가 합격은 아닙니다.

보고서를 읽는 명령은 새 서버가 대기 중이면 같은 VM의 다른 터미널에서, 차단되어 프롬프트로 돌아왔으면 같은 터미널에서 실행할 수 있습니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'PY'
from pathlib import Path
print(Path('../qlora_quality_gate_v1.json').read_text(encoding='utf-8'))
PY
```

이 보고서는 새 평가의 결과입니다. 22장의 과거 성능 보고서가 이 평가와 같은 실행에서 만들어졌다고 소급 증명하지 않으며, 파일 지문은 서명·접근 제어가 아닙니다.

## 6. 코드 흐름과 통제 범위

1. 기존 출력·경로 충돌을 확인하고 선택한 모델·등록 정보를 검증합니다.
2. 24장의 로더로 같은 NF4 원본+어댑터를 추론 전용으로 한 번 복원합니다. 19장에서는 질문과 채점만 가져옵니다.
3. 정답을 생성 입력에 넣지 않고 질문만으로 응답을 생성합니다. 고정된 질문/정답과 비교하고 그룹별로 판정합니다.
4. 평가 후 선택·receipt·모델·성능 보고서를 재검사합니다. 평가 도중 입력이 달라졌다면 승인 보고서를 쓰거나 서버를 열지 않습니다.
5. 새 JSON에 판정을 저장합니다. 미달이면 종료2이고, 합격하면서 `--serve`인 경우에만 같은 메모리 모델을 서버에 넘깁니다. 합격 후 정상 종료 시에는 `Server stopped`가 표시됩니다.

**이 게이트가 통제하는 것은 이번 실행 파일의 서버 시작 경로입니다.** 이전21·22·24장 실행 파일을 직접 호출하는 것을 차단하는 보안 장치가 아닙니다. 기존 서버를 중지하거나 기존 모델을 삭제하지 않습니다. 보고서를 고쳐 합격으로 바꾸는 재사용 경로도 제공하지 않습니다. 운영 환경의 인증·권한·배포 정책 강제는 이번 범위가 아닙니다.

## 7. 검증 상태와 재검사

| 근거 | 상태 |
|---|---|
| 로컬 문자열 판정·시작 분기 검사 | 고정 문항·문자열 채점·누락/중복/변경 거부·출력 충돌·판정 후 입력 변경 거부 통과. 합성 응답과 가로챈 서버 경계로 분기를 검사한 것이며 실제 승인 모델의 HTTP 검사는 아님 |
| 로컬 실제 CPU NF4·미학습 zero-B fixture | 실제 여덟 생성·BLOCKED 0/8·CLI 종료2·READY/바인딩 없음·새 보고서 재계산/덮어쓰기 거부·원본/후보/레지스트리/성능 보고서/소스 불변 통과. 학습 후보의 CUDA 품질과 구분 |
| 별도 VM 기존 QLoRA의 평가·판정·새 보고서 | 아직 미확인 |
| 별도 VM의 합격 후 서버 시작·종료 | 실제 합격 경로를 실행했을 때만 확인. 미달 시 시작하지 않는 것이 목표 |

개발자 재검사는 저장소 루트에서 실행합니다. 첫 검사는 표준 라이브러리 기반 판정/분기 검사, 두 번째는 실제 ML 의존성을 갖춘 CPU 통합 검사입니다. 합성 PASS 테스트와 실제 모델 추론의 PASS를 혼동하지 않습니다.

```bash
python source/25-quality-gate/test_gate.py
python source/25-quality-gate/test_gate.py --integration
python source/check_guide.py
```

실제 모델 검사는 기존 Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2의 분리된 로컬 환경을 사용합니다. VM의 CUDA 환경을 CPU 패키지로 교체하지 않습니다. 체크포인트 학습·optimizer 갱신 없이 zero-B 어댑터로 로딩·생성·차단 경로를 검사하며, 그 점수를 학습 후보의 CUDA 품질 결과로 쓰지 않습니다.

## 8. 복습과 다음 묶음

- [ ] 파일 무결성·서버 생존·응답 품질이 서로 다른 검사임을 설명한다.
- [ ] 완전 일치 채점과 그룹별 승인 조건을 구분한다.
- [ ] 품질 미달에 따른 종료2가 환경 오류나 실행 실패와 다름을 이해한다.
- [ ] 기준 미달이면 보고서는 생겨도 새 서버가 시작하지 않음을 확인한다.
- [ ] 기존 진단 문항의 합격을 독립 최종 평가나 운영 승인으로 과장하지 않는다.
- [ ] 이 게이트는 새 실행 경로의 분기이며 기존 서버나 다른 진입점을 막지 않음을 설명한다.

다음 묶음은 **실험 추적·재현성**입니다. 현재 구현하지 않은 후속 기능을 이 장의 검증 결과에 포함하지 않습니다.

[← 이전: 등록 버전 API](24-registered-serving.md) · [목차](README.md)
