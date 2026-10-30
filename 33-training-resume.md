# 33 · 학습 상태를 저장하고 이어서 학습하기

[← 이전: QLoRA 학습량 비교](32-quality-improvement.md) · [목차](README.md) · [학습 점검](07-resume.md) · [다음: 작은 Transformer 구조 비교 →](34-structure-comparison.md)

## 1. 목표: 모델 복원과 학습 재개 구분

32장은 어댑터를 복원해 같은 답변과 logits를 확인했다. 하지만 저장된 어댑터만으로 optimizer의 누적 상태나 학습률 진행 위치·다음 무작위 배치를 복원한 것은 아니다.

이번에는 같은 시작 어댑터에서 다음 두 경로를 실제로 계산한다.

```text
연속 경로: 시작 → 1 → 2 → 3 → 4 → 5 → 6
분할 경로: 시작 → 1 → 2 → 3 → 저장·프로세스 종료
                                    ↓ 새 Python 프로세스
                                  복원 → 4 → 5 → 6
```

두 경로의 **학습 문항 순서·학습률 경로·최종 어댑터·optimizer·scheduler·난수 상태**를 비교한다. 모델 품질 개선이나 test 재평가는 하지 않는다. 같은 조건에서 중간 저장·정상 종료 후 이어하기를 배우는 실습이며 강제 종료·전원 장애·불완전 파일 자동 복구를 검증하는 것은 아니다.

## 2. 무엇을 함께 저장해야 하는가?

| 상태 | 역할 | 이번 코드 |
|---|---|---|
| 어댑터 가중치 | 현재 모델이 배운 파라미터 | HF 어댑터 파일과 `adapter` 텐서 사본 |
| optimizer | 다음 파라미터 갱신에 필요한 누적값 | `AdamW.state_dict()` |
| scheduler | 학습률 조정의 진행 위치 | `StepLR.state_dict()` |
| 갱신 수 | 몇 번 갱신했고 어디서 이어갈지 | `step` |
| 학습 파라미터 순서 | optimizer 상태를 어느 파라미터에 연결할지 | `param_names` |
| 난수 상태 | 다음 무작위 문항 선택을 이어가기 | Torch CPU RNG·CUDA 사용 시 CUDA RNG |
| 데이터·설정·환경 | 두 경로의 비교 조건 확인 | 데이터 사본/지문·장치·패키지·실험 설정 |

이전32장의 SGD(momentum0)는 내부 누적 상태가 단순했다. 이번에는 **AdamW(lr0.001, weight_decay0) + StepLR(2회마다 학습률을 절반으로 조정)**을 사용해 optimizer와 scheduler 복원이 실제로 필요한 예제를 만든다. 32장과 optimizer·학습률·배치 구성이 다르므로 이전 결과와 품질 우열을 비교하지 않는다.

매 갱신마다31장 train8개에서 `torch.randperm()`으로 **서로 다른2개**를 고른다. 다음 갱신에는 다시 고르므로 갱신 사이에는 같은 문항이 나올 수 있다. 이것은 전체 데이터 한 바퀴를 뜻하는 epoch가 아니라 미니배치 갱신6회다. 무작위 문항 선택에 실제로 사용되는 Torch CPU RNG와 CUDA 사용 시 장치 RNG를 저장한다. Python `random`·NumPy·멀티 worker DataLoader는 사용하지 않으므로 그 상태나 미리 가져온 배치 큐까지 복구했다고 주장하지 않는다.

체크포인트는 **optimizer 갱신과 scheduler 갱신을 모두 마친3회 시점**에 저장한다. 갱신 도중의 gradient accumulation 복원은 포함하지 않는다. 모든 다음 갱신은 `zero_grad()`에서 시작한다.

### 복원 순서

1. 같은 구조·원본 revision·NF4 설정으로 저장 어댑터를 불러온다.
2. 같은 파라미터 순서로 optimizer와 scheduler를 만든다.
3. scheduler 상태를 불러오고 optimizer 상태를 불러온다. scheduler 초기화는 optimizer 상태 로딩 전에 마친다.
4. 모델·optimizer 초기화가 모두 끝난 뒤 **다음 무작위 배치 직전에 난수 상태를 복원**한다.

초기화 중에도 난수를 사용할 수 있으므로 실행 첫머리에 seed만 다시 지정하는 것과 중간 RNG 상태 복원은 다르다. 현재 실습은 Torch에서 사용하는 상태에 범위를 한정한다.

## 3. 실행 위치·입력·새 저장 위치

기존 Ubuntu VM의 **`~/llm-lab/source`**, 기존 `.venv`에서 실행한다. 새 패키지·드라이버 설치나 원본 모델 다운로드는 없다.32장 코드의 데이터 검사·응답 labels·NF4 로더·동결 검사·파일 보호를 재사용한다.

필요한 파일:

- `~/llm-lab/smollm2_qlora_v1/`: 기존 QLoRA 어댑터·토크나이저·설정.
- `~/llm-lab/source/evaluation_splits_v1.json`:31장 데이터.
- `improve_qlora.py`, `quality_data_helpers.py`, `quality_eval_helpers.py`, `quality_api_helpers.py`:32장에서 배치한 기존 코드4개.

새 코드: `~/llm-lab/source/resume_qlora.py`

새 결과: **`~/llm-lab/smollm2_resume_v1/`**

```text
smollm2_resume_v1/
  dataset.json
  experiment.json          조건·기준/체크포인트 지문
  reference/               연속6회 어댑터·토크나이저·training_state.pt
  checkpoint/              중간3회 어댑터·토크나이저·training_state.pt
  resumed/                 별도 프로세스에서6회까지 이어간 저장물
  resumed.json             재개 결과 파일 지문
  verify.json              최종 비교 보고서
```

시작점은32장 선택 후보가 아니라 기존 `smollm2_qlora_v1`이다. 그 어댑터에서 **새 AdamW 실험을 시작하고**, 이번 실험의3회 상태를 완전히 저장해 재개한다. 저장하지 않았던 과거 SGD 상태를 되살리는 것은 아니다. 입력 모델·32장 결과·레지스트리·품질 게이트는 변경하지 않는다.

## 4. VM에서 실행

정본: [재개 코드](source/33-training-resume/01-resume-qlora.py) · [작은 실제 텐서 검사](source/33-training-resume/test_resume.py) · [CLI/보호 검사](source/33-training-resume/test_resume_guards.py) · [실제 NF4 통합 검사](source/33-training-resume/test_resume_integration.py)

아래 블록 전체를 실행한다. 새 코드와 출력 폴더는 덮어쓰지 않는다. 기존 파일이 있다는 오류가 나오면 지우지 말고 내용을 확인한다. 다시 실험할 목적이면 새 코드/출력 이름을 사용하며 `--output`을 세 명령 모두 동일하게 지정한다. 실패한 새 저장 폴더는 일부 파일만 남을 수 있고 자동 삭제하지 않는다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
(set -C; cat > resume_qlora.py <<'PY'
"""AdamW/StepLR/Torch RNG 재개: 6회 연속과 3+3회 별도 프로세스를 비교합니다."""
import argparse
import copy
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import runpy

TOTAL, SPLIT, BATCH, SEED = 6, 3, 2, 1234
ATOL, RTOL = 1e-6, 1e-5


def quality():
    here = Path(__file__).resolve().parent
    path = here / "improve_qlora.py"
    if not path.is_file():
        path = here.parent / "32-quality-improvement/01-improve-qlora.py"
    return runpy.run_path(str(path))


def cpu_copy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture(params, optimizer, scheduler, step, trace):
    import torch

    cuda = any(p.device.type == "cuda" for p in params.values())
    return {"step": step, "param_names": list(params), "adapter": cpu_copy(params),
        "optimizer": cpu_copy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if cuda else [],
        "trace": copy.deepcopy(trace)}


def restore(params, optimizer, scheduler, state):
    import torch

    if list(params) != state["param_names"]:
        raise ValueError("학습 파라미터 순서 불일치")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(state["adapter"][name].to(parameter.device))
    # Both objects must already exist; restore LR scheduling before optimizer state.
    scheduler.load_state_dict(state["scheduler"])
    optimizer.load_state_dict(state["optimizer"])
    # Restore after all model/optimizer setup, immediately before the next random batch.
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"]:
        torch.cuda.set_rng_state_all(state["rng_cuda"])


def compare(reference, resumed):
    import torch

    if reference["step"] != resumed["step"] or reference["param_names"] != resumed["param_names"]:
        raise ValueError("최종 갱신 수/파라미터 순서 불일치")
    trace_keys = lambda state: [(r["step"], r["batch"], r["lr"]) for r in state["trace"]]
    if trace_keys(reference) != trace_keys(resumed):
        raise ValueError("학습 문항 순서 또는 학습률 경로 불일치")
    for key in ("adapter", "optimizer", "scheduler"):
        torch.testing.assert_close(reference[key], resumed[key], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close([r["loss"] for r in reference["trace"]],
                               [r["loss"] for r in resumed["trace"]], atol=ATOL, rtol=RTOL)
    if not torch.equal(reference["rng_cpu"], resumed["rng_cpu"]):
        raise ValueError("최종 CPU 난수 상태 불일치")
    if len(reference["rng_cuda"]) != len(resumed["rng_cuda"]) or any(
            not torch.equal(a, b) for a, b in zip(reference["rng_cuda"], resumed["rng_cuda"])):
        raise ValueError("최종 CUDA 난수 상태 불일치")
    maximum = max((reference["adapter"][name] - resumed["adapter"][name]).abs().max().item()
                  for name in reference["adapter"])
    return {"max_adapter_abs_diff": maximum, "atol": ATOL, "rtol": RTOL}


def runtime(device):
    return {"device": device, "packages": {name: version(name) for name in
        ("torch", "transformers", "peft", "bitsandbytes")}, "total": TOTAL, "split": SPLIT,
        "batch": BATCH, "seed": SEED, "optimizer": "AdamW lr0.001 wd0 foreach=False fused=False",
        "scheduler": "StepLR step_size2 gamma0.5"}


def setup(adapter, device, q):
    import torch

    deps = q["helpers"]()
    tokenizer, model, count = deps["api"]["load_model"](adapter, device)
    model.set_adapter("default")
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if (sum(p.numel() for p in params.values()) != 230400 or not all(
            "lora_" in name and (".q_proj." in name or ".v_proj." in name) for name in params)):
        raise ValueError("q/v rank4 어댑터만 학습해야 합니다.")
    optimizer = torch.optim.AdamW(params.values(), lr=0.001, weight_decay=0.0, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    print(f"DEVICE={model.device} NF4={count} TRAINABLE={sum(p.numel() for p in params.values())}", flush=True)
    return tokenizer, model, params, optimizer, scheduler, deps["evaluation"]["example"]


def advance(tokenizer, model, params, optimizer, scheduler, rows, start, end, trace, example, label):
    import torch

    model.train()
    for step in range(start, end):
        batch = torch.randperm(len(rows))[:BATCH].tolist()  # CPU RNG controls real training data order.
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for index in batch:
            row = rows[index]
            ids, labels, _ = example(tokenizer, row["prompt"], row["answer"], model.device)
            loss = model(ids, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("유한하지 않은 학습 loss")
            (loss / BATCH).backward()
            mean_loss += loss.item() / BATCH
        if (any(p.grad is not None for p in model.parameters() if not p.requires_grad)
                or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values())):
            raise ValueError("동결 원본/어댑터 기울기 검사 실패")
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()  # The checkpoint is always taken after a completed update/scheduler step.
        trace.append({"step": step + 1, "batch": batch, "lr": lr, "loss": mean_loss})
        print(f"{label} STEP={step + 1} BATCH={batch} LR={lr:.8f} LOSS={mean_loss:.6f}", flush=True)


def save(folder, tokenizer, model, params, optimizer, scheduler, step, trace, settings, q):
    import torch

    folder.mkdir()
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    q["write_json"](folder / "qlora_settings.json", dict(settings, training={
        "completed_updates": step, "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0,
        "scheduler": "StepLR step_size2 gamma0.5", "batch": BATCH, "seed": SEED}))
    with (folder / "training_state.pt").open("xb") as stream:
        torch.save(capture(params, optimizer, scheduler, step, trace), stream)


def prepare(args, q):
    import torch

    data = q["read_data"](args.data)
    input_data, input_files = args.data.read_bytes(), q["files"](args.adapter)
    tokenizer, model, params, optimizer, scheduler, example = setup(args.adapter, args.device, q)
    frozen = q["frozen_digest"](model)
    settings = json.loads((args.adapter / "qlora_settings.json").read_text(encoding="utf-8"))
    torch.manual_seed(SEED)
    initial = capture(params, optimizer, scheduler, 0, [])
    args.output.mkdir(parents=True, exist_ok=False)
    q["write_json"](args.output / "dataset.json", data)
    trace = []
    advance(tokenizer, model, params, optimizer, scheduler, data["splits"]["train"],
            0, TOTAL, trace, example, "CONTINUOUS")
    if not any(not torch.equal(initial["adapter"][name], p.detach().cpu()) for name, p in params.items()):
        raise ValueError("어댑터가 갱신되지 않았습니다.")
    save(args.output / "reference", tokenizer, model, params, optimizer, scheduler, TOTAL, trace, settings, q)
    restore(params, optimizer, scheduler, initial)
    trace = []
    advance(tokenizer, model, params, optimizer, scheduler, data["splits"]["train"],
            0, SPLIT, trace, example, "PREFIX")
    save(args.output / "checkpoint", tokenizer, model, params, optimizer, scheduler, SPLIT, trace, settings, q)
    if (q["frozen_digest"](model) != frozen or q["files"](args.adapter) != input_files
            or args.data.read_bytes() != input_data):
        raise ValueError("동결된 원본 또는 기존 입력 변경")
    q["write_json"](args.output / "experiment.json", {"contract": runtime(args.device),
        "dataset_sha256": q["files"](args.output)["dataset.json"],
        "checkpoint_files": q["files"](args.output / "checkpoint"),
        "reference_files": q["files"](args.output / "reference"),
        "frozen_base_unchanged": True, "input_unchanged": True})
    print(f"PREPARED reference_updates={TOTAL} checkpoint_updates={SPLIT}", flush=True)


def experiment(args, q):
    metadata = json.loads((args.output / "experiment.json").read_text(encoding="utf-8"))
    if metadata["contract"] != runtime(args.device):
        raise ValueError("학습 장치/패키지/실험 설정 불일치")
    data = q["read_data"](args.output / "dataset.json")
    if q["files"](args.output)["dataset.json"] != metadata["dataset_sha256"]:
        raise ValueError("저장 데이터 변경")
    if q["files"](args.output / "checkpoint") != metadata["checkpoint_files"]:
        raise ValueError("중간 체크포인트 파일 변경")
    if q["files"](args.output / "reference") != metadata["reference_files"]:
        raise ValueError("연속 학습 기준 파일 변경")
    return metadata, data


def resume(args, q):
    import torch

    metadata, data = experiment(args, q)
    folder = args.output / "checkpoint"
    state = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=True)
    if state["step"] != SPLIT or len(state["trace"]) != SPLIT:
        raise ValueError("중간 갱신 위치 불일치")
    tokenizer, model, params, optimizer, scheduler, example = setup(folder, args.device, q)
    torch.testing.assert_close(cpu_copy(params), state["adapter"], atol=0, rtol=0)
    frozen = q["frozen_digest"](model)
    restore(params, optimizer, scheduler, state)
    trace = copy.deepcopy(state["trace"])
    advance(tokenizer, model, params, optimizer, scheduler, data["splits"]["train"],
            state["step"], TOTAL, trace, example, "RESUMED")
    if q["frozen_digest"](model) != frozen:
        raise ValueError("재개 중 동결된 원본 변경")
    settings = json.loads((folder / "qlora_settings.json").read_text(encoding="utf-8"))
    save(args.output / "resumed", tokenizer, model, params, optimizer, scheduler, TOTAL, trace, settings, q)
    experiment(args, q)  # Existing dataset, checkpoint and reference must stay unchanged.
    q["write_json"](args.output / "resumed.json", {"files": q["files"](args.output / "resumed"),
                                                "from_step": SPLIT, "final_step": TOTAL})
    print(f"RESUMED_FROM={SPLIT} FINAL_STEP={TOTAL} | optimizer/scheduler/RNG restored", flush=True)


def verify(args, q):
    import torch

    experiment(args, q)
    before = q["files"](args.output)
    reference = torch.load(args.output / "reference/training_state.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(args.output / "resumed/training_state.pt", map_location="cpu", weights_only=True)
    if reference["step"] != TOTAL or resumed["step"] != TOTAL:
        raise ValueError("최종 갱신 위치 불일치")
    result = compare(reference, resumed)
    saved = json.loads((args.output / "resumed.json").read_text(encoding="utf-8"))
    if q["files"](args.output / "resumed") != saved["files"] or q["files"](args.output) != before:
        raise ValueError("재개 결과 파일 변경")
    report = dict(result, ok=True, total_updates=TOTAL, checkpoint_step=SPLIT, device=args.device,
        checks={key: True for key in ("batch_order", "lr_path", "adapter", "optimizer", "scheduler", "rng")},
        lr_path=[row["lr"] for row in reference["trace"]],
        batch_order=[row["batch"] for row in reference["trace"]], quality_evaluation_performed=False)
    q["write_json"](args.output / "verify.json", report)
    print("VERIFY=PASS (batch order/LR/adapter/optimizer/scheduler/RNG)", flush=True)
    print(f"MAX_ADAPTER_ABS_DIFF={result['max_adapter_abs_diff']:.9g} ATOL={ATOL} RTOL={RTOL}", flush=True)
    print("SAVED verify.json | quality not evaluated; existing models/registry unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "resume", "verify"])
    parser.add_argument("--adapter", type=Path, default=Path("../smollm2_qlora_v1"))
    parser.add_argument("--data", type=Path, default=Path("evaluation_splits_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("../smollm2_resume_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    target = args.output if args.mode == "prepare" else args.output / (
        "resumed" if args.mode == "resume" else "verify.json")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"기존 출력 보존: {target}")
    if args.mode == "prepare" and args.output.resolve().is_relative_to(args.adapter.resolve()):
        raise ValueError("입력 어댑터 안에 출력할 수 없습니다.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode != "verify" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    q = quality()
    {"prepare": prepare, "resume": resume, "verify": verify}[args.mode](args, q)


if __name__ == "__main__":
    main()
PY
) &&
python -B resume_qlora.py prepare --device cuda &&
python -B resume_qlora.py resume --device cuda &&
python -B resume_qlora.py verify --device cuda
```

- **`prepare`**: 기준 경로를6회 계산해 저장한다. 같은 시작 가중치·새 optimizer 초기 상태·RNG로 되돌아가 분할 경로의 앞3회를 계산하고 체크포인트를 저장한 뒤 종료한다. 기준 경로도 실제로 계산하므로 단순히 총6회만 실행하는 명령은 아니다.
- **`resume`**: 새 Python 프로세스에서3회 체크포인트를 불러와4~6회 갱신을 수행하고 새 폴더에 저장한다. 앞3회를 다시 학습하지 않는다.
- **`verify`**: 모델을 다시 학습하지 않고 저장 상태를 CPU로 읽어 두 경로를 비교한다. 중간 체크포인트·기준 경로 파일을 유지하고 새 `verify.json`만 만든다.

셸의 `&&`는 앞 명령이 실패하면 다음 명령을 실행하지 않는다. 각 모드의 출력이 이미 있으면 보호를 위해 거부한다. CUDA가 없을 때 prepare/resume는 CPU로 조용히 전환하지 않는다. verify 자체는 저장 상태 비교이며 GPU 학습을 수행하지 않는다.

## 5. 코드 읽기

- **`quality()`**: 기존32장 코드를 `__main__` 경로 없이 읽는다. 기존 데이터/로더/응답 labels/동결 검사/배타적 JSON 저장을 재사용하며 과거 학습이나 서버를 시작하지 않는다.
- **`capture()`**: 현재 어댑터와 optimizer 텐서를 CPU 사본으로 만들고 scheduler·갱신 수·문항 순서/loss 기록·Torch RNG를 저장한다. 현재 값과 메모리를 공유하지 않으므로 후속 갱신이 과거 스냅샷을 바꾸지 않는다.
- **`restore()`**: 파라미터 순서를 검사한 뒤 어댑터·scheduler·optimizer·RNG 순서로 복원한다. optimizer는 상태 사전을 로딩하며 학습률과 내부 누적값을 함께 되돌린다.
- **`advance()`**: train만 사용한다. 한 갱신은2개 문항 loss의 평균에 대한 역전파·gradient clipping·AdamW 갱신·StepLR 갱신이다. 출력 `LR`은 **그 갱신에 실제 사용한 학습률**, `LOSS`는 해당 배치의 **갱신 전 평균 loss**다. 갱신마다 문항이 다르므로 loss가 매번 낮아져야 하는 실험이 아니다.
- **`save()`**: 별도 새 폴더에 추론용 HF 어댑터와 학습 상태 사전 `training_state.pt`를 함께 저장한다. 학습 상태 사전의 어댑터 사본과 HF 어댑터가 정확히 같은지도 재개 전에 검사한다. 로딩에는 `weights_only=True`를 사용하며, 임의의 외부 체크포인트가 아니라 이 실습이 만든 파일을 사용한다.
- **`experiment()`**: 같은 장치·패키지 버전·데이터·설정과 기준/중간 저장물 지문을 확인한다. 체크섬은 단일 사용자 실습에서 파일 변화를 찾는 도구이며 서명된 보안 provenance나 파일 백업은 아니다.
- **`compare()`**: 문항 순서·학습률 경로와 최종 RNG는 정확히 비교한다. 어댑터·optimizer·scheduler와 단계별 loss는 명시한 `atol=1e-6`, `rtol=1e-5`로 비교한다. `MAX_ADAPTER_ABS_DIFF`는 실제 어댑터 원소의 최대 절대 차이다.

### 재현 범위

두 경로는 같은 장치·패키지·구조·설정으로 실행해야 한다. 스크립트는 deterministic algorithms를 요청하고 TF32를 끄며, 프로세스 안에서 필요한 cuBLAS 설정의 기본값을 지정한다. 지원되지 않는 연산이 있으면 오류를 보존하고 확인한다. **이 설정만으로 모든 외부 커널·모든 장치·다른 버전 사이의 비트 단위 재현을 보장하지 않는다.** 비교 기준은 실제 저장 상태와 명시한 허용오차다.

float32·단일 프로세스·단일 사용자의 작은 실습이다. AMP GradScaler·분산 학습·다중 worker sampler 위치·Python/NumPy 난수·갱신 중간 gradient·전원 장애 후 atomic 복구는 다루지 않는다. 이런 상태를 실제로 사용하는 확장에서는 그 상태도 설계해야 하지만 이번 완료 조건에 추가하지 않는다.

## 6. 실제 검증 결과

작은 실제 PyTorch 파라미터의6회/3+3회 비교에서는 전체 상태를 복원하면 일치했고 optimizer·scheduler·RNG를 하나씩 빠뜨린 세 경우는 각각 비교 실패로 잡았다. 이는 실제 텐서/optimizer 검사지 학습된 LLM의 품질 결과는 아니다.

실제 NF4 통합 검증 결과는 아래에 기록한다. 로컬 입력은 미학습 zero-B 어댑터 fixture이며 사용자 CUDA 학습 어댑터와 구분한다. 이번 실습은 dev/test 생성 정답률을 평가하지 않는다.

### 실제 로컬 CPU NF4 예시

Python3.12·torch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2에서 실행했다. NF4 모듈210개·q/v LoRA 학습 대상230,400개인 실제 모델을 사용했다. 미학습 zero-B 입력으로 출발해 연속 경로와 분할 경로 모두 실제 AdamW 갱신을 수행했다.

연속 경로의 갱신 기록은 다음과 같다. 분할 경로의1~3회와 새 프로세스의4~6회도 같은 문항 순서·학습률·loss 기준을 통과했다.

| 갱신 | train 문항 인덱스 | 사용한 학습률 | 갱신 전 배치 평균 loss |
|---:|---|---:|---:|
| 1 | [7, 5] | 0.00100000 | 11.344495 |
| 2 | [1, 0] | 0.00100000 | 12.393765 |
| 3 | [3, 4] | 0.00050000 | 9.256082 |
| 4 | [0, 3] | 0.00050000 | 9.261672 |
| 5 | [6, 2] | 0.00025000 | 9.510821 |
| 6 | [5, 0] | 0.00025000 | 8.528906 |

```text
PREPARED reference_updates=6 checkpoint_updates=3
RESUMED_FROM=3 FINAL_STEP=6 | optimizer/scheduler/RNG restored
VERIFY=PASS (batch order/LR/adapter/optimizer/scheduler/RNG)
MAX_ADAPTER_ABS_DIFF=0 ATOL=1e-06 RTOL=1e-05
SAVED verify.json | quality not evaluated; existing models/registry unchanged
```

최종 상태 비교와 기존 입력/캐시·중간 체크포인트/기준 경로 보존을 확인했다. verify에서 저장 RNG가 손상된 경우 보고서를 만들지 않고 실패했고, 각 모드의 기존 출력 덮어쓰기도 거부했다. 작은 실제 텐서 검사의 optimizer/scheduler/RNG 누락3종과 별도로, 이 통합 검사는 실제 NF4 모델을 서로 다른 Python 프로세스에서 재개한 결과다.

검증 중 기존 TF32 설정 API와 PEFT 저장 시 config 조회 관련 UserWarning이 있었지만 실제 학습·재개·비교는 종료0으로 완료됐다. 경고가 해소됐거나 미래 버전 호환성이 검증됐다는 뜻은 아니다. **여기서 차이0은 이 로컬 실행의 결과**이며 모든 CUDA 환경·다른 버전에서의 비트 동일성 보장이 아니다. 다음의 별도 CUDA 예시와 입력·실행환경을 구분한다. 생성 정답률이나 품질 개선은 이번에 평가하지 않았다.


### 별도 CUDA VM 실행 예시

다음은 기존 학습 QLoRA 어댑터에서 출발한 CUDA VM의 제공된 콘솔 출력이다. 새 코드 생성 뒤 기준6회·저장 전3회·새 프로세스 재개4~6회와 최종 검증·보고서 저장·셸 복귀를 확인했다. 전체 `verify.json`이나 체크포인트를 직접 다시 읽은 결과는 아니다.

```text
DEVICE=cuda:0 NF4=210 TRAINABLE=230400
CONTINUOUS STEP=1 BATCH=[7, 5] LR=0.00100000 LOSS=9.822600
CONTINUOUS STEP=2 BATCH=[1, 0] LR=0.00100000 LOSS=10.183512
CONTINUOUS STEP=3 BATCH=[3, 4] LR=0.00050000 LOSS=7.007626
CONTINUOUS STEP=4 BATCH=[0, 3] LR=0.00050000 LOSS=6.891398
CONTINUOUS STEP=5 BATCH=[6, 2] LR=0.00025000 LOSS=5.935735
CONTINUOUS STEP=6 BATCH=[5, 0] LR=0.00025000 LOSS=5.478755
PREFIX STEP=1 BATCH=[7, 5] LR=0.00100000 LOSS=9.822600
PREFIX STEP=2 BATCH=[1, 0] LR=0.00100000 LOSS=10.183512
PREFIX STEP=3 BATCH=[3, 4] LR=0.00050000 LOSS=7.007626
PREPARED reference_updates=6 checkpoint_updates=3
DEVICE=cuda:0 NF4=210 TRAINABLE=230400
RESUMED STEP=4 BATCH=[0, 3] LR=0.00050000 LOSS=6.891398
RESUMED STEP=5 BATCH=[6, 2] LR=0.00025000 LOSS=5.935735
RESUMED STEP=6 BATCH=[5, 0] LR=0.00025000 LOSS=5.478755
RESUMED_FROM=3 FINAL_STEP=6 | optimizer/scheduler/RNG restored
VERIFY=PASS (batch order/LR/adapter/optimizer/scheduler/RNG)
MAX_ADAPTER_ABS_DIFF=0 ATOL=1e-06 RTOL=1e-05
SAVED verify.json | quality not evaluated; existing models/registry unchanged
```

분할 경로의 앞3회는 연속 경로의1~3회와, 재개한4~6회는 연속 경로의4~6회와 문항 인덱스·학습률·표시된 loss가 일치했다. 검증은 문항 순서·학습률 경로·최종 Torch RNG의 정확한 비교와 어댑터/optimizer/scheduler/loss의 명시한 허용오차 비교를 통과했다. 최종 어댑터의 실제 최대 절대 차이는0이었다. **이 CUDA 실행에서도 중간 저장·새 프로세스 학습 재개가 확인됐다.** 모든 환경에서의 비트 동일성이나 생성 정답률 개선을 뜻하지 않는다.

bitsandbytes CUDA backend의 `_check_is_size` 관련 FutureWarning도 있었지만 이후 재개·검증·저장과 셸 복귀까지 이어졌다. 경고 해소나 미래 호환성을 검증한 것은 아니며, 이 결과를 기록하기 위해 패키지를 바꾸거나 학습을 다시 실행하지 않는다. 수치 종료 코드는 콘솔에 별도로 제시되지 않았다. 기존32장의 test0/4와 품질 승인 배포 미실행은 그대로다.

VM에서는 `CONTINUOUS STEP=1..6`, `PREFIX STEP=1..3`, 다음 프로세스의 `RESUMED STEP=4..6`을 확인한다. 마지막 `VERIFY=PASS`와 `SAVED verify.json`이 이번 실습의 결과다. 로컬 참고 loss·최대 차이 수치를 맞추려고 사용자 모델을 초기화하거나 기존 체크포인트를 지우지 않는다.

## 7. 복습과 다음 묶음

- [ ] 어댑터 가중치 복원과 학습 상태 복원을 구분한다.
- [ ] scheduler와 optimizer의 초기화/상태 복원 순서를 설명한다.
- [ ] seed 재지정과 중간 RNG 상태 복원이 다름을 설명한다.
- [ ] 문항 순서·학습률·최종 파라미터/누적값을 함께 비교한다.
- [ ] 재개 성공을 정답률 개선이나 배포 승인으로 해석하지 않는다.

다음 묶음은 **[작은 모델 구조 확장](34-structure-comparison.md)**이다. 여러 head·층·문맥 길이의 효과를 같은 조건에서 비교하는 방향이며 이번 장에서 함께 구현하거나 실행하지 않는다. RAG는 제외한다.
