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
