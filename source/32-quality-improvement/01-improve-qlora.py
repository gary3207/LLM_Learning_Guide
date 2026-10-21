"""QLoRA 추가 0/20/60회 비교: dev 선택 후 새 프로세스 복원과 최종 test."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import runpy

STEPS = (0, 20, 60)


def helpers():
    here = Path(__file__).resolve().parent
    paths = {
        "data": ("quality_data_helpers.py", "31-evaluation-data/01-prepare-data.py"),
        "evaluation": ("quality_eval_helpers.py", "19-model-evaluation/01-evaluate-models.py"),
        "api": ("quality_api_helpers.py", "21-api-serving/01-serve-qlora.py"),
    }
    return {key: runpy.run_path(str(here / flat if (here / flat).is_file() else here.parent / relative))
            for key, (flat, relative) in paths.items()}


def read_data(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    module = helpers()["data"]
    expected = module["build_dataset"]()
    if data != expected:
        raise ValueError("31장 고정 데이터와 다릅니다. test를 train으로 옮기지 마세요.")
    module["validate"](data)
    return data


def choose(candidates):
    if (any(type(row["step"]) is not int for row in candidates)
            or sorted(row["step"] for row in candidates) != list(STEPS)):
        raise ValueError("추가 학습0/20/60 시점이 모두 필요합니다.")
    for row in candidates:
        dev = row["summary"]["dev"]
        if (type(dev["correct"]) is not int or not 0 <= dev["correct"] <= 4
                or dev["total"] != 4 or not math.isfinite(dev["mean_loss"]) or dev["mean_loss"] < 0):
            raise ValueError("잘못된 dev 집계")
    return min(candidates, key=lambda row: (-row["summary"]["dev"]["correct"],
               row["summary"]["dev"]["mean_loss"], row["step"]))["step"]


def files(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_digest(model):
    import torch

    digest = hashlib.sha256()
    # NF4 state_dict includes packed weights AND packed quantization state.
    for name, tensor in sorted(model.state_dict().items()):
        if ".lora_" in name:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluate(tokenizer, model, groups, evaluation):
    import torch
    from transformers import GenerationConfig

    generation = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
        repetition_penalty=1.0, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    records, logits = [], []
    model.eval()
    with torch.inference_mode():
        for split, rows in groups.items():
            for row in rows:
                ids, labels, inputs = evaluation["example"](
                    tokenizer, row["prompt"], row["answer"], model.device)
                loss = model(ids, labels=labels, use_cache=False).loss.item()
                score = model(inputs, use_cache=False).logits[:, -1, :].cpu()
                if not math.isfinite(loss) or not torch.isfinite(score).all():
                    raise ValueError("유한하지 않은 평가 loss/logits")
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                           generation_config=generation)
                text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
                error = evaluation["classify"](text, row["answer"])
                records.append({"group": split, "prompt": row["prompt"], "answer": row["answer"],
                    "text": text, "error": error, "correct": error == "correct", "loss": loss})
                logits.append(score)
    return records, torch.cat(logits)


def train(args, deps, data):
    import torch

    old_files = files(args.adapter)
    input_data = args.data.read_bytes()
    tokenizer, model, count = deps["api"]["load_model"](args.adapter, args.device)
    torch.manual_seed(0)
    model.set_adapter("default")  # PEFT re-enables gradients on this loaded adapter only.
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if (sum(p.numel() for p in trainable.values()) != 230400
            or not all("lora_" in name and (".q_proj." in name or ".v_proj." in name)
                       for name in trainable)):
        raise ValueError("q/v rank4 어댑터만 학습해야 합니다.")
    before = {name: p.detach().clone() for name, p in trainable.items()}
    frozen_before = frozen_digest(model)
    settings = json.loads((args.adapter / "qlora_settings.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "dataset.json", data)
    optimizer = torch.optim.SGD(trainable.values(), lr=0.01, momentum=0.0, weight_decay=0.0)
    candidates, references = [], {}
    print(f"DEVICE={model.device} NF4={count} TRAINABLE={sum(p.numel() for p in trainable.values())}", flush=True)
    for step in range(STEPS[-1] + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # ponytail: 8 sequential backward calls form one full-batch update; no batching stack.
            for row in data["splits"]["train"]:
                ids, labels, _ = deps["evaluation"]["example"](
                    tokenizer, row["prompt"], row["answer"], model.device)
                loss = model(ids, labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("유한하지 않은 학습 loss")
                (loss / len(data["splits"]["train"])).backward()
            if (any(p.grad is not None for p in model.parameters() if not p.requires_grad)
                    or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values())):
                raise ValueError("동결 원본/어댑터 기울기 검사 실패")
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        if step not in STEPS:
            continue
        records, logits = evaluate(tokenizer, model,
            {split: data["splits"][split] for split in ("train", "dev")}, deps["evaluation"])
        summary = deps["evaluation"]["summarize"](records)
        folder = args.output / f"step-{step:03d}"
        folder.mkdir()
        model.save_pretrained(folder, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        saved_settings = dict(settings, training={"additional_steps": step, "optimizer": "SGD",
            "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0, "clip_norm": 1.0, "seed": 0})
        write_json(folder / "qlora_settings.json", saved_settings)
        candidates.append({"step": step, "summary": summary, "records": records, "files": files(folder)})
        references[step] = logits[len(data["splits"]["train"]):].clone()
        for split, score in summary.items():
            print(f"STEP={step} {split}={score['correct']}/{score['total']} loss={score['mean_loss']:.6f}", flush=True)
    changed = any(not torch.equal(before[name], p) for name, p in trainable.items())
    if not changed or frozen_digest(model) != frozen_before:
        raise ValueError("어댑터 미갱신 또는 동결된 원본/양자화 상태 변경")
    if files(args.adapter) != old_files or args.data.read_bytes() != input_data:
        raise ValueError("입력 어댑터/데이터가 실행 중 변경됐습니다.")
    selected = choose(candidates)
    torch.save({"dev_logits": references[selected]}, args.output / "reference.pt")
    report = {"schema": 1, "metadata": {"device": args.device, "base_model": deps["api"]["MODEL"],
        "revision": deps["api"]["REVISION"], "packages": {name: version(name) for name in
            ("torch", "transformers", "peft", "bitsandbytes")}, "additional_steps": list(STEPS),
        "selection": "dev exact-match count, then mean response/eos loss, then fewer updates",
        "optimizer_state_restored": False, "test_used_for_selection": False},
        "candidates": candidates, "selected_step": selected, "adapter_changed": changed,
        "frozen_base_unchanged": True, "input_adapter_unchanged": True,
        "dataset_sha256": hashlib.sha256((args.output / "dataset.json").read_bytes()).hexdigest()}
    write_json(args.output / "report.json", report)
    print(f"SELECTED_ADDITIONAL_STEPS={selected} TEST_NOT_RUN=True", flush=True)
    print("SAVED report.json | existing model/registry/quality gate unchanged", flush=True)


def check(args, deps):
    import torch

    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    data = read_data(args.output / "dataset.json")
    if (hashlib.sha256((args.output / "dataset.json").read_bytes()).hexdigest() != report["dataset_sha256"]
            or report["metadata"]["device"] != args.device):
        raise ValueError("데이터 또는 학습/복원 장치 불일치")
    for candidate in report["candidates"]:
        expected = [(split, row["prompt"], row["answer"])
                    for split in ("train", "dev") for row in data["splits"][split]]
        if [(r["group"], r["prompt"], r["answer"]) for r in candidate["records"]] != expected:
            raise ValueError("저장된 train/dev 문항 불일치")
        for row in candidate["records"]:
            if (not math.isfinite(row["loss"]) or row["loss"] < 0
                    or row["error"] != deps["evaluation"]["classify"](row["text"], row["answer"])
                    or row["correct"] != (row["error"] == "correct")):
                raise ValueError("저장된 평가 형식/정오답 불일치")
        if deps["evaluation"]["summarize"](candidate["records"]) != candidate["summary"]:
            raise ValueError("저장된 dev 집계 불일치")
    selected = choose(report["candidates"])
    if type(report["selected_step"]) is not int or selected != report["selected_step"]:
        raise ValueError("dev 선택 결과 불일치")
    original = files(args.output)
    for step in {0, selected}:
        candidate = next(c for c in report["candidates"] if c["step"] == step)
        if files(args.output / f"step-{step:03d}") != candidate["files"]:
            raise ValueError("저장된 어댑터 파일 변경")
    tokenizer, model, _ = deps["api"]["load_model"](args.output / f"step-{selected:03d}", args.device)
    dev, logits = evaluate(tokenizer, model, {"dev": data["splits"]["dev"]}, deps["evaluation"])
    saved = next(c for c in report["candidates"] if c["step"] == selected)
    expected_dev = [r for r in saved["records"] if r["group"] == "dev"]
    reference = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    if ([r["text"] for r in dev] != [r["text"] for r in expected_dev]
            or not torch.allclose(logits, reference["dev_logits"], atol=1e-4, rtol=1e-4)):
        raise ValueError("복원 후 dev 생성/logits 불일치: test 미실행")
    print("RESTORE=PASS (fresh-process dev text/logits)", flush=True)
    test, _ = evaluate(tokenizer, model, {"test": data["splits"]["test"]}, deps["evaluation"])
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    if selected:
        tokenizer, baseline, _ = deps["api"]["load_model"](args.output / "step-000", args.device)
        baseline_test, _ = evaluate(tokenizer, baseline, {"test": data["splits"]["test"]}, deps["evaluation"])
        del baseline
    else:
        baseline_test = test
    selected_summary = deps["evaluation"]["summarize"](test)["test"]
    baseline_summary = deps["evaluation"]["summarize"](baseline_test)["test"]
    if files(args.output) != original:
        raise ValueError("복원 중 저장 파일 변경")
    final = {"selected_step": selected, "restore_verified": True,
        "test_used_for_selection": False, "selected_summary": selected_summary,
        "baseline_summary": baseline_summary, "selected_records": test, "baseline_records": baseline_test,
        "test_accuracy_improved": selected_summary["correct"] > baseline_summary["correct"]}
    write_json(args.output / "final_test.json", final)
    print(f"FINAL_TEST baseline={baseline_summary['correct']}/4 selected={selected_summary['correct']}/4", flush=True)
    print(f"TEST_ACCURACY_IMPROVED={final['test_accuracy_improved']}", flush=True)
    print("SAVED final_test.json | no retraining during check; no registration/deployment", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "check"])
    parser.add_argument("--adapter", type=Path, default=Path("../smollm2_qlora_v1"))
    parser.add_argument("--data", type=Path, default=Path("evaluation_splits_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("../smollm2_quality_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.mode == "train":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"기존 출력 보존: {args.output}")
        if args.output.resolve().is_relative_to(args.adapter.resolve()):
            raise ValueError("입력 어댑터 내부에 출력할 수 없습니다.")
        if not args.adapter.is_dir():
            raise FileNotFoundError(args.adapter)
    elif (args.output / "final_test.json").exists() or (args.output / "final_test.json").is_symlink():
        raise FileExistsError("기존 최종 test 결과 보존")
    deps = helpers()
    data = read_data(args.data) if args.mode == "train" else None
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    if args.mode == "train":
        train(args, deps, data)
    else:
        check(args, deps)


if __name__ == "__main__":
    main()
