# 32 · QLoRA 학습량 비교와 모델 품질 개선

[← 이전: 독립 평가 데이터](31-evaluation-data.md) · [목차](README.md) · [다음: 학습 재개 →](33-training-resume.md)

## 1. 목표와 비교할 조건

31장에서 준비한 데이터로 **기존 어댑터를 읽기 → train으로 추가 학습 → dev로 후보 선택 → 새 프로세스 복원 확인 → 선택을 바꾸지 않고 최종 test 비교**를 실행한다. 학습 실행 성공과 정답률 개선은 다르며, 정답률이 늘지 않아도 결과를 그대로 남긴다.

기존 `smollm2_qlora_v1/`와 레지스트리·보고서·품질 게이트는 변경하지 않는다. 저장된 어댑터에서 새 최적화를 시작하지만 optimizer·scheduler·난수 상태를 복구하는 완전한 학습 재개는 아니다. 그 주제는 다음 묶음이다.

| 고정할 조건 | 내용 |
|---|---|
| 입력 | 같은 기존 QLoRA 어댑터에서 출발 |
| 데이터 | 31장 고정 train8/dev4/test4·기존 진단6개 |
| 구조 | 같은 pinned SmolLM2·NF4/double quant·q/v rank4·alpha8·dropout0 |
| 갱신 | SGD·lr0.01·momentum0·weight_decay0·clip_norm1.0 |
| 생성 | greedy·새 토큰 최대8개·동일한 대화 템플릿 |
| 바꾸는 조건 | **추가 갱신 횟수0/20/60** |

0은 모델을 새로 초기화했다는 뜻이 아니다. 기존 어댑터를 로딩한 직후이며, 이 실습에서 추가로 갱신한 횟수가0이다. 한 학습 경로를60회까지 이어가며 중간0/20/60 시점을 저장한다. 세 모델을 독립 초기화하는 실험이 아니다.

이전20장과 비교하면 학습 데이터도 늘었으므로 과거0/8과 이번 결과의 차이를 학습량 하나의 효과로 단정하지 않는다. **이번 실행 내부의 세 시점끼리** 데이터와 설정을 고정해 비교한다. 같은 데이터라도 갱신 횟수가 다르면 예제 노출량과 계산량이 다르다.

## 2. 후보 선택과 최종 test

후보를 고르는 기준은 실행 전에 다음 순서로 고정한다.

1. dev4개에서 **완전 일치 정답 수가 더 많은 후보**.
2. 정답 수가 같으면 dev의 **응답/eos 평균 loss가 더 낮은 후보**.
3. 둘 다 같으면 **추가 학습 횟수가 더 적은 후보**.

train 정답 수·test 정답 수는 후보 선택에 사용하지 않는다. 추가 학습0회가 가장 좋으면0회를 선택하며 더 오래 학습했다는 이유로60회를 강제 선택하지 않는다. loss만 좋아진 경우 정답률 개선이라고 부르지 않는다.

`train` 명령은 test의 모델 출력을 생성하지 않는다. `check` 명령이 저장된 dev 기준으로 선택을 검증하고, 별도 프로세스에서 선택 후보의 dev 생성 문자열·logits를 복원 확인한 다음에만 최종 test를 실행한다. 같은 최종 비교에서 추가 학습0회 기준과 선택 후보를 같은 test4개로 평가한다. 선택 후보가0회이면 같은 모델의 결과를 재사용한다. **test를 본 뒤 선택을 바꾸지 않는다.**

test4개는 두 장치의 작은 표현 보류 집합이다. 블라인드 시험·실제 장애 진단·운영 품질 승인이 아니다. 테스트 결과를 보고 새로 튜닝하면 그 문항도 개발에 사용한 데이터가 된다는31장 한계는 유지한다.

## 3. 실행 위치·입력·새 저장물

실행 위치는 기존 Ubuntu VM의 **`~/llm-lab/source`**, 환경은 `.venv`다. 기존 PyTorch·Transformers·PEFT·bitsandbytes와 캐시를 사용한다. 이번 실습을 위해 드라이버·패키지·기존 모델을 재설치하거나 다운로드하지 않는다. 원본 모델 로딩은 `local_files_only=True`다. 필요한 코드 파일 다운로드만 아래에서 수행한다.

입력:

- `~/llm-lab/smollm2_qlora_v1/`: 기존 어댑터·토크나이저·`qlora_settings.json`.
- `~/llm-lab/source/evaluation_splits_v1.json`:31장의 데이터 파일.

새 저장 위치: **`~/llm-lab/smollm2_quality_v1/`**. 기존 폴더가 있으면 학습 전에 거부한다. 실패 후 생긴 부분 폴더는 자동으로 지우지 않으며, 원인을 확인한 뒤 새 출력 이름으로 다시 실행한다.

```text
smollm2_quality_v1/
  dataset.json         이번 실행의 데이터 사본
  step-000/            추가 학습 전 어댑터 사본
  step-020/            추가20회 어댑터
  step-060/            추가60회 어댑터
  reference.pt         선택 후보의 dev logits (복원 비교용)
  report.json          세 시점 train/dev 생성·loss·오류·선택·설정
  final_test.json      check 완료 후 생성되는 최종 test 비교
```

모두 추론용 어댑터 저장물이며 optimizer/scheduler 상태 체크포인트가 아니다. 학습 중간에 종료되면 일부 새 저장물만 남을 수 있다. 기존 입력 모델이나 레지스트리에 덮어쓰지 않는다.

## 4. VM에서 실행

기존19장 채점·응답 labels 함수,21장 QLoRA 로더,31장 데이터 검사기를 재사용한다. 과거 스크립트를 수정하지 않고, 검증한 코드의 별도 helper 이름을 만든다. helper들은 `__main__` 실행 경로를 호출하지 않으므로 이들을 읽는 것만으로 모델 학습이나 서버 시작을 하지 않는다.

정본: [추가 학습 코드](source/32-quality-improvement/01-improve-qlora.py) · [표준 라이브러리 검사](source/32-quality-improvement/test_quality.py) · [실제 CPU 통합 검사](source/32-quality-improvement/test_quality_integration.py)

아래는 완전한 실행 블록이다. helper3개는 대상 파일을 먼저 확인하고 모두 다운로드/해시 검사를 마친 다음 새 파일로 기록한다. 이미 있는 파일은 보존한다. 뒤의 본문 코드도 `set -C`로 덮어쓰기를 거부한다. 다운로드나 파일 생성·학습이 실패하면 `&&` 뒤 단계는 실행하지 않는다. 여러 파일 생성은 디스크 오류까지 되돌리는 트랜잭션은 아니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
(
python - <<'FETCH'
from pathlib import Path
from urllib.request import urlopen
import hashlib

base = "https://raw.githubusercontent.com/gary3207/LLM_Learning_Guide/d9bb46bbcb4e8f66e872713328c0e036e948c138/"
items = [
    ('quality_data_helpers.py', 'source/31-evaluation-data/01-prepare-data.py', '3ef91b49b4987a9c4f306f3678226ec2d64990a5711a05eb6f9fd64897c6cf3d'),
    ('quality_eval_helpers.py', 'source/19-model-evaluation/01-evaluate-models.py', 'affa3b5fbff38efba4ae4ecc56fb3410870952d03ab797a5a17eb13ec72afc14'),
    ('quality_api_helpers.py', 'source/21-api-serving/01-serve-qlora.py', 'f203922443338a58e690f02cf3d838d20cd0a5ac2362769766b4144dc1af4e14'),
]
for name, _, _ in items:
    path = Path(name)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"기존 파일 보존: {path}")
payloads = []
for name, relative, expected in items:
    with urlopen(base + relative, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"해시 불일치: {name}")
    payloads.append((name, payload))
for name, payload in payloads:
    with Path(name).open("xb") as stream:
        stream.write(payload)
    print("CREATED", name)
FETCH
) &&
(set -C; cat > improve_qlora.py <<'PY'
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
PY
) &&
python -B improve_qlora.py train --device cuda &&
python -B improve_qlora.py check --device cuda
```

`--output ../smollm2_quality_v2`처럼 다른 이름을 쓰려면 **train/check 양쪽에 같은 경로를 지정**한다. CPU로 조용히 전환하지 않으며 CUDA가 없으면 새 결과 폴더 생성 전에 거부한다. 기본 흐름은 학습과 복원/최종평가를 별도 Python 프로세스로 실행한다. `final_test.json`이 이미 있으면 check는 기존 test 결과 보존을 위해 다시 실행하지 않는다.

## 5. 코드 읽기

- **`helpers()` / `read_data()`**: 기존 함수들을 실행 경로 없이 읽는다. 입력 JSON을31장 정본 데이터와 대조해 split 이동·문항 변경을 거부한다. test 정답이 파일에 존재하는 것과 test 모델 출력을 튜닝에 사용하는 것은 다르다.
- **`evaluate()`**:19장의 응답/eos labels와 채점 함수를 재사용한다. 질문 토큰의 labels는-100으로 무시하지만 질문 자체는 모델 입력에 남긴다. loss는 정답을 넣는 teacher forcing으로 계산하고, 응답 생성은 질문만 주고 실행한다. 각 문항의 응답/eos 평균 loss를 문항 수로 다시 평균한다.
- **`set_adapter("default")`**: 로딩한 어댑터의 학습 가능 상태를 활성화한다. 실제 학습 대상이 q/v LoRA의230,400개 파라미터인지 검사한다. NF4 원본 전체를 학습하지 않는다.
- **학습 루프**: 한 갱신마다 train8개 각각의 loss를8로 나눠 역전파를 누적하고, 마지막에 한 번 clip/SGD step을 한다. 큰 그래프8개를 한꺼번에 유지하지 않는 full-batch 평균 갱신이다.20회는 문항20개가 아니라 이 전체 갱신을20번 했다는 뜻이다.
- **원본 검사**: NF4 state_dict의 packed weights와 quantization state를 포함한 비어댑터 지문, 기존 입력 폴더·데이터를 비교한다. 어댑터가 실제로 바뀌었고 원본은 바뀌지 않았는지 확인한다. 단일 사용자가 실행 중 입력을 수정하지 않는 실습이며 동시 변경 방지·보안 서명은 아니다.
- **저장과 선택**: 세 시점을 모두 새 폴더에 저장하고 train/dev 문항별 원문·오류·loss를 기록한다. 선택은 dev만 사용한다. 새 결과를 기존 레지스트리에 등록하거나 품질 기준을 바꾸지 않는다.
- **`check()`**: 저장된 문항·채점·집계·선택·파일 지문을 재검사한다. 선택 후보를 복원해 dev 생성과 전체 마지막 위치 logits를 비교하고 그 뒤 최종 test를 수행한다. 입력 저장물은 유지하고 새 `final_test.json`만 만든다.

채점은 앞뒤 공백만 제거하는 대소문자 구분 완전 일치다. `failed.` 같은 형식 오류를 정답으로 바꾸지 않는다. `wrong_label`, `extra_text`, `other`는 오류 분석용이며 성공 점수를 부풀리는 보정이 아니다.

## 6. 검증 결과와 실행 출력 해석

아래에는 실제 로컬 통합 검사와 별도 CUDA VM의 실행 예시를 구분해 기록한다. 로컬 검증 입력은 기존 학습 어댑터가 아니라 **미학습 zero-B 어댑터 fixture**이므로 CUDA 학습 후보와 시작 조건이 다르다. 두 실행 사이의 loss 차이를 CPU/GPU 효과나 품질 우열로 해석하지 않는다.

### 실제 CPU NF4 검증 예시

Python3.12·torch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1·bitsandbytes0.48.2, CUDA 미사용. 실제 SmolLM2 NF4/double-quant 모듈210개와 학습 대상230,400개를 사용해 미학습 zero-B 입력에서 추가60회 갱신했다.

| 추가 갱신 | train 정답 | train 평균 loss | dev 정답 | dev 평균 loss |
|---:|---:|---:|---:|---:|
| 0 | 0/8 | 12.075057 | 0/4 | 10.760363 |
| 20 | 0/8 | 10.249809 | 0/4 | 9.075310 |
| 60 | 0/8 | 4.679242 | 0/4 | 4.545344 |

```text
SELECTED_ADDITIONAL_STEPS=60 TEST_NOT_RUN=True
SAVED report.json | existing model/registry/quality gate unchanged
RESTORE=PASS (fresh-process dev text/logits)
FINAL_TEST baseline=0/4 selected=0/4
TEST_ACCURACY_IMPROVED=False
SAVED final_test.json | no retraining during check; no registration/deployment
```

모든 시점의 dev 정답 수가0이므로 더 낮은 dev loss로60회를 선택했다. **이 예시에서 정답률 개선은 없었다.** 실제 최적화·어댑터 변경·동결 NF4 원본 불변·저장·별도 프로세스의 dev 문자열/logits 일치·선택 후 test·기존 입력/캐시 불변과 기존 출력 거부를 확인했다. 사용자 CUDA 어댑터의 품질 결과나 품질 승인 배포의 증거로 바꾸지 않는다.

표준 라이브러리 검사는 고정 데이터/오염 거부, dev 정답 우선·loss/횟수 동률 처리, 잘못된 dev 값 거부, flat helper 배치, 출력 덮어쓰기 거부,32장 가이드 검사 포함을 확인했다. 문서 검사기의 `training_executed: false`는 그 문서 검사 명령이 학습을 하지 않았다는 뜻이며, 위 실제 CPU 통합 검사의 최적화 실행까지 없었다는 뜻은 아니다.


### 별도 CUDA VM 실행 예시

다음은 기존 학습 QLoRA 어댑터에서 출발한 CUDA VM의 제공된 콘솔 출력이다. 코드4개 생성 뒤 아래 학습·복원·최종 test 결과와 셸 복귀를 확인했다. VM의 전체 `report.json`/`final_test.json` 파일을 직접 다시 읽은 결과와는 구분한다.

```text
DEVICE=cuda:0 NF4=210 TRAINABLE=230400
STEP=0 train=0/8 loss=10.493134
STEP=0 dev=0/4 loss=9.391413
STEP=20 train=0/8 loss=7.829241
STEP=20 dev=0/4 loss=7.103381
STEP=60 train=0/8 loss=3.235943
STEP=60 dev=0/4 loss=3.677019
SELECTED_ADDITIONAL_STEPS=60 TEST_NOT_RUN=True
SAVED report.json | existing model/registry/quality gate unchanged
RESTORE=PASS (fresh-process dev text/logits)
FINAL_TEST baseline=0/4 selected=0/4
TEST_ACCURACY_IMPROVED=False
SAVED final_test.json | no retraining during check; no registration/deployment
```

세 후보의 train 정답 수는 모두0/8, dev는 모두0/4였다. dev 정답 수가 동률이므로 dev loss가 가장 낮은 추가60회를 선택했다. **loss는 감소했지만 생성 정답 수는 늘지 않았고, 최종 test도0/4→0/4였다.** 따라서 학습량 비교·어댑터 저장·새 프로세스 복원·선택 후 test 실습은 완료됐지만 정답률 개선이나 품질 승인 배포를 달성한 것은 아니다. 문항별 생성 문자열을 이 콘솔만으로 알 수 없으므로 형식 오류인지 다른 라벨/무관한 답변인지 단정하지 않는다.

실행 중 bitsandbytes CUDA backend의 `_check_is_size` 관련 `FutureWarning`도 출력됐다. 이 로그에서는 경고 뒤 학습·복원·최종 test·저장과 셸 복귀까지 이어졌다. 경고 해소나 향후 버전 호환성을 검증한 것은 아니며, 결과 기록을 위해 패키지를 바꾸거나 실습을 다시 실행하지 않는다. 수치 종료 코드는 콘솔에 별도로 제시되지 않았다.

실행 중 `STEP=0/20/60 train=... dev=...`가 보인다. `SELECTED_ADDITIONAL_STEPS`는 dev 기준 선택이며 추가60회를 반드시 선택하는 것은 아니다. `TEST_NOT_RUN=True`는 후보 선택까지 test 생성을 하지 않았다는 뜻이다. 이어서 새 프로세스에서 `RESTORE=PASS`, `FINAL_TEST baseline=... selected=...`, `TEST_ACCURACY_IMPROVED=...`, `SAVED final_test.json`을 확인한다. 숫자는 환경·입력 모델에 따라 달라지며 성공 출력 숫자를 맞추기 위해 학습 조건을 바꾸지 않는다.

| 결과 | 뜻 | 뜻하지 않는 것 |
|---|---|---|
| dev loss만 감소 | 정답 응답의 teacher-forced loss 감소 | 생성 정답률 개선 |
| dev 정답 수 증가 | 후보 선택용 문항에서 개선 | test나 운영 성능 개선 보장 |
| RESTORE=PASS | 선택 후보의 저장/복원 출력 일치 | optimizer 상태까지 학습 재개 |
| test 정답 수 증가 | 이 네 보류 문항에서 기준 후보보다 개선 | 넓은 일반화·품질 게이트 승인 |
| TEST_ACCURACY_IMPROVED=False | 이 test에서 정답 수 개선 없음 | 학습/저장/복원 코드 실행 실패 |

## 7. 복습과 다음 묶음

- [ ] 추가 학습0회와 원본 사전학습 모델을 구분한다.
- [ ] 학습 횟수 비교에서 고정한 조건을 설명한다.
- [ ] dev로 고른 뒤 test를 보며 선택을 바꾸지 않는다.
- [ ] loss 감소와 생성 정답 수 증가를 따로 해석한다.
- [ ] 어댑터 복원과 optimizer/scheduler/RNG 복원이 다름을 설명한다.

다음 묶음은 **[학습 재현·재개](33-training-resume.md)**다. 이번 결과를 본 뒤 현재 비교의 한계를 정리하고 optimizer·scheduler·난수 상태를 포함하는 학습 이어하기로 확장한다. RAG·새 서빙 플랫폼·자동 배포를 추가하지 않는다.
