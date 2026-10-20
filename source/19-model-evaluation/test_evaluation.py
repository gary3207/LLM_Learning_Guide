"""Small scorer checks; --integration also exercises real untrained model fixtures, not SFT quality."""
import importlib.util
from pathlib import Path
import sys

script = Path(__file__).with_name("01-evaluate-models.py")
assert script.is_file(), "모델 비교·평가 코드가 아직 없습니다"
spec = importlib.util.spec_from_file_location("evaluation", script)
assert spec is not None and spec.loader is not None
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
assert evaluation.classify(" failed\n", "failed") == "correct"
assert evaluation.classify("healthy", "failed") == "wrong_label"
assert evaluation.classify("failed because of an error", "failed") == "extra_text"
assert evaluation.classify('The GPU status is "failed".', "failed") == "extra_text"
assert evaluation.classify("unfailed", "failed") == "other"
assert evaluation.classify("FAILED", "failed") == "other"
assert evaluation.classify("", "failed") == "other"
assert len(evaluation.ROWS) == 8
assert len({row[1] for row in evaluation.ROWS}) == 8
rows = [{"group": "probe", "correct": True, "loss": 1.0},
        {"group": "probe", "correct": False, "loss": 3.0}]
assert evaluation.summarize(rows) == {"probe": {"correct": 1, "total": 2, "accuracy": 0.5, "mean_loss": 2.0}}
print("검사 통과: 빈 출력·공백·대소문자·다른 라벨·추가 문구 판정, 문항/그룹별 집계")

if "--integration" in sys.argv:
    import hashlib
    import json
    import math
    import os
    import subprocess
    import tempfile
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    torch.set_num_threads(1)
    options = dict(revision=evaluation.REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(evaluation.MODEL_ID, **options)
    model = AutoModelForCausalLM.from_pretrained(
        evaluation.MODEL_ID, use_safetensors=True, dtype=torch.float32,
        attn_implementation="eager", **options).eval()
    # Independent response/eos CE from actual logits; this is not a trained candidate.
    prompt, answer = evaluation.ROWS[0][1:]
    ids, labels, _ = evaluation.example(tokenizer, prompt, answer, "cpu")
    with torch.inference_mode():
        output = model(ids, labels=labels, use_cache=False)
    losses = []
    for position in range(1, ids.shape[1]):
        if labels[0, position].item() != -100:
            values = output.logits[0, position - 1].double().tolist()
            maximum = max(values)
            losses.append(maximum + math.log(sum(math.exp(x - maximum) for x in values))
                          - values[labels[0, position].item()])
    assert losses and math.isclose(output.loss.item(), sum(losses) / len(losses), abs_tol=2e-5)

    def hashes(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        full, adapter = root / "untrained-full-fixture", root / "zero-adapter-fixture"
        model.save_pretrained(full, safe_serialization=True)
        tokenizer.save_pretrained(full)
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                            target_modules=["q_proj", "v_proj"], bias="none", revision=evaluation.REVISION)
        model = get_peft_model(model, config)
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        del model
        before = hashes(root)
        report = root / "report.json"
        command = [sys.executable, str(script), "--sft", str(full), "--lora", str(adapter),
                   "--output", str(report), "--device", "cpu"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(report.read_text(encoding="utf-8"))
        assert len(data["records"]) == 24 and set(data["summary"]) == {"base", "sft", "lora"}
        base = [{k: v for k, v in r.items() if k != "model"} for r in data["records"][:8]]
        for start in (8, 16):
            candidate = [{k: v for k, v in r.items() if k != "model"}
                         for r in data["records"][start:start + 8]]
            for left, right in zip(base, candidate):
                assert {k: v for k, v in left.items() if k != "loss"} == {k: v for k, v in right.items() if k != "loss"}
                assert math.isclose(left["loss"], right["loss"], abs_tol=2e-5)
        assert all(hashes(root)[name] == value for name, value in before.items())
        report_before = report.read_bytes()
        refused = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        assert refused.returncode != 0 and report.read_bytes() == report_before
        if not torch.cuda.is_available():
            no_cuda = root / "no-cuda.json"
            rejected = subprocess.run(command[:-3] + [str(no_cuda), "--device", "cuda"],
                                      capture_output=True, text=True, encoding="utf-8")
            assert rejected.returncode != 0 and not no_cuda.exists()
            assert "CUDA 사용 불가" in rejected.stderr
        print(result.stdout, end="")
    print("실제 CPU 통합 검사 통과: 미학습 전체모델·zero LoRA fixture 동등성·24건·수동CE·입력파일불변·덮어쓰기거부")
    print("주의: 통합 fixture는 미학습 모델입니다. 기존 SFT/LoRA 품질 비교 결과가 아닙니다.")
