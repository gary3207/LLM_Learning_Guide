"""Real CPU NF4 integration with an explicitly UNTRAINED zero-B input adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name("01-improve-qlora.py")


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "train" in help_result.stdout and "check" in help_result.stdout, (
        "train/check CLI is missing")
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.manual_seed(0)
    module = runpy.run_path(str(SCRIPT))
    deps = module["helpers"]()
    api = deps["api"]
    cache = Path(snapshot_download(api["MODEL"], revision=api["REVISION"], allow_patterns=[
        "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]))
    original_cache = files(cache)
    lora = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"], "bias": "none"}
    settings = {"base_model": api["MODEL"], "revision": api["REVISION"], "dtype": "float32",
                "attention": "eager", "lora": lora,
                "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float32"}}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        root = Path(tmp)
        old = root / "untrained_input"
        tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(cache, local_files_only=True, dtype=torch.float32,
                                                  attn_implementation="eager")
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                              revision=api["REVISION"], **lora))
        # A local cache path is not the base model identifier required by the real API loader.
        model.peft_config["default"].base_model_name_or_path = api["MODEL"]
        model.save_pretrained(old, safe_serialization=True)
        tokenizer.save_pretrained(old)
        (old / "qlora_settings.json").write_text(json.dumps(settings), encoding="utf-8")
        del model, base
        original_adapter = files(old)
        data = root / "data.json"
        data.write_text(json.dumps(deps["data"]["build_dataset"]()), encoding="utf-8")
        original_data = data.read_bytes()
        output = root / "quality"
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
        command = [sys.executable, "-B", str(SCRIPT), "train", "--adapter", str(old),
                   "--data", str(data), "--output", str(output), "--device", "cpu"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1200)
        print(result.stdout, flush=True)
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["selected_step"] == module["choose"](report["candidates"])
        assert "test" not in json.dumps(report["candidates"])  # No test generation during selection.
        assert report["frozen_base_unchanged"] and report["adapter_changed"]
        before_check = files(output)
        restored = subprocess.run([sys.executable, "-B", str(SCRIPT), "check", "--output", str(output),
                                  "--device", "cpu"], env=env, capture_output=True, text=True,
                                  encoding="utf-8", timeout=300)
        print(restored.stdout, flush=True)
        assert restored.returncode == 0, restored.stderr
        final = json.loads((output / "final_test.json").read_text(encoding="utf-8"))
        assert final["restore_verified"] and final["selected_step"] == report["selected_step"]
        assert final["selected_summary"]["total"] == final["baseline_summary"]["total"] == 4
        assert {k: v for k, v in files(output).items() if k != "final_test.json"} == before_check
        refused = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
        assert refused.returncode != 0 and "FileExistsError" in refused.stderr
        assert files(old) == original_adapter and data.read_bytes() == original_data
        assert files(cache) == original_cache
        if not torch.cuda.is_available():
            missing_gpu = root / "no_gpu"
            bad = command.copy()
            bad[bad.index(str(output))] = str(missing_gpu)
            bad[-1] = "cuda"
            rejected = subprocess.run(bad, env=env, capture_output=True, text=True, encoding="utf-8")
            assert rejected.returncode != 0 and not missing_gpu.exists()
        evidence = {"scope": "real CPU NF4 optimization from an UNTRAINED zero-B input; not user CUDA quality",
                    "report": report, "final": final,
                    "cache_and_input_unchanged": True, "fresh_process_restore": True,
                    "output_refusal": True}
        if args.evidence:
            with args.evidence.open("x", encoding="utf-8") as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("PASS actual CPU NF4 train/dev selection, fresh restore/final test, input/cache preservation, output refusal")


if __name__ == "__main__":
    main()
