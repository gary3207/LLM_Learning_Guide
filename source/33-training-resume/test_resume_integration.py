"""Real CPU NF4: uninterrupted6 vs checkpoint3 + fresh-process3; zero-B input fixture."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name("01-resume-qlora.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "prepare" in help_result.stdout, "resume CLI missing"
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    module = runpy.run_path(str(SCRIPT))
    q = module["quality"]()
    deps = q["helpers"]()
    api = deps["api"]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cache = Path(snapshot_download(api["MODEL"], revision=api["REVISION"], allow_patterns=[
        "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]))
    before_cache = q["files"](cache)
    lora = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"], "bias": "none"}
    settings = {"base_model": api["MODEL"], "revision": api["REVISION"], "dtype": "float32",
        "attention": "eager", "lora": lora, "quantization": {"load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float32"}}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        flat = root / "source"
        flat.mkdir()
        copies = {"resume_qlora.py": SCRIPT,
            "improve_qlora.py": SCRIPT.parents[1] / "32-quality-improvement/01-improve-qlora.py",
            "quality_data_helpers.py": SCRIPT.parents[1] / "31-evaluation-data/01-prepare-data.py",
            "quality_eval_helpers.py": SCRIPT.parents[1] / "19-model-evaluation/01-evaluate-models.py",
            "quality_api_helpers.py": SCRIPT.parents[1] / "21-api-serving/01-serve-qlora.py"}
        for name, source in copies.items():
            (flat / name).write_bytes(source.read_bytes())
        old = root / "untrained_input"
        tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(cache, local_files_only=True,
                    dtype=torch.float32, attn_implementation="eager")
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                              revision=api["REVISION"], **lora))
        model.peft_config["default"].base_model_name_or_path = api["MODEL"]
        model.save_pretrained(old, safe_serialization=True)
        tokenizer.save_pretrained(old)
        (old / "qlora_settings.json").write_text(json.dumps(settings), encoding="utf-8")
        del base, model
        old_files = q["files"](old)
        data = flat / "evaluation_splits_v1.json"
        data.write_text(json.dumps(deps["data"]["build_dataset"]()), encoding="utf-8")
        old_data = data.read_bytes()
        output = root / "resume"
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
        common = [sys.executable, "-B", str(flat / "resume_qlora.py")]
        prepare = common + ["prepare", "--adapter", str(old), "--data", str(data),
                            "--output", str(output), "--device", "cpu"]
        # Inherited stdout/stderr streams show actual child progress in the terminal.
        assert subprocess.run(prepare, env=env, timeout=600).returncode == 0
        checkpoint_files = q["files"](output / "checkpoint")
        reference_files = q["files"](output / "reference")
        resume = common + ["resume", "--output", str(output), "--device", "cpu"]
        assert subprocess.run(resume, env=env, timeout=600).returncode == 0
        assert q["files"](output / "checkpoint") == checkpoint_files
        assert q["files"](output / "reference") == reference_files
        verify = common + ["verify", "--output", str(output), "--device", "cpu"]
        state_path = output / "resumed/training_state.pt"
        state_bytes = state_path.read_bytes()
        corrupted = torch.load(state_path, map_location="cpu", weights_only=True)
        corrupted["rng_cpu"][0] ^= 1
        torch.save(corrupted, state_path)
        rejected = subprocess.run(verify, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert rejected.returncode != 0 and not (output / "verify.json").exists()
        assert "난수" in rejected.stderr, rejected.stderr
        state_path.write_bytes(state_bytes)
        before_verify = q["files"](output)
        assert subprocess.run(verify, env=env, timeout=60).returncode == 0
        report = json.loads((output / "verify.json").read_text(encoding="utf-8"))
        assert report["ok"] and report["total_updates"] == 6 and report["checkpoint_step"] == 3
        assert report["max_adapter_abs_diff"] == 0.0
        assert {key: value for key, value in q["files"](output).items() if key != "verify.json"} == before_verify
        for command in (prepare, resume, verify):
            refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            assert refused.returncode != 0 and "FileExistsError" in refused.stderr
        assert q["files"](old) == old_files and data.read_bytes() == old_data
        assert q["files"](cache) == before_cache
        if not torch.cuda.is_available():
            no_gpu = root / "no_gpu"
            bad = prepare.copy()
            bad[bad.index(str(output))] = str(no_gpu)
            bad[-1] = "cuda"
            rejected = subprocess.run(bad, env=env, capture_output=True, text=True, timeout=60)
            assert rejected.returncode != 0 and not no_gpu.exists()
        reference = torch.load(output / "reference/training_state.pt", map_location="cpu", weights_only=True)
        if args.evidence:
            with args.evidence.open("x", encoding="utf-8") as stream:
                json.dump({"scope": "real CPU NF4 from an untrained zero-B adapter, not user CUDA replay",
                    "verification": report, "trace": reference["trace"], "input_cache_unchanged": True,
                    "fresh_process_resume": True, "rng_tamper_rejected": True,
                    "overwrite_refusals": True}, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("PASS real NF4 uninterrupted/checkpoint/fresh-resume equivalence, RNG corruption guard, preservation/refusals")


if __name__ == "__main__":
    main()
