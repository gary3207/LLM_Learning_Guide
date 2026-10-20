"""Real NF4 CPU integration: run with the tutorial's torch/transformers/peft/bnb environment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

script = Path(__file__).with_name("01-qlora.py")
assert script.is_file(), "QLoRA tutorial script is missing"

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
# Populate required input files before measuring cache immutability.
base = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=[
    "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
])) / "model.safetensors"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files(directory):
    return {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()}


def run(mode, output, device="cpu", offline=False):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run([sys.executable, str(script), mode, str(output), "--device", device],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=900)


original = files(base.parent)
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    output = Path(temporary) / "smollm2_qlora_v1"
    train = run("train", output)
    assert train.returncode == 0, train.stdout + train.stderr
    assert "NF4" in train.stdout and "원본/양자화 상태 불변" in train.stdout
    adapter = output / "adapter_model.safetensors"
    weights = load_file(adapter)
    assert weights and all("lora_" in key for key in weights)
    assert all(".q_proj." in key or ".v_proj." in key for key in weights)
    with safe_open(base, framework="pt", device="cpu") as original_weights:
        targets = [key for key in original_weights.keys()
                   if key.endswith(("self_attn.q_proj.weight", "self_attn.v_proj.weight"))]
        expected = sum(4 * sum(original_weights.get_slice(key).get_shape()) for key in targets)
    assert expected == sum(value.numel() for value in weights.values()) == 230400
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert any("lora_B" in key and torch.count_nonzero(value).item() for key, value in weights.items())
    assert not (output / "model.safetensors").exists() and not (output / "pytorch_model.bin").exists()
    saved = files(output)
    check = run("check", output, offline=True)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "QLoRA 복원 검사 통과" in check.stdout
    assert files(output) == saved, "check modified the candidate"
    refused = run("train", output, offline=True)
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
    assert files(output) == saved

    reference = output / "reference.pt"
    reference_bytes = reference.read_bytes()
    corrupted = torch.load(reference, weights_only=True, map_location="cpu")
    corrupted["logits"] = torch.zeros_like(corrupted["logits"])
    torch.save(corrupted, reference)
    mismatch = run("check", output, offline=True)
    assert mismatch.returncode != 0 and "logits 불일치" in mismatch.stderr, mismatch.stdout + mismatch.stderr
    reference.write_bytes(reference_bytes)

    # Invalid metadata must fail before even attempting to load the base.
    for filename, field, bad_value, message in (
        ("qlora_settings.json", "base_model", "invalid/base", "QLoRA settings mismatch"),
        ("qlora_settings.json", "quantization", {}, "QLoRA settings mismatch"),
        ("adapter_config.json", "base_model_name_or_path", "invalid/base", "adapter config mismatch"),
        ("adapter_config.json", "revision", "wrong-revision", "adapter config mismatch"),
    ):
        path = output / filename
        before = path.read_bytes()
        data = json.loads(before)
        data[field] = bad_value
        path.write_text(json.dumps(data), encoding="utf-8")
        rejected = run("check", output, offline=True)
        assert rejected.returncode != 0 and message in rejected.stderr, rejected.stdout + rejected.stderr
        path.write_bytes(before)
    if not torch.cuda.is_available():
        no_gpu = Path(temporary) / "no-gpu"
        rejected = run("train", no_gpu, "cuda")
        assert rejected.returncode != 0 and "CUDA 사용 불가" in rejected.stderr
        assert not no_gpu.exists()
        print("CUDA unavailable: explicit rejection before output creation verified; CUDA training not run")
    assert files(output) == saved
    assert files(base.parent) == original, "input base cache changed"
    print(train.stdout + check.stdout, end="")
    print(f"Adapter: {adapter.stat().st_size} bytes, {expected} q/v LoRA parameters; nonzero B")
print("PASS: real NF4 CPU train/fresh offline restore, frozen base+quant state, cache/candidate hashes, refusal/tamper checks")
