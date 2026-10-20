"""Run real CPU LoRA, verify adapter-only saving and fresh-process restore."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import tempfile
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

script = Path(__file__).with_name("01-lora-gpu.py")
assert script.is_file(), "LoRA 통합 실습이 아직 없습니다"
base = Path(hf_hub_download("HuggingFaceTB/SmolLM2-135M-Instruct", "model.safetensors",
                          revision="12fd25f77366fa6b3b4b768ec3050bf629380bac"))
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

original = digest(base)
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    output = Path(temporary) / "smollm2_lora_v1"
    train = subprocess.run([sys.executable, str(script), "train", str(output), "--device", "cpu"],
                           capture_output=True, text=True, encoding="utf-8")
    assert train.returncode == 0, train.stdout + train.stderr
    assert "원본 동결 가중치 불변" in train.stdout
    adapter = output / "adapter_model.safetensors"
    assert adapter.is_file() and not (output / "model.safetensors").exists()
    weights = load_file(adapter)
    assert weights and all("lora_" in key for key in weights)
    with safe_open(base, framework="pt", device="cpu") as original_weights:
        targets = [key for key in original_weights.keys()
                   if key.endswith(("self_attn.q_proj.weight", "self_attn.v_proj.weight"))]
        expected_count = sum(4 * sum(original_weights.get_slice(key).get_shape()) for key in targets)
    assert targets and sum(value.numel() for value in weights.values()) == expected_count
    assert all(".q_proj." in key or ".v_proj." in key for key in weights)
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert any("lora_B" in key and torch.count_nonzero(value).item() for key, value in weights.items())
    saved = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    check = subprocess.run([sys.executable, str(script), "check", str(output), "--device", "cpu"],
                           capture_output=True, text=True, encoding="utf-8")
    assert check.returncode == 0, check.stdout + check.stderr
    assert "LoRA 복원 검사 통과" in check.stdout
    refused = subprocess.run([sys.executable, str(script), "train", str(output)],
                             capture_output=True, text=True, encoding="utf-8")
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
    assert {p.name: digest(p) for p in output.iterdir() if p.is_file()} == saved
    assert digest(base) == original
    reference = output / "reference.pt"
    corrupted = torch.load(reference, weights_only=True)
    corrupted["logits"] = torch.zeros_like(corrupted["logits"])
    torch.save(corrupted, reference)
    mismatch = subprocess.run([sys.executable, str(script), "check", str(output)],
                              capture_output=True, text=True, encoding="utf-8")
    assert mismatch.returncode != 0 and "logits 불일치" in mismatch.stderr
    if not torch.cuda.is_available():
        no_gpu = Path(temporary) / "no-gpu"
        rejected = subprocess.run([sys.executable, str(script), "train", str(no_gpu), "--device", "cuda"],
                                  capture_output=True, text=True, encoding="utf-8")
        assert rejected.returncode != 0 and "CUDA 사용 불가" in rejected.stderr
        assert not no_gpu.exists()
    print(train.stdout + check.stdout, end="")
    print(f"어댑터 파일 검증: {adapter.stat().st_size} bytes | 원본 가중치 바이너리 미포함")
print("검사 통과: 실제 CPU LoRA·어댑터만 저장·원본/후보 불변·재저장 거부·변조 탐지")
