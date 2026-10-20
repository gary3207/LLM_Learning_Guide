"""Exercise real SFT/save/fresh-process restore and preserve cached base weights."""
from pathlib import Path
import ast
import hashlib
import os
import subprocess
import sys
import tempfile
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

script = Path(__file__).with_name("01-sft-compare.py")
assert script.is_file(), "SFT 통합 실습이 아직 없습니다"
base = Path(hf_hub_download("HuggingFaceTB/SmolLM2-135M-Instruct", "model.safetensors",
                          revision="12fd25f77366fa6b3b4b768ec3050bf629380bac"))
tokenizer = AutoTokenizer.from_pretrained(base.parent, local_files_only=True)
definition = next(node for node in ast.parse(script.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.FunctionDef) and node.name == "example")
namespace = {"tokenizer": tokenizer, "torch": torch}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(script), "exec"), namespace)
messages = [{"role": "user", "content": "Device gpu status? Reply with one word."}]
prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
ids, labels = namespace["example"](messages[0]["content"], "failed")
target = tokenizer.encode("failed", add_special_tokens=False) + [tokenizer.eos_token_id]
assert ids.tolist() == [prefix + target]
assert labels.tolist() == [[-100] * len(prefix) + target], "질문 mask·응답/eos 정답 오류"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

original = digest(base)
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
    output = Path(temporary) / "smollm2_sft_v1"
    train = subprocess.run([sys.executable, str(script), "train", str(output)],
                           capture_output=True, text=True, encoding="utf-8")
    assert train.returncode == 0, train.stdout + train.stderr
    assert "추가 학습·별도 저장 완료" in train.stdout
    assert (output / "model.safetensors").is_file()
    saved = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    check = subprocess.run([sys.executable, str(script), "check", str(output)],
                           capture_output=True, text=True, encoding="utf-8")
    assert check.returncode == 0, check.stdout + check.stderr
    assert "복원 검사 통과" in check.stdout
    refused = subprocess.run([sys.executable, str(script), "train", str(output)],
                             capture_output=True, text=True, encoding="utf-8")
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
    assert {p.name: digest(p) for p in output.iterdir() if p.is_file()} == saved
    assert digest(base) == original
    print(train.stdout + check.stdout, end="")
    reference = output / "reference.pt"
    corrupted = torch.load(reference, weights_only=True)
    corrupted["logits"] = torch.zeros_like(corrupted["logits"])
    torch.save(corrupted, reference)
    mismatch = subprocess.run([sys.executable, str(script), "check", str(output)],
                              capture_output=True, text=True, encoding="utf-8")
    assert mismatch.returncode != 0 and "logits 불일치" in mismatch.stderr
print("검사 통과: 응답 mask·실제 SFT·새 프로세스 복원·재저장 거부·후보/원본 불변·불일치 탐지")
