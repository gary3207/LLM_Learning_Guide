"""Run the real cached model and check tokenization and greedy generation (no training)."""
from pathlib import Path
import runpy
import torch

script = Path(__file__).with_name("01-tokenize-generate.py")
assert script.is_file(), "사전학습 모델 통합 실습이 아직 없습니다"
ns = runpy.run_path(str(script))
tokenizer, model = ns["tokenizer"], ns["model"]
inputs, output, new_ids = ns["inputs"], ns["output"], ns["new_ids"]
assert tokenizer.decode(ns["plain_ids"], clean_up_tokenization_spaces=False) == ns["prompt"]
assert torch.equal(output[:, :inputs["input_ids"].shape[1]], inputs["input_ids"])
assert 1 <= new_ids.numel() <= 48
assert new_ids[0].item() == ns["logits"][0, -1].argmax().item(), "greedy 첫 토큰 불일치"
assert ns["logits"].shape == (1, inputs["input_ids"].shape[1], model.config.vocab_size)
assert torch.isfinite(ns["logits"]).all()
assert not model.training and all(p.grad is None for p in model.parameters())
assert ns["answer"] == tokenizer.decode(new_ids, skip_special_tokens=True)
assert ns["answer"].strip(), "응답 본문이 비어 있습니다"
print("검사 통과: 토큰 복원·입력 보존·첫 argmax·생성 길이·유한 logits·gradient 없음")
