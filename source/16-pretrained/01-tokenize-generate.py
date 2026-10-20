"""사전학습된 작은 Instruct 모델로 토큰화·점수·반복 생성을 관찰합니다. 학습은 하지 않습니다."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
prompt = "Explain what a GPU is in one short sentence."
torch.set_num_threads(1)
# ponytail: 첫 추론은 CPU·float32·단일 입력; GPU 성능 비교는 별도 실습입니다.
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=False,
    use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
).to("cpu").eval()
print(f"모델: {model_id}")
print(f"revision: {revision}")
print(f"장치: {model.device} | dtype: {model.dtype} | 파라미터: {model.num_parameters():,}")
plain_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"질문: {prompt}")
print("질문 토큰:", list(zip(plain_ids, tokenizer.convert_ids_to_tokens(plain_ids))))
assert tokenizer.decode(plain_ids, clean_up_tokenization_spaces=False) == prompt
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
)
print("대화 형식:", repr(tokenizer.decode(inputs["input_ids"][0])))
print("input_ids shape:", tuple(inputs["input_ids"].shape))
with torch.inference_mode():
    logits = model(**inputs).logits
    output = model.generate(
        **inputs, max_new_tokens=48, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
assert torch.isfinite(logits).all()
new_ids = output[0, inputs["input_ids"].shape[1]:]
assert 1 <= new_ids.numel() <= 48
assert new_ids[0].item() == logits[0, -1].argmax().item()
answer = tokenizer.decode(new_ids, skip_special_tokens=True)
print("logits shape:", tuple(logits.shape))
print("첫 예측 토큰:", new_ids[0].item(), repr(tokenizer.decode(new_ids[:1])))
print("응답:", answer)
print(f"새 토큰 수: {new_ids.numel()} | 마지막이 eos: {new_ids[-1].item() == tokenizer.eos_token_id}")
print("추론 완료: 재학습 없음·기존 작은 모델 파일 사용/변경 없음")
