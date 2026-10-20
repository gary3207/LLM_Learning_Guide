"""11장의 VM 배치 폴더에서 기존 모델을 바꾸지 않고 새 입력을 관찰합니다."""
from pathlib import Path
import runpy
import torch

torch.set_num_threads(1)
lm = runpy.run_path("tiny_transformer_checkpoint.py")
checkpoint = Path("../tiny_transformer_v1.pt")
original = checkpoint.read_bytes()
model = lm["restore"](checkpoint)
generate = lm["generate"]

controls = {"gpu current status": ["gpu", "current", "status", "failed", "<eos>"],
            "network current status": ["network", "current", "status", "healthy", "<eos>"]}
# ponytail: 선택한 입력의 정성 관찰뿐; 일반화 점수는 별도 정답 데이터로 평가합니다.
for prompt in [*controls, "gpu status", "network status", "status current gpu",
               "gpu current status healthy"]:
    tokens, reason = generate(model, prompt.split())
    if prompt in controls:
        assert tokens == controls[prompt] and reason == "eos"
    print(f"입력: {prompt} -> {' '.join(tokens)} | 종료: {reason}")

try:
    generate(model, ["storage", "status"])
except ValueError as exc:
    print(f"미등록 입력 거부: storage status | {exc}")
else:
    raise AssertionError("미등록 토큰을 허용했습니다")
assert checkpoint.read_bytes() == original, "체크포인트가 변경됐습니다"
print("검사 통과: 학습 접두 문맥 재현·미등록 입력 거부·체크포인트 불변")
