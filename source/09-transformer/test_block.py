"""Run with the same PyTorch environment as 01-transformer-block.py."""
from pathlib import Path
import runpy
import torch

script = Path(__file__).with_name("01-transformer-block.py")
assert script.is_file(), "작은 Transformer 블록 실습이 아직 없습니다"
ns = runpy.run_path(str(script))  # 실습 자체의 역전파·SGD 검사도 실행합니다.
model, X = ns["model"], ns["X"]
with torch.no_grad():
    attention, hidden, logits = model(X)
    # LayerNorm과 두 잔차 경로를 실제 모듈과 독립적인 수식으로 비교합니다.
    def norm(x, layer):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + layer.eps) * layer.weight + layer.bias

    normalized = norm(X, model.norm1)
    Q, K, V = model.q(normalized), model.k(normalized), model.v(normalized)
    mask = torch.ones((3, 3), dtype=torch.bool).triu(diagonal=1)
    expected_attention = torch.softmax(((Q @ K.T) / (2 ** 0.5)).masked_fill(mask, -torch.inf), -1)
    residual = X + expected_attention @ V
    f = norm(residual, model.norm2) @ model.ff[0].weight.T + model.ff[0].bias
    f = torch.maximum(f, torch.zeros_like(f))
    expected_hidden = residual + f @ model.ff[2].weight.T + model.ff[2].bias
    assert torch.allclose(attention, expected_attention, rtol=0, atol=1e-12)
    assert torch.allclose(hidden, expected_hidden, rtol=0, atol=1e-12)
    assert torch.allclose(logits, expected_hidden[-1:] @ model.head.weight.T, rtol=0, atol=1e-12)
print("독립 검사 통과: 정규화 수식·두 잔차·FFN·어휘 출력 일치")
