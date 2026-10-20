"""Small real-PyTorch check; does not read or write checkpoints."""
from pathlib import Path
import runpy
import torch

script = Path(__file__).with_name("02-embedding-next-token.py")
assert script.is_file(), "임베딩·다음 토큰 연결 실습이 아직 없습니다"
ns = runpy.run_path(str(script))
model, x, y, vocab = (ns[k] for k in ("model", "x", "y", "vocab"))
assert [[vocab[i] for i in row] for row in x.tolist()] == [
    ["gpu", "current", "status", "failed"],
    ["network", "current", "status", "healthy"]]
assert [[vocab[i] for i in row] for row in y.tolist()] == [
    ["current", "status", "failed", "<eos>"],
    ["current", "status", "healthy", "<eos>"]]
with torch.no_grad():
    logits, attention = model(x)
    assert logits.shape == (2, 4, 7) and attention.shape == (2, 4, 4)
    assert torch.allclose(logits[0:1], model(x[0:1])[0], rtol=0, atol=1e-12)
    assert torch.allclose(logits[1:2], model(x[1:2])[0], rtol=0, atol=1e-12)
    changed = x.clone()
    changed[:, 2:] = ns["to_id"]["gpu"]
    assert torch.allclose(logits[:, :2], model(changed)[0][:, :2], rtol=0, atol=1e-12)
    # 각 접두 문맥만 계산해도 같은 위치의 출력이 일치하는지 검사합니다.
    for length in range(1, 5):
        prefix_logits, _ = model(x[:, :length])
        assert torch.allclose(prefix_logits[:, -1], logits[:, length-1], rtol=0, atol=1e-12)
    assert torch.equal(logits.argmax(-1), y)
print("독립 검사 통과: 정답 한 칸 이동·문장 분리·batch 분리·prefix 일치")
