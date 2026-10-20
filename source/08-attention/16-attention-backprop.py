from math import isclose, sqrt
import torch

# 12절의 gpu/current/gpu 토큰+위치 벡터를 유지합니다. X는 학습하지 않습니다.
X = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
                 dtype=torch.float64, device="cpu")  # 3위치×2좌표
vocabulary = ["failed", "healthy", "<eos>"]
target = "failed"  # 마지막 위치의 교육용 다음 토큰 정답
learning_rate = 0.01
assert target in vocabulary
y = torch.tensor([vocabulary.index(target)], dtype=torch.long, device="cpu")
initial = {"Wq": [[1.0, 0.0], [1.0, 1.0]],
           "Wk": [[0.0, 1.0], [1.0, 0.0]],
           "Wv": [[2.0, 0.0], [0.0, 3.0]],
           "Wo": [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]}
parameters = {name: torch.nn.Parameter(torch.tensor(values, dtype=torch.float64, device="cpu"))
              for name, values in initial.items()}
future = torch.ones((3, 3), dtype=torch.bool, device="cpu").triu(diagonal=1)


def forward(wq, wk, wv, wo):
    # ponytail: 단일 head·3위치 전용; Transformer 블록 확장은 다음 단계입니다.
    Q, K, V = X @ wq, X @ wk, X @ wv
    scores = (Q @ K.T) / sqrt(Q.shape[-1])
    attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    hidden = (attention @ V)[-1:]  # 마지막 위치만 loss에 연결, shape 1×2
    logits = hidden @ wo  # 1×3, bias 없음
    return attention, hidden, logits


loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
optimizer = torch.optim.SGD(list(parameters.values()), lr=learning_rate,
                            momentum=0.0, weight_decay=0.0)
before = {name: value.detach().clone() for name, value in parameters.items()}
optimizer.zero_grad(set_to_none=True)
attention, hidden, logits = forward(*parameters.values())
expected_hidden = torch.tensor([[1.61332902229994, 5.931436569567589]], dtype=torch.float64)
assert torch.allclose(hidden, expected_hidden, rtol=0.0, atol=1e-12)
assert torch.all(attention[future] == 0.0)
assert torch.allclose(attention.sum(dim=-1), torch.ones(3, dtype=torch.float64))
hidden.retain_grad()  # 중간 결과의 기울기도 표시하기 위해 보관합니다.
loss = loss_fn(logits, y)
assert isclose(loss.item(), 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
loss.backward()
assert hidden.grad is not None
# z=hWo → dL/dh=dL/dz × Wo의 전치. 갱신 전 Wo로 비교합니다.
d_logits = torch.softmax(logits.detach(), dim=-1)
d_logits[0, y.item()] -= 1.0
manual_hidden_grad = d_logits @ parameters["Wo"].detach().T
assert torch.allclose(hidden.grad, manual_hidden_grad, rtol=0.0, atol=1e-12)
assert not X.requires_grad and X.grad is None
print("교육용 정답 토큰:", target)
print("갱신 전 hidden:", [round(v, 4) for v in hidden.detach()[0].tolist()])
print("hidden 기울기:", [round(v, 4) for v in hidden.grad[0].tolist()])
for name, parameter in parameters.items():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.norm().item() > 0.0
    assert torch.equal(parameter.detach(), before[name])
    print(f"{name}: 기울기 있음 | 크기={parameter.grad.norm().item():.4f}")
optimizer.step()
for name, parameter in parameters.items():
    assert not torch.equal(parameter.detach(), before[name])
    assert torch.allclose(parameter.detach(), before[name] - learning_rate * parameter.grad,
                          rtol=0.0, atol=1e-12)
with torch.no_grad():
    _, new_hidden, new_logits = forward(*parameters.values())
    new_loss = loss_fn(new_logits, y).item()
    old_probability = torch.softmax(logits.detach(), dim=-1)[0, y.item()].item()
    new_probability = torch.softmax(new_logits, dim=-1)[0, y.item()].item()
assert torch.isfinite(new_hidden).all() and torch.isfinite(new_logits).all()
assert not torch.allclose(new_hidden, hidden.detach(), rtol=0.0, atol=1e-12)
assert new_loss < loss.item() and new_probability > old_probability
print("갱신 후 hidden:", [round(v, 4) for v in new_hidden[0].tolist()])
print(f"failed 확률: {old_probability:.4f} → {new_probability:.4f}")
print(f"loss: {loss.item():.4f} → {new_loss:.4f}")
print("검사 통과: hidden을 거쳐 QKV·출력 행렬까지 역전파하고 함께 한 번 갱신했습니다. 저장은 없습니다")
