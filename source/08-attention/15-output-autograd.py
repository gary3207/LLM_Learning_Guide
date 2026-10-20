from math import exp, isclose, log
import torch

# 앞 실습과 같은 초기값입니다. 20회 갱신한 행렬을 가져오지 않습니다.
hidden = [1.61332902229994, 5.931436569567589]
vocabulary = ["failed", "healthy", "<eos>"]
weights = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]  # 2×3, bias 없음
target = "failed"
learning_rate = 0.01
assert target in vocabulary
target_index = vocabulary.index(target)

# 앞 표준 라이브러리 계산으로 수동 비교값을 만듭니다.
scores = [sum(hidden[r] * weights[r][c] for r in range(2)) for c in range(3)]
maximum = max(scores)
exps = [exp(z - maximum) for z in scores]
probabilities = [value / sum(exps) for value in exps]
manual_loss = log(sum(exps)) - (scores[target_index] - maximum)
d_logits = [p - (1.0 if c == target_index else 0.0)
            for c, p in enumerate(probabilities)]
manual_grad = [[x * gradient for gradient in d_logits] for x in hidden]
manual_updated = [[weights[r][c] - learning_rate * manual_grad[r][c]
                   for c in range(3)] for r in range(2)]

# CPU·float64로 비교합니다. h는 고정, W만 학습 대상입니다.
h = torch.tensor([hidden], dtype=torch.float64, device="cpu")  # 1×2
W = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64, device="cpu"))
y = torch.tensor([target_index], dtype=torch.long, device="cpu")  # 정답 번호 1개
initial_W = W.detach().clone()
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
optimizer = torch.optim.SGD([W], lr=learning_rate, momentum=0.0, weight_decay=0.0)
optimizer.zero_grad(set_to_none=True)
assert W.grad is None
logits = h @ W  # 1×3, 정답은 shape (1,)
loss = loss_fn(logits, y)  # softmax 전 점수를 넣습니다.
assert isclose(loss.item(), manual_loss, rel_tol=0.0, abs_tol=1e-12)
loss.backward()
assert W.grad is not None
expected_grad = torch.tensor(manual_grad, dtype=torch.float64, device="cpu")
assert torch.allclose(W.grad, expected_grad, rtol=0.0, atol=1e-12)
assert torch.equal(W.detach(), initial_W)  # backward는 가중치를 바꾸지 않습니다.
assert not h.requires_grad and h.grad is None
print("교육용 정답 토큰:", target)
print("비교 환경: CPU / float64")
for r in range(2):
    for c, token in enumerate(vocabulary):
        print(f"W[{r},{c}] → {token}: 수동={manual_grad[r][c]:+.4f}, 자동={W.grad[r,c].item():+.4f}")

optimizer.step()  # 계산된 기울기로 W를 한 번 바꿉니다.
expected_W = torch.tensor(manual_updated, dtype=torch.float64, device="cpu")
assert torch.allclose(W.detach(), expected_W, rtol=0.0, atol=1e-12)
with torch.no_grad():
    new_logits = h @ W
    new_loss = loss_fn(new_logits, y).item()
    new_probability = torch.softmax(new_logits, dim=1)[0, target_index].item()
assert isclose(new_loss, 1.1487649561797542, rel_tol=0.0, abs_tol=1e-12)
assert isclose(new_probability, 0.31702807125207865, rel_tol=0.0, abs_tol=1e-12)
assert new_loss < loss.item()
updated_W = W.detach().clone()
optimizer.zero_grad(set_to_none=True)  # 다음 반복에 대비해 기울기만 지웁니다.
assert W.grad is None and torch.equal(W.detach(), updated_W)
print("SGD 갱신과 수동 갱신: 가중치 6개 일치")
print(f"failed 확률: {probabilities[target_index]:.4f} → {new_probability:.4f}")
print(f"loss: {loss.item():.4f} → {new_loss:.4f}")
print("검사 통과: 수동·자동 기울기와 한 번 갱신 결과가 일치합니다. 파일 저장은 없습니다")
