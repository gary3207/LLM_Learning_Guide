from math import exp, isclose, isfinite, log

# 앞 실습의 고정 입력·출력 행렬·교육용 정답을 유지합니다.
hidden = [1.61332902229994, 5.931436569567589]
vocabulary = ["failed", "healthy", "<eos>"]
w_vocab = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]  # 2×3, bias 없음
original = [row[:] for row in w_vocab]
target = "failed"
learning_rate = 0.01
assert target in vocabulary
assert isfinite(learning_rate) and learning_rate > 0.0
target_index = vocabulary.index(target)
assert len(w_vocab) == len(hidden)
assert all(len(row) == len(vocabulary) for row in w_vocab)
assert all(isfinite(x) for x in hidden + [w for row in w_vocab for w in row])


def forward(matrix):
    # ponytail: 고정 hidden·한 위치 전용; 전체 모델 학습은 PyTorch로 전환합니다.
    scores = [sum(x * row[column] for x, row in zip(hidden, matrix, strict=True))
              for column in range(len(vocabulary))]
    maximum = max(scores)
    exps = [exp(z - maximum) for z in scores]
    total = sum(exps)
    probabilities = [value / total for value in exps]
    loss = log(total) - (scores[target_index] - maximum)
    return probabilities, loss


before_probabilities, before_loss = forward(w_vocab)
d_logits = [p - (1.0 if index == target_index else 0.0)
            for index, p in enumerate(before_probabilities)]
d_weights = [[x * gradient for gradient in d_logits] for x in hidden]
# 갱신 전의 같은 행렬에서 구한 기울기로 모든 좌표를 한 번 갱신합니다.
updated = [[weight - learning_rate * gradient
            for weight, gradient in zip(row, gradients, strict=True)]
           for row, gradients in zip(w_vocab, d_weights, strict=True)]
after_probabilities, after_loss = forward(updated)

# 독립 고정밀 Decimal 계산의 기준값과 비교합니다.
assert isclose(before_loss, 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
assert isclose(after_loss, 1.1487649561797542, rel_tol=0.0, abs_tol=1e-12)
assert isclose(after_probabilities[target_index], 0.31702807125207865,
               rel_tol=0.0, abs_tol=1e-12)
assert len(updated) == 2 and all(len(row) == 3 for row in updated)
assert all(isfinite(w) for row in updated for w in row)
assert after_loss < before_loss
assert after_probabilities[target_index] > before_probabilities[target_index]
assert w_vocab == original
assert all(updated[r][c] != w_vocab[r][c] for r in range(2) for c in range(3))
print("교육용 정답 토큰:", target)
print(f"학습률: {learning_rate:.4f}")
for r, row in enumerate(updated):
    print(f"W[{r}] 갱신 전: {[round(w, 4) for w in w_vocab[r]]}")
    print(f"W[{r}] 갱신 후: {[round(w, 4) for w in row]}")
for token, before, after in zip(vocabulary, before_probabilities, after_probabilities, strict=True):
    print(f"{token} 확률: {before:.4f} → {after:.4f}")
print(f"loss: {before_loss:.4f} → {after_loss:.4f}")
print("검사 통과: 출력 행렬을 한 번 갱신해 정답 확률이 오르고 loss가 줄었습니다. 파일 저장은 없습니다")
