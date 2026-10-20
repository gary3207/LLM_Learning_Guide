from math import exp, isclose, isfinite, log

# 앞 실습의 초기 행렬에서 다시 시작합니다. 앞 실행의 변수가 필요하지 않습니다.
hidden = [1.61332902229994, 5.931436569567589]
vocabulary = ["failed", "healthy", "<eos>"]
w_vocab = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]  # 2×3, bias 없음
initial_weights = [row[:] for row in w_vocab]
target = "failed"
learning_rate = 0.01
steps = 20
assert target in vocabulary
assert isfinite(learning_rate) and learning_rate > 0.0
assert isinstance(steps, int) and steps > 0
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


probabilities, loss = forward(w_vocab)
initial_loss = loss
assert isclose(initial_loss, 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
print("교육용 정답 토큰:", target)
print(f"학습률: {learning_rate:.4f} | 갱신 횟수: {steps}")
print(f"갱신  0회 | failed 확률: {probabilities[target_index]:.4f} | loss: {loss:.4f}")
for step in range(1, steps + 1):
    # 확률·기울기는 직전 갱신이 반영된 현재 행렬의 값입니다.
    d_logits = [p - (1.0 if index == target_index else 0.0)
                for index, p in enumerate(probabilities)]
    d_weights = [[x * gradient for gradient in d_logits] for x in hidden]
    w_vocab = [[weight - learning_rate * gradient
                for weight, gradient in zip(row, gradients, strict=True)]
               for row, gradients in zip(w_vocab, d_weights, strict=True)]
    previous_loss = loss
    probabilities, loss = forward(w_vocab)
    assert isfinite(loss) and loss < previous_loss
    assert all(isfinite(w) for row in w_vocab for w in row)
    if step == 1:
        assert isclose(loss, 1.1487649561797542, rel_tol=0.0, abs_tol=1e-12)
    if step in (1, 5, 10, steps):
        print(f"갱신 {step:2d}회 | failed 확률: {probabilities[target_index]:.4f} | loss: {loss:.4f}")

# 이 입력·학습률·20회에 대한 독립 고정밀 Decimal 기준값입니다.
assert isclose(loss, 0.08110804034183388, rel_tol=0.0, abs_tol=1e-12)
assert isclose(probabilities[target_index], 0.9220940627050058, rel_tol=0.0, abs_tol=1e-12)
assert len(w_vocab) == 2 and all(len(row) == 3 for row in w_vocab)
assert w_vocab != initial_weights
prediction = vocabulary[max(range(len(vocabulary)), key=probabilities.__getitem__)]
assert prediction == target
print("최대 확률 후보:", prediction)
print("검사 통과: 매번 기울기를 다시 계산해 출력 행렬을 20회 갱신했습니다. 파일 저장은 없습니다")
