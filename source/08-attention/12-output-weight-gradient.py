from math import exp, isclose, isfinite, log

# 앞 출력층의 정보 벡터·행렬·어휘와 교육용 정답을 유지합니다.
hidden = [1.61332902229994, 5.931436569567589]
vocabulary = ["failed", "healthy", "<eos>"]
w_vocab = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]  # 2×3, bias 없음
original = [row[:] for row in w_vocab]
target = "failed"
assert target in vocabulary
target_index = vocabulary.index(target)
assert len(w_vocab) == len(hidden)
assert all(len(row) == len(vocabulary) for row in w_vocab)
assert all(isfinite(x) for x in hidden + [w for row in w_vocab for w in row])


def scores_at(matrix):
    return [sum(x * row[column] for x, row in zip(hidden, matrix, strict=True))
            for column in range(len(vocabulary))]


def loss_at(matrix):
    # ponytail: 고정 hidden·한 위치 전용; 전체 모델 자동 미분은 PyTorch로 전환합니다.
    scores = scores_at(matrix)
    maximum = max(scores)
    return log(sum(exp(z - maximum) for z in scores)) - (scores[target_index] - maximum)


logits = scores_at(w_vocab)
exps = [exp(z - max(logits)) for z in logits]
probabilities = [value / sum(exps) for value in exps]
d_logits = [p - (1.0 if index == target_index else 0.0)
            for index, p in enumerate(probabilities)]
# 각 가중치의 기울기 = 해당 입력 좌표 × 해당 출력 점수의 기울기
# 행은 hidden 좌표, 열은 failed·healthy·<eos>입니다.
d_weights = [[x * gradient for gradient in d_logits] for x in hidden]
assert isclose(loss_at(w_vocab), 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
assert len(d_weights) == 2 and all(len(row) == 3 for row in d_weights)
print("교육용 정답 토큰:", target)
print(f"기준 loss: {loss_at(w_vocab):.4f}")
epsilon = 1e-5
for row_index, row in enumerate(d_weights):
    for column, (token, gradient) in enumerate(zip(vocabulary, row, strict=True)):
        plus = [values[:] for values in w_vocab]
        minus = [values[:] for values in w_vocab]
        plus[row_index][column] += epsilon
        minus[row_index][column] -= epsilon
        numerical = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon)
        assert isclose(gradient, numerical, rel_tol=0.0, abs_tol=1e-8)
        print(f"W[{row_index},{column}] → {token}: 기울기={gradient:+.4f}, 수치검사={numerical:+.4f}")
assert w_vocab == original
print("검사 통과: 출력 행렬 가중치 6개의 기울기가 수치검사와 일치합니다. 가중치는 갱신하지 않았습니다")
