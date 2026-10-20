from math import exp, isclose, isfinite, log

# 앞 출력층의 반올림 전 점수와 앞 loss 실습의 교육용 정답을 유지합니다.
vocabulary = ["failed", "healthy", "<eos>"]
logits = [1.61332902229994, 2.9657182847837945, -1.61332902229994]
target = "failed"
assert len(logits) == len(vocabulary) and all(isfinite(z) for z in logits)
assert target in vocabulary
target_index = vocabulary.index(target)


def loss_at(scores):
    # ponytail: 한 위치·고정 정답 전용; batch와 자동 미분은 PyTorch로 전환합니다.
    maximum = max(scores)
    return log(sum(exp(z - maximum) for z in scores)) - (scores[target_index] - maximum)


exps = [exp(z - max(logits)) for z in logits]
probabilities = [value / sum(exps) for value in exps]
# softmax + 한 정답 cross-entropy의 logits에 대한 기울기입니다.
gradients = [p - (1.0 if index == target_index else 0.0)
             for index, p in enumerate(probabilities)]
assert isclose(loss_at(logits), 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
assert isclose(sum(gradients), 0.0, abs_tol=1e-12)
assert gradients[0] < 0.0 and all(g > 0.0 for g in gradients[1:])
print("교육용 정답 토큰:", target)
print(f"기준 loss: {loss_at(logits):.4f}")
epsilon = 1e-5
for index, (token, probability, gradient) in enumerate(
        zip(vocabulary, probabilities, gradients, strict=True)):
    plus, minus = logits[:], logits[:]
    plus[index] += epsilon
    minus[index] -= epsilon
    numerical = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon)
    assert isclose(gradient, numerical, rel_tol=0.0, abs_tol=1e-8)
    print(f"{token}: 확률={probability:.4f}, 기울기={gradient:+.4f}, 수치검사={numerical:+.4f}")
assert logits == [1.61332902229994, 2.9657182847837945, -1.61332902229994]
print("검사 통과: 출력 점수의 기울기와 수치검사가 일치합니다. 가중치는 수정하지 않았습니다")
