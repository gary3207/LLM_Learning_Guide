from math import exp, isclose

# 앞 실습 위치 2의 반올림 전 Attention 출력입니다. 여기서는 재계산하지 않습니다.
hidden = [1.61332902229994, 5.931436569567589]
# 출력 후보만 담은 교육용 어휘입니다. 앞 입력 토큰 목록과 다릅니다.
vocabulary = ["failed", "healthy", "<eos>"]
# 행벡터 hidden (1×2) × 직접 정한 출력 행렬 (2×3); bias는 없습니다.
# ponytail: 고정 교육용 계수; 학습된 모델의 예측 품질을 나타내지 않습니다.
w_vocab = [[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]]
assert len(w_vocab) == len(hidden)
assert all(len(row) == len(vocabulary) for row in w_vocab)
logits = [sum(value * row[column] for value, row in zip(hidden, w_vocab, strict=True))
          for column in range(len(vocabulary))]
exps = [exp(score - max(logits)) for score in logits]
probabilities = [number / sum(exps) for number in exps]
expected_logits = [1.61332902229994, 2.9657182847837945, -1.61332902229994]
# 독립 Decimal 계산의 검사 기준이며 확률 계산에는 쓰지 않습니다.
expected_probabilities = [0.20381779795457093, 0.7880926900858558, 0.008089511959573232]
assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(logits, expected_logits, strict=True))
assert all(isclose(a, b, abs_tol=1e-12)
           for a, b in zip(probabilities, expected_probabilities, strict=True))
assert all(0.0 < probability < 1.0 for probability in probabilities)
assert isclose(sum(probabilities), 1.0)
chosen = max(range(len(vocabulary)), key=probabilities.__getitem__)
assert vocabulary[chosen] == "healthy"
print("Attention 정보 벡터:", hidden)
for token, score, probability in zip(vocabulary, logits, probabilities, strict=True):
    print(f"{token}: 점수={score:.4f}, 확률={probability:.4f}")
print(f"확률 합: {sum(probabilities):.4f}")
print("가장 큰 확률의 후보:", vocabulary[chosen])
print("검사 통과: 정보 벡터를 어휘 점수와 확률로 바꿨습니다. 학습·문장 생성은 하지 않았습니다")
