from math import isclose, isfinite, log

# 앞 출력층의 반올림 전 확률을 고정 입력으로 가져옵니다. 재계산하지 않습니다.
vocabulary = ["failed", "healthy", "<eos>"]
probabilities = [0.20381779795457093, 0.7880926900858558, 0.008089511959573232]
# ponytail: 세 후보·한 위치 전용 산술; 실제 학습에서는 logits와 CrossEntropyLoss를 사용합니다.
assert vocabulary == ["failed", "healthy", "<eos>"]
assert len(probabilities) == len(vocabulary)
assert all(isfinite(p) and 0.0 < p <= 1.0 for p in probabilities)
assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)
# 정답은 argmax가 아니라 데이터가 제공합니다. 여기서는 설명을 위해 failed로 정합니다.
target = "failed"
assert target in vocabulary
target_index = vocabulary.index(target)
target_probability = probabilities[target_index]
loss = -log(target_probability)
assert isclose(loss, 1.5905288315131196, rel_tol=0.0, abs_tol=1e-12)
assert isclose(-log(1.0), 0.0, abs_tol=1e-12)
assert -log(0.8) < -log(0.2)
print("교육용 정답 토큰:", target)
print(f"정답에 부여한 확률: {target_probability:.4f}")
print(f"한 위치의 loss = -ln(정답 확률): {loss:.4f}")
print("같은 분포에서 정답만 달리 가정한 비교 (학습 전후 아님):")
# 독립 Decimal 계산 기준이며 loss 계산에 사용하지 않습니다.
expected_losses = [1.5905288315131196, 0.23813956902926515, 4.817186876113]
for token, probability, expected in zip(vocabulary, probabilities, expected_losses, strict=True):
    candidate_loss = -log(probability)
    assert isclose(candidate_loss, expected, rel_tol=0.0, abs_tol=1e-12)
    print(f"  정답={token}: 확률={probability:.4f}, loss={candidate_loss:.4f}")
print("검사 통과: 정답 확률에서 loss를 계산했습니다. 역전파·가중치 수정은 하지 않았습니다")
