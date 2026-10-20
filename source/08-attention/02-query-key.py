from math import exp, isclose, sqrt

# 교육용 Q·K를 직접 지정합니다. 학습된 벡터가 아닙니다.
tokens = ["gpu", "current", "status"]
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
cases = [
    ("Q1", [1.0, 0.0], [1.0, 0.0, 1.0], 0, 1),
    ("Q2", [0.0, 1.0], [0.0, 1.0, 1.0], 1, 0),
]
print("Key 순서:", tokens)
for label, query, expected_scores, higher, lower in cases:
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    assert scores == expected_scores
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert len(weights) == len(keys)
    assert all(0.0 < weight < 1.0 for weight in weights)
    assert isclose(sum(weights), 1.0)
    assert weights[higher] > weights[lower]
    assert isclose(weights[higher], weights[2])
    assert isclose(weights[higher] / weights[lower], exp(1.0 / sqrt(2.0)))
    print(f"{label}: {query}")
    print("  내적 점수:", scores)
    print("  크기 조정:", [round(number, 4) for number in scaled])
    print("  softmax 비중:", [round(number, 4) for number in weights])
    print(f"  비중 합: {sum(weights):.4f}")

print("검사 통과: 같은 Key라도 Query에 따라 비중이 달라집니다")
