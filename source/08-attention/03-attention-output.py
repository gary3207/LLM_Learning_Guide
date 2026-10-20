from math import exp, isclose, sqrt

# 교육용 Q·K·V입니다. Key는 비교용, Value는 섞을 정보입니다.
tokens = ["gpu", "current", "status"]
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
values = [[10.0, 0.0], [0.0, 10.0], [1.0, 1.0]]
cases = [
    ("Q1", [1.0, 0.0], [4.412233019477645, 2.378870239084068]),
    ("Q2", [0.0, 1.0], [2.378870239084068, 4.412233019477645]),
]
assert len(tokens) == len(keys) == len(values)
assert all(len(vector) == 2 for vector in keys + values)
print("토큰 순서:", tokens)
print("Value:", values)
for label, query, expected in cases:
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert isclose(sum(weights), 1.0)
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert len(output) == len(values[0])
    assert all(isclose(actual, target) for actual, target in zip(output, expected, strict=True))
    print(f"{label}: {query}")
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])

print("검사 통과: 같은 Key·Value도 Query에 따라 출력 벡터가 달라집니다")
