from math import isclose

# 교육용 숫자입니다. 학습된 임베딩이나 단어의 실제 의미가 아닙니다.
tokens = ["gpu", "current", "status"]
values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
flattened = [number for vector in values for number in vector]
assert flattened == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
print("토큰:", tokens)
print("펼친 입력:", flattened)

# ponytail: 가중치를 직접 지정; 다음 단계에서 Query·Key로 계산합니다.
cases = [
    ("gpu 쪽 비중", [0.8, 0.1, 0.1], [0.9, 0.2]),
    ("current 쪽 비중", [0.1, 0.8, 0.1], [0.2, 0.9]),
]
for label, weights, expected in cases:
    assert len(weights) == len(values) and isclose(sum(weights), 1.0)
    assert all(weight >= 0 for weight in weights)
    weighted = [sum(weight * vector[column] for weight, vector in zip(weights, values, strict=True))
                for column in range(len(values[0]))]
    assert len(weighted) == 2
    assert all(isclose(actual, target) for actual, target in zip(weighted, expected, strict=True))
    print(f"{label}: {weights} -> {[round(number, 3) for number in weighted]}")

print("검사 통과: 같은 입력도 가중치에 따라 가중합이 달라집니다")
