from math import exp, isclose, sqrt

# 앞 실습과 같은 교육용 입력·계수입니다. 학습된 값이 아닙니다.
tokens = ["gpu", "current", "status"]
inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
wq = [[1.0, 0.0], [1.0, 1.0]]
wk = [[0.0, 1.0], [1.0, 0.0]]
wv = [[2.0, 0.0], [0.0, 3.0]]


def project(x, matrix):
    # ponytail: 2차원 행벡터 전용; 차원 확장은 PyTorch 행렬곱으로 전환합니다.
    return [x[0] * matrix[0][column] + x[1] * matrix[1][column]
            for column in range(2)]


queries = [project(x, wq) for x in inputs]
keys = [project(x, wk) for x in inputs]
values = [project(x, wv) for x in inputs]
assert queries == [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
assert keys == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
assert values == [[2.0, 0.0], [0.0, 3.0], [2.0, 3.0]]
# 앞쪽 토큰만으로 별도 계산한 검사 기준입니다. 출력 계산에는 사용하지 않습니다.
expected_weights = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                    [0.140029245043378, 0.28399540974126, 0.575975345215362]]
expected_outputs = [[2.0, 0.0], [1.0, 1.5], [1.43200918051748, 2.579912264869866]]
print("참고할 토큰 순서:", tokens)
for position, (token, query) in enumerate(zip(tokens, queries, strict=True)):
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    scaled = [score / sqrt(len(query)) for score in scores]
    masked = [score if key_position <= position else float("-inf")
              for key_position, score in enumerate(scaled)]
    exps = [exp(score - max(masked)) for score in masked]
    weights = [number / sum(exps) for number in exps]
    assert isclose(sum(weights), 1.0)
    assert all(weight == 0.0 for weight in weights[position + 1:])
    assert all(weight > 0.0 for weight in weights[:position + 1])
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(weights, expected_weights[position], strict=True))
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert all(isclose(a, b, abs_tol=1e-12)
               for a, b in zip(output, expected_outputs[position], strict=True))
    print(f"{token}: 허용={tokens[:position + 1]}")
    print("  mask 적용 점수:", [round(number, 4) for number in masked])
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 각 위치는 자신과 앞 토큰만 참고하고 뒤 토큰의 비중은 0입니다")
