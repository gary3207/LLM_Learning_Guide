from math import exp, isclose, sqrt

# 앞 실습과 같은 교육용 토큰·위치 벡터입니다. 학습된 값이 아닙니다.
tokens = ["gpu", "current", "gpu"]
token_vectors = {"gpu": [1.0, 0.0], "current": [0.0, 1.0]}
position_vectors = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
inputs = [token_vectors[token][:] for token in tokens]
combined = [[x + p for x, p in zip(vector, position, strict=True)]
            for vector, position in zip(inputs, position_vectors, strict=True)]
assert combined == [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]
# 이전 QKV 실습과 같은 변환 행렬입니다.
wq = [[1.0, 0.0], [1.0, 1.0]]
wk = [[0.0, 1.0], [1.0, 0.0]]
wv = [[2.0, 0.0], [0.0, 3.0]]


def project(x, matrix):
    # ponytail: 2차원 행벡터 전용; 차원 확장은 PyTorch 행렬곱으로 전환합니다.
    return [x[0] * matrix[0][column] + x[1] * matrix[1][column]
            for column in range(2)]


queries = [project(x, wq) for x in combined]
keys = [project(x, wk) for x in combined]
values = [project(x, wv) for x in combined]
assert queries == [[1.0, 0.0], [2.0, 2.0], [3.0, 2.0]]
assert keys == [[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
assert values == [[2.0, 0.0], [0.0, 6.0], [2.0, 6.0]]
# 허용된 앞쪽 토큰만의 독립 Decimal 계산 기준이며 출력 계산에는 쓰지 않습니다.
expected_weights = [[1.0, 0.0, 0.0], [0.1955703174930431, 0.8044296825069569, 0.0],
                    [0.011427238405401788, 0.19333548885002997, 0.7952372727445682]]
expected_outputs = [[2.0, 0.0], [0.3911406349860862, 4.826578095041741],
                    [1.61332902229994, 5.931436569567589]]
print("참고할 위치·토큰:", list(enumerate(tokens)))
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
    print(f"위치 {position}: {token}, 합친 입력={combined[position]}")
    print(f"  Q={query}, K={keys[position]}, V={values[position]}")
    print("  내적 점수:", scores)
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 위치 포함 입력에서 Q·K·V와 causal Attention 출력을 계산했습니다")
