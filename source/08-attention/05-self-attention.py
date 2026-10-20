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
# 별도 Decimal 고정밀 계산의 검사 기준이며, 출력 계산에는 사용하지 않습니다.
expected_scores = [[0.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
expected_outputs = [[1.1977758146404282, 2.4066725560787155],
                    [1.5034898434845539, 2.255234765226831],
                    [1.43200918051748, 2.579912264869866]]
print("참고할 토큰 순서:", tokens)
for token, query, score_check, output_check in zip(
        tokens, queries, expected_scores, expected_outputs, strict=True):
    scores = [sum(q * k for q, k in zip(query, key, strict=True)) for key in keys]
    assert scores == score_check
    scaled = [score / sqrt(len(query)) for score in scores]
    exps = [exp(score - max(scaled)) for score in scaled]
    weights = [number / sum(exps) for number in exps]
    assert len(weights) == len(tokens) and isclose(sum(weights), 1.0)
    assert all(0.0 < weight < 1.0 for weight in weights)
    output = [sum(weight * value[column] for weight, value in zip(weights, values, strict=True))
              for column in range(len(values[0]))]
    assert all(isclose(actual, target) for actual, target in zip(output, output_check, strict=True))
    print(f"{token}: Q={query}")
    print("  내적 점수:", scores)
    print("  비중:", [round(number, 4) for number in weights])
    print("  출력 벡터:", [round(number, 4) for number in output])
print("검사 통과: 입력 X에서 Q·K·V를 만들고 각 토큰의 Attention 출력을 계산했습니다")
