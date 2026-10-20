# 교육용 입력과 변환 계수입니다. 학습된 값이 아닙니다.
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
for token, x, q, k, v in zip(tokens, inputs, queries, keys, values, strict=True):
    print(f"{token}: X={x} -> Q={q}, K={k}, V={v}")
print("검사 통과: 같은 입력에서 서로 다른 선형 변환으로 Q·K·V를 만들었습니다")
