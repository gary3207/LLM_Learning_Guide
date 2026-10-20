# 교육용 토큰·위치 벡터입니다. 학습된 임베딩이 아닙니다.
# 같은 토큰을 비교하려고 앞 실습의 문장을 gpu current gpu로 바꿉니다.
tokens = ["gpu", "current", "gpu"]
token_vectors = {"gpu": [1.0, 0.0], "current": [0.0, 1.0]}
# ponytail: 위치 세 개 전용 수동 표; 실제 모델에서는 위치 표현 방식을 별도로 정합니다.
position_vectors = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
inputs = [token_vectors[token][:] for token in tokens]
combined = [[x + p for x, p in zip(vector, position, strict=True)]
            for vector, position in zip(inputs, position_vectors, strict=True)]

assert inputs == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
assert combined == [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]
assert inputs[0] == inputs[2]
assert combined[0] != combined[2]
assert all(len(vector) == 2 for vector in combined)
for index, (token, vector, position, result) in enumerate(
        zip(tokens, inputs, position_vectors, combined, strict=True)):
    print(f"위치 {index}: {token}")
    print(f"  토큰 벡터: {vector}, 위치 벡터: {position}")
    print("  합친 입력:", result)
print("검사 통과: 같은 gpu 벡터도 위치 벡터를 더하면 입력이 달라집니다")
