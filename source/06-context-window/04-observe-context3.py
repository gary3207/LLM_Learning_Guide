# 원문 Python 본문 보존; 출처 주석만 추가.
sentences = [
    "gpu current status failed",
    "network current status healthy",
]

for size in (2, 3):
    counts = {}
    for sentence in sentences:
        tokens = sentence.split()
        context = tuple(tokens[-(size + 1):-1])
        target = tokens[-1]
        counts.setdefault(context, {})
        counts[context][target] = counts[context].get(target, 0) + 1

    print(f"[마지막 {size}토큰으로 최종 단어 예측]")
    for context, targets in counts.items():
        print(" ".join(context), "->", targets)

    correct = sum(max(targets.values()) for targets in counts.values())
    print(f"고정된 최빈값 예측의 최대 정답 수: {correct}/{len(sentences)}")
    assert correct == (1 if size == 2 else 2)
    print()

print("검사 통과: 두 토큰 입력 충돌, 세 토큰 입력 구분")
