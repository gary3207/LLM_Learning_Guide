# 원문 Python 본문 보존; 출처 주석만 추가.
sentences = [
    "gpu memory error",
    "network connection error",
    "disk storage full",
]

pairs = []
for sentence in sentences:
    tokens = sentence.split() + ["<eos>"]
    pairs.extend(zip(tokens[:-1], tokens[1:]))

assert len(pairs) == 9
assert pairs[0] == ("gpu", "memory")
assert sum(target == "<eos>" for _, target in pairs) == len(sentences)
assert ("<eos>", "network") not in pairs

for current, target in pairs:
    print(f"{current:>10} -> {target}")
print(f"학습 쌍: {len(pairs)}개")
print("검사 통과: 문장별 종료 토큰 포함, 문장 사이 연결 없음")
