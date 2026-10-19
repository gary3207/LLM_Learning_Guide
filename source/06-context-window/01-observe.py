# 원문 Python 본문 보존; 출처 주석만 추가.
from collections import Counter

sentences = ["gpu status failed", "network status healthy"]
one_token = {}
two_tokens = {}

for sentence in sentences:
    first, current, target = sentence.split()
    one_token.setdefault(current, Counter())[target] += 1
    two_tokens.setdefault((first, current), Counter())[target] += 1

print("[마지막 한 토큰만 사용]")
for context, counts in one_token.items():
    print(f"{context} -> {dict(counts)}")

print("\n[앞의 두 토큰 사용]")
for context, counts in two_tokens.items():
    print(f"{' '.join(context)} -> {dict(counts)}")

one_correct = sum(max(counts.values()) for counts in one_token.values())
two_correct = sum(max(counts.values()) for counts in two_tokens.values())

assert one_token["status"] == Counter({"failed": 1, "healthy": 1})
assert one_correct == 1 and two_correct == 2

print(f"\n한 토큰 입력의 최대 정답 수: {one_correct}/2")
print(f"두 토큰 입력의 최대 정답 수: {two_correct}/2")
print("검사 통과: 한 토큰 입력 충돌, 두 토큰 입력 구분")
