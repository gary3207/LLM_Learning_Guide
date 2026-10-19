from pathlib import Path

original = Path("train_classifier.py")
target = Path("train_classifier_augmented.py")
source = original.read_text(encoding="utf-8")

changes = [
    (
        '    ("connection timeout", 1),',
        '    ("connection timeout", 1),\n'
        '    ("network unavailable", 1),',
    ),
    (
        '/ "classifier.pt"',
        '/ "classifier_augmented.pt"',
    ),
]

for old, new in changes:
    assert source.count(old) == 1, f"변경 위치를 확인하세요: {old}"
    source = source.replace(old, new, 1)

# 파일을 만들기 전에 Python 문법 검사
compile(source, str(target), "exec")

# 기존 파일이 있으면 덮어쓰지 않음
with target.open("x", encoding="utf-8") as file:
    file.write(source)

print(f"생성 완료: {target.resolve()}")
print("원본 코드와 classifier.pt는 변경하지 않았습니다.")
