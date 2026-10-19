import ast
import json
from pathlib import Path

# 학습 파일을 실행하지 않고 data 값만 읽습니다.
tree = ast.parse(Path("train_classifier_augmented.py").read_text(encoding="utf-8"))
assignments = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "data" for t in node.targets)]
assert len(assignments) == 1, "data 정의를 확인하세요."
train = ast.literal_eval(assignments[0].value)

splits = {
    "train": train,
    "validation": [
        ("network error", 1),
        ("connection unavailable", 1),
        ("disk unavailable", 2),
        ("cuda error", 0),
    ],
    "test": [
        ("cuda device error", 0),
        ("network connection unavailable", 1),
        ("disk space full", 2),
    ],
}

seen = set()
train_vocab = {word for text, _ in train for word in text.split()}
for name, examples in splits.items():
    assert examples, f"빈 데이터: {name}"
    for text, label in examples:
        key = " ".join(text.lower().split())
        assert key and key not in seen, f"빈 문장 또는 중복 문장: {text}"
        assert type(label) is int and label in (0, 1, 2), f"잘못된 라벨: {label}"
        assert set(key.split()) <= train_vocab, f"학습 어휘에 없는 단어: {text}"
        seen.add(key)

payload = {"labels": ["GPU", "네트워크", "스토리지"], **splits}
with Path("classifier_splits.json").open("x", encoding="utf-8") as file:
    json.dump(payload, file, ensure_ascii=False, indent=2)

for name, examples in splits.items():
    print(f"{name}: {len(examples)}개")
print("검사 통과: 문장 중복 없음, 라벨 유효, 평가 문장은 학습 어휘만 사용")
print("저장 완료:", Path("classifier_splits.json").resolve())
