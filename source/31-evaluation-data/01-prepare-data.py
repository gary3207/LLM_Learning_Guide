"""합성 규칙의 학습·개발·보류 테스트를 준비합니다. 모델 실행은 없습니다."""
import argparse
import json
from pathlib import Path

LABELS = {"gpu": "failed", "network": "healthy"}
# Split whole wording templates, not individual device rows.
TEMPLATES = {
    "train": [
        "Device {device} status? Reply with one word.",
        "State the {device} status. Reply with one word.",
        "For device {device}, give its status in one word.",
        "Device: {device}. Status (one word):",
    ],
    "dev": [
        "Give one word for the status of {device}.",
        "What status applies to {device}? Answer with one word.",
    ],
    "test": [
        "Provide a single-word status for {device}.",
        "Return only the status label of {device}.",
    ],
}
# Previously inspected probe/diagnostic wording from chapter 19; never fresh test.
LEGACY_TEMPLATES = [
    "What is the {device} status? Reply with one word.",
    "{device} status? Reply with one word.",
    "Report the {device} status in one word.",
]


def build_dataset():
    splits = {}
    for split, templates in TEMPLATES.items():
        splits[split] = [
            {"template_id": f"{split}-{index}", "device": device,
             "prompt": template.format(device=device), "answer": answer}
            for index, template in enumerate(templates, 1)
            for device, answer in LABELS.items()
        ]
    legacy = []
    for index, template in enumerate(LEGACY_TEMPLATES):
        for device, answer in LABELS.items():
            name = ("GPU" if device == "gpu" else "Network") if index == 1 else device
            legacy.append({"prompt": template.format(device=name), "answer": answer})
    return {
        "schema_version": 1,
        "task": "synthetic device-to-label mapping; wording holdout only",
        "scoring": "strip-only, case-sensitive exact match",
        "splits": splits,
        "legacy_dev": legacy,
    }


def validate(data):
    # ponytail: case/whitespace checks and declared template groups, not semantic deduplication.
    seen = {" ".join(row["prompt"].casefold().split()) for row in data["legacy_dev"]}
    groups = {}
    if set(data["splits"]) != {"train", "dev", "test"}:
        raise ValueError("train/dev/test 분리가 필요합니다.")
    for split, rows in data["splits"].items():
        if not rows:
            raise ValueError(f"빈 split: {split}")
        for row in rows:
            device, prompt = row["device"], row["prompt"]
            if device not in LABELS or row["answer"] != LABELS[device]:
                raise ValueError("장치·합성 정답 불일치")
            key = " ".join(prompt.casefold().split())
            if not key or key in seen:
                raise ValueError("빈 문장 또는 기존 문항/split 간 중복")
            seen.add(key)
            template_id = row["template_id"]
            if not isinstance(template_id, str) or not template_id:
                raise ValueError("문장 틀 ID가 필요합니다.")
            pattern = " ".join(prompt.replace(device, "{device}").casefold().split())
            owner, previous, devices = groups.setdefault(template_id, (split, pattern, set()))
            if owner != split or previous != pattern or device in devices:
                raise ValueError("문장 틀 분리·대응 불일치")
            devices.add(device)
    if any(devices != set(LABELS) for _, _, devices in groups.values()):
        raise ValueError("각 문장 틀에는 두 장치가 모두 필요합니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation_splits_v1.json"))
    args = parser.parse_args()
    data = build_dataset()
    validate(data)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    for split, rows in data["splits"].items():
        print(f"{split}: {len(rows)} rows, {len(TEMPLATES[split])} templates")
    print(f"legacy_dev: {len(data['legacy_dev'])} rows (not fresh test)")
    print("CHECK=PASS (duplicates, labels, template groups)")
    print("SAVED", args.output.name)
    print("MODEL_EXECUTED=False TRAINING_EXECUTED=False")


if __name__ == "__main__":
    main()
