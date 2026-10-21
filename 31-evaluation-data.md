# 31 · 독립 평가 데이터 설계

[← 이전: Docker 실행환경](30-container-runtime.md) · [목차](README.md) · [학습 점검](07-resume.md)

## 1. 목표와 이번 범위

**학습용 train, 개선 판단용 dev, 마지막 확인용 test를 나누고, 겹침을 검사한 JSON을 만든다.** 모델 학습·추론·품질 게이트 변경은 하지 않는다. GPU나 새 패키지도 필요 없다.

기존17~20장과 같은 합성 규칙을 사용한다.

| 장치 | 정답 |
|---|---|
| gpu | `failed` |
| network | `healthy` |

이는 장비 상태를 측정한 정답이 아니다. 같은 장치에 실제로 다른 장애 상태가 발생하는 과제도 아니다. **알려진 두 장치와 고정 라벨의 대응을 새로운 질문 표현에서도 출력하는지** 확인할 준비다. 원본 모델에 이 임의 규칙을 알려주지 않고 틀렸다는 이유만으로 실제 진단 능력이 없다고 해석하지 않는다.

## 2. train/dev/test를 왜 나누는가?

| 데이터 | 하는 일 | 하지 않는 일 |
|---|---|---|
| train | 정답으로 loss를 계산하고 가중치를 갱신 | 이 점수를 새로운 입력의 성능으로 발표 |
| dev | 후보의 출력·오류를 보고 데이터/설정 개선과 후보 선택 | 그 정답으로 직접 역전파 |
| test | 개선 방법과 후보를 정한 뒤 마지막 확인 | 결과를 반복해서 보면서 설정 선택 |
| legacy_dev | 앞서 결과를 본 probe·diagnostic의 비교용 보관 | 새 독립 test로 이름만 변경 |

비유하면 train은 연습문제, dev는 공부 방법을 고르는 모의고사, test는 방법을 정한 뒤 보는 시험이다. dev도 개선 방향에 영향을 주므로 최종 test와 역할이 다르다. test 결과를 보고 다시 개선하면 그 test는 이후 개발에 사용한 데이터다. 당시 결과는 보존하고 더 이상 새 독립 평가라고 부르지 않는다. 이를 막는 별도 보안 장치는 이번에 만들지 않는다.

**문장을 무작위로 섞지 않고 문장 틀 전체를 분리한다.** 예를 들어 `Device {device} status? Reply with one word.`에서 gpu와 network를 넣은 두 질문은 같은 train에 둔다. 한쪽은 train, 다른 쪽은 test로 보내면 동일한 표현을 이미 연습한 셈이 된다.

| 묶음 | 문장 틀 수 | 문항 수 | 라벨 구성 |
|---|---:|---:|---|
| train | 4 | 8 | failed 4 / healthy 4 |
| dev | 2 | 4 | failed 2 / healthy 2 |
| test | 2 | 4 | failed 2 / healthy 2 |
| legacy_dev | 3 | 6 | 이전 probe 2개·diagnostic 4개 |

train에는 기존 학습 질문 두 개가 포함된다. 새 train/dev/test의 문항 수는 합계16개이고 legacy_dev6개는 별도다. 기존 19장 여덟 문항은 학습2개와 기존 진단6개로 보존되며 새 test에 넣지 않는다. 비율은 이 교육용 예제의 선택이며 일반적인 정답 비율이 아니다.

### 여기서 말하는 독립성의 한계

- 새 dev/test 표현은 이번 추가 학습용 train과 문장 틀이 다르고, 아직 이 실습에서 모델 출력을 보고 선택하지 않은 보류 문항이다.
- 문장 틀 ID와 문자열 중복 검사는 **의미상 유사성까지 제거하지 않는다.** 모든 문항은 같은 두 장치·같은 합성 규칙을 공유한다. 새로운 장치·실제 사건·넓은 일반화 평가는 아니다.
- test는 코드/JSON에서 읽을 수 있다. 접근이 차단된 블라인드 시험이 아니며, 사전학습 데이터에 같은 문장이 없었다고 증명하지 않는다. 개발 중 test 응답을 보며 튜닝하지 않는 사용 절차가 중요하다.
- test4개만으로 운영 품질을 보장하지 않는다. 이후 점수는 전체 합계 하나로 섞지 않고 split별 정답 수/문항 수로 보고한다.

## 3. 실행 위치와 결과 파일

기존 **Ubuntu VM의 `~/llm-lab/source`**에서 실행한다. `source .venv/bin/activate`는 기존 Python 환경을 선택한다. `python --version`은 실제 사용할 인터프리터를 확인한다. 설치·모델 로딩·서버 시작은 없다.

- 새 코드: `~/llm-lab/source/prepare_evaluation_data.py`
- 새 데이터: `~/llm-lab/source/evaluation_splits_v1.json`
- 기존 모델·레지스트리·보고서·25장 품질 판정은 변경하지 않는다.

아래 블록 전체를 한 번 실행한다. `(set -C; cat > ... )`는 기존 코드 덮어쓰기를 거부한다. Python의 `open("x")`도 기존 JSON 덮어쓰기를 거부한다. `&&`는 앞 단계가 실패하면 다음 단계를 실행하지 않는다. 재실행 시 기존 파일을 지우지 말고 보존한다. 다시 만들 목적이 있을 때만 코드와 출력에 각각 새 이름을 쓴다.

정본: [준비 코드](source/31-evaluation-data/01-prepare-data.py) · [데이터 예시](source/31-evaluation-data/evaluation_splits_v1.json) · [작은 검사 코드](source/31-evaluation-data/test_dataset.py)

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
(set -C; cat > prepare_evaluation_data.py <<'PY'
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
PY
) &&
python -B prepare_evaluation_data.py --output evaluation_splits_v1.json
```

## 4. 코드 읽기

1. **`LABELS`**: 합성 정답표다. 모델의 예측이나 실제 장비 상태가 아니다.
2. **`TEMPLATES`**: split을 문장 틀 단위로 먼저 고정한다. 각 틀에 두 장치를 넣어 라벨 균형을 맞춘다. 난수로 다시 나누지 않으므로 실행마다 같은 내용을 만든다.
3. **`build_dataset()`**: 각 문항에 `template_id`, `device`, `prompt`, `answer`를 저장한다. `answer`는 나중에 학습/채점할 정답이지 생성 질문에 붙일 힌트가 아니다. 기존 probe·diagnostic은 `legacy_dev`에 따로 둔다.
4. **`validate()`**: 빈 split·중복 질문·장치/정답 불일치·한 틀의 split 분산·두 장치 대응 누락을 검사한다. 중복 탐지에는 대소문자와 연속 공백을 정규화하지만, **실제 질문이나 생성 응답을 고치지는 않는다.** 의미상 중복을 찾는 알고리즘이나 범용 JSON 스키마 검증기는 아니다.
5. **저장과 출력**: 검사 후 새 JSON을 만들고 문항 수만 출력한다. test 응답을 생성하거나 test 점수를 계산하지 않는다. 디스크 쓰기 중 실패하면 새 파일 일부가 남을 수 있으며 자동 삭제하지 않는다.

예를 들어 train 한 문항은 다음 구조다.

```json
{
  "template_id": "train-1",
  "device": "gpu",
  "prompt": "Device gpu status? Reply with one word.",
  "answer": "failed"
}
```

### 나중에 적용할 채점 기준

19장 기준을 유지한다. 생성 문자열의 앞뒤 공백만 제거하고, 정답과 대소문자를 구분해 완전히 일치해야 정답이다. ` failed\n`은 정답이고 `FAILED`, `failed.`, `The status is failed.`는 오답이다. 데이터 중복 탐지용 정규화를 응답 채점에 적용하지 않는다.

다음 품질 개선 실습에서는 train으로만 가중치를 갱신하고 dev로 오류를 비교한다. test는 후보/설정을 고른 뒤 한 번의 최종 비교 묶음에 사용한다. 서로 다른 모델을 비교할 때는 같은 질문·채점·생성 조건을 사용하고, QLoRA의 양자화 조건 차이는 따로 명시한다. **이번 장은 그 평가나 추가 학습을 아직 실행하지 않는다.**

## 5. 실제 로컬 실행 예시

아래는 모델을 실행하지 않고 위 준비 코드를 로컬 Python에서 실행해 얻은 출력이다. VM의 데이터 준비 완료는 VM 실행 출력으로 별도 확인한다.

```text
train: 8 rows, 4 templates
dev: 4 rows, 2 templates
test: 4 rows, 2 templates
legacy_dev: 6 rows (not fresh test)
CHECK=PASS (duplicates, labels, template groups)
SAVED evaluation_splits_v1.json
MODEL_EXECUTED=False TRAINING_EXECUTED=False
```

`CHECK=PASS`는 데이터 구조와 제한된 중복/그룹 검사 통과라는 뜻이다. 모델이16개를 맞혔다는 뜻도, 기존 품질0/8이 개선됐다는 뜻도 아니다.

저장소 개발용 검사는 저장소 루트에서 다음과 같이 실행한다. VM에서 위 단일 파일만 만든 경우에는 이 저장소용 테스트 파일이 없어도 준비 코드 자체의 검사가 실행된다.

```bash
python -B source/31-evaluation-data/test_dataset.py
python source/check_guide.py
```

실제 검사는 라벨 균형·19장 기존 문항 보존·JSON 읽기 일치·덮어쓰기 거부와, 잘못된 데이터6종의 거부를 확인했다. 모델 가중치·optimizer·CUDA 경로는 검사 대상이 아니다.

## 6. 복습과 다음 묶음

- [ ] train의 정답으로 가중치를 바꾸는 것과 dev로 후보를 고르는 것을 구분한다.
- [ ] 이미 본 평가 문항을 새 test로 이름만 바꿀 수 없는 이유를 설명한다.
- [ ] 문장 틀 분리와 의미상 독립성·실제 장애 사건 분리가 다름을 설명한다.
- [ ] 데이터 검사 성공과 모델 정답률을 구분한다.
- [ ] 기존 품질 보고서를 덮어쓰거나 합격 기준을 낮추지 않는다.

다음 묶음은 **모델 품질 개선**이다. 이번 JSON을 입력으로 연결하고, dev 오류를 보며 한 조건씩 비교하는 학습·평가를 진행한다. 이번 장에서 자동으로 시작하지 않는다. RAG는 추가하지 않는다.
