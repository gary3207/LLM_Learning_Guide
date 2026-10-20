# 19 · 원본·SFT·LoRA를 같은 기준으로 평가하기

[← 이전: GPU·LoRA](18-lora-gpu.md) · [목차](README.md) · [다음: QLoRA →](20-qlora.md)

## 1. 이번에 완성할 결과물

18장은 학습·저장·복원이 성공해도 정답 생성은 실패할 수 있음을 보여줬습니다. 이번에는 **기존 원본·전체 SFT·LoRA를 재학습 없이 불러오기 → 같은 입력·생성 조건으로 채점 → 문항별 오류와 그룹별 정답률/loss를 JSON에 저장**합니다. 모델을 고치거나 배포 후보를 자동 선택하지 않습니다.

- 입력: 원본 모델 캐시, 17장 `smollm2_sft_v1/`, 18장 `smollm2_lora_v1/`.
- 코드: VM의 `~/llm-lab/source/smollm2_evaluate.py`.
- 새 결과: `~/llm-lab/smollm2_evaluation_v1.json`.
- 기존 모델·어댑터는 읽기만 합니다. optimizer·backward·학습·체크포인트 저장은 없습니다.

## 2. 먼저 평가 기준을 고정하기

**정답은 실제 장비 상태가 아니라 이전 학습과 같은 합성 규칙 `gpu → failed`, `network → healthy`입니다.** 원본 모델에는 이 규칙을 질문에 추가로 알려주지 않습니다. 따라서 원본의 오답은 실제 장비 진단 실패를 뜻하지 않습니다. 세 모델의 입력을 똑같이 유지합니다.

| 그룹 | 입력 | 정답 | 의미 |
|---|---|---|---|
| train | `Device gpu status? Reply with one word.` | failed | 이전 학습 질문 재현 |
| train | `Device network status? Reply with one word.` | healthy | 이전 학습 질문 재현 |
| probe | `What is the gpu status? Reply with one word.` | failed | 이미 결과를 본 진단 질문 |
| probe | `What is the network status? Reply with one word.` | healthy | 이미 결과를 본 진단 질문 |
| diagnostic | `GPU status? Reply with one word.` | failed | 대소문자·표현 변경 |
| diagnostic | `Network status? Reply with one word.` | healthy | 대소문자·표현 변경 |
| diagnostic | `Report the gpu status in one word.` | failed | 요청 표현 변경 |
| diagnostic | `Report the network status in one word.` | healthy | 요청 표현 변경 |

추가 네 질문은 이번 코드의 첫 실행 전에 지정합니다. 다만 같은 기기·정답을 공유하는 작은 표현 변경이며, **독립 최종 test나 넓은 일반화 평가가 아닙니다.** 모두 진단 데이터로 다루고, 이전에 본 probe를 새 test라고 이름만 바꾸지 않습니다. 그룹을 섞은 전체 정답률로 학습 질문의 성공을 부풀리지 않습니다.

### 생성 결과 채점

생성 문자열의 **앞뒤 공백만 제거하고 대소문자를 구분해 완전히 일치**해야 정답입니다. `failed`는 정답이지만 `FAILED`, `failed.`, `The GPU status is failed.`는 오답입니다. 단어가 포함됐다는 이유로 성공 처리하지 않습니다.

| 분류 | 규칙 | 예시: 정답 failed |
|---|---|---|
| correct | 공백 제거 후 완전 일치 | ` failed\n` |
| wrong_label | 다른 합성 라벨만 생성 | `healthy` |
| extra_text | 완전 일치는 아니지만 정답 단어가 단어 경계로 포함됨 | `"failed"`, `failed.` |
| other | 위 조건에 들지 않음 | 빈 응답, `FAILED`, `Device GPU status: 1` |

이는 결정적인 문자열 분류이지 의미 이해 판정이 아닙니다. `not failed`도 extra_text이고 정답은 아닙니다. 문항별 원문을 보면서 틀린 상태·형식 위반·무관한 응답을 해석합니다.

### loss와 정답률은 서로 다른 측정

- **정답률:** 질문만 주고 모델이 스스로 생성한 응답을 채점합니다.
- **문항 loss:** 정답 응답을 입력에 포함해 teacher forcing으로 계산하되, 질문 labels는 `-100`으로 무시합니다. 응답 토큰과 첫 eos가 평가 대상입니다.
- **그룹 평균 loss:** 문항별 응답/eos 평균 loss를 다시 문항 수로 나눈 값입니다. 모든 문항의 토큰을 합친 가중 평균이 아닙니다.

정답의 토큰 수가 다르면 문항 내부에서 eos가 차지하는 비중도 달라집니다. 같은 문항·토크나이저·채점 범위끼리 비교합니다. 모델이 정답에 주는 확률이 높아져 loss가 줄더라도 greedy 생성이 정답으로 바뀌었다는 뜻은 아닙니다.

## 3. Ubuntu VM에서 실행

**실행 위치는 기존 `~/llm-lab/source`이고 가상환경은 `.venv`입니다.** 17·18장 저장 폴더와 원본 모델 캐시가 필요합니다. 기존 Transformers4.57.1·PEFT0.17.1을 사용하며 새 설치나 모델 다운로드는 하지 않습니다. 캐시가 없거나 경로가 다르면 오류를 확인하고, 평가를 위해 원본/후보를 지우거나 다시 학습하지 않습니다.

아래는 NVIDIA CUDA로 세 모델을 **모두 같은 장치·float32·eager attention·greedy·최대 새 토큰8개**로 평가하는 명령입니다. GenerationConfig 하나를 명시적으로 공유하므로 저장 모델마다 다른 생성 설정을 조용히 적용하지 않습니다. 이전 학습 장치가 달랐다는 사실까지 없애는 비교는 아닙니다.

`cat`은 코드를 새 파일에 기록하고 `(set -C; ...)`는 기존 코드 덮어쓰기를 거부합니다. `&&`는 앞 명령이 실패하면 실행을 멈춥니다. JSON도 `open("x")`로 기존 결과 덮어쓰기를 거부합니다. 같은 평가를 보관하며 다시 실행하려면 `--output ../smollm2_evaluation_v2.json`처럼 새 이름을 사용합니다. `python -O`는 검사문을 생략하므로 사용하지 않습니다.

소스: [01-evaluate-models.py](source/19-model-evaluation/01-evaluate-models.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
(set -C; cat > smollm2_evaluate.py <<'PY'
"""기존 원본·SFT·LoRA를 재학습 없이 같은 조건으로 평가하고 JSON에 저장합니다."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ROWS = [
    ("train", "Device gpu status? Reply with one word.", "failed"),
    ("train", "Device network status? Reply with one word.", "healthy"),
    ("probe", "What is the gpu status? Reply with one word.", "failed"),
    ("probe", "What is the network status? Reply with one word.", "healthy"),
    ("diagnostic", "GPU status? Reply with one word.", "failed"),
    ("diagnostic", "Network status? Reply with one word.", "healthy"),
    ("diagnostic", "Report the gpu status in one word.", "failed"),
    ("diagnostic", "Report the network status in one word.", "healthy"),
]


def classify(text, answer):
    text = text.strip()
    if text == answer:
        return "correct"
    if text in ("failed", "healthy"):
        return "wrong_label"
    if re.search(r"\b" + re.escape(answer) + r"\b", text):
        return "extra_text"
    return "other"


def summarize(records):
    groups = {}
    for group in dict.fromkeys(row["group"] for row in records):
        rows = [row for row in records if row["group"] == group]
        correct = sum(row["correct"] for row in rows)
        groups[group] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                         "mean_loss": sum(row["loss"] for row in rows) / len(rows)}
    return groups


def example(tokenizer, prompt, answer, device):
    import torch
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True)
    assert full[:len(prefix)] == prefix, "대화 형식 접두 불일치"
    end = full.index(tokenizer.eos_token_id, len(prefix)) + 1
    ids = torch.tensor([full[:end]], device=device)
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    assert (labels[:, 1:] != -100).any(), "응답/eos 정답 없음"
    return ids, labels, torch.tensor([prefix], device=device)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftConfig, PeftModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("../smollm2_sft_v1"))
    parser.add_argument("--lora", type=Path, default=Path("../smollm2_lora_v1"))
    parser.add_argument("--output", type=Path, default=Path("../smollm2_evaluation_v1.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"평가 결과 덮어쓰기 거부: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    for path in (args.sft, args.lora):
        if not path.is_dir():
            raise FileNotFoundError(f"기존 저장 폴더 없음: {path}")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True,
                                             trust_remote_code=False)
    for path in (args.sft, args.lora):
        saved = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        assert (saved.get_vocab(), saved.eos_token_id, saved.chat_template) == (
            tokenizer.get_vocab(), tokenizer.eos_token_id, tokenizer.chat_template), "토크나이저 불일치"
    adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
    assert adapter.base_model_name_or_path == MODEL_ID and adapter.revision == REVISION, "LoRA 원본/revision 불일치"
    generation = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                  bos_token_id=tokenizer.bos_token_id, use_cache=True)
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "device": args.device, "dtype": "float32",
                "torch": version("torch"), "transformers": version("transformers"), "peft": version("peft"),
                "generation": generation.to_dict(), "training_executed": False,
                "scoring": "strip-only, case-sensitive exact match; per-example mean response/eos CE",
                "sources": {"sft": str(args.sft), "lora": str(args.lora)}}
    records, summaries = [], {}
    print(f"장치: {args.device} | float32 | 재학습 없음 | 모델당 {len(ROWS)}문항")
    # ponytail: 모델을 순서대로 한 개씩 적재; 대규모 평가는 batching/전용 평가 도구로 확장합니다.
    for name in ("base", "sft", "lora"):
        location = args.sft if name == "sft" else MODEL_ID
        options = {} if name == "sft" else {"revision": REVISION}
        model = AutoModelForCausalLM.from_pretrained(
            location, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager", **options).to(args.device)
        if name == "lora":
            model = PeftModel.from_pretrained(model, args.lora, is_trainable=False, local_files_only=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for index, (group, prompt, answer) in enumerate(ROWS, 1):
                ids, labels, inputs = example(tokenizer, prompt, answer, args.device)
                loss = model(ids, labels=labels, use_cache=False).loss.item()
                assert math.isfinite(loss), "유한하지 않은 loss"
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
                text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
                error = classify(text, answer)
                records.append({"model": name, "id": index, "group": group, "prompt": prompt,
                                "answer": answer, "text": text, "correct": error == "correct",
                                "error": error, "loss": loss})
                print(f"[{name}/{group}/{index}] {error} | 정답={answer} | 생성={text!r} | loss={loss:.4f}")
        summaries[name] = summarize([row for row in records if row["model"] == name])
        for group, result in summaries[name].items():
            print(f"[{name}/{group}] 정답 {result['correct']}/{result['total']} | 평균 loss={result['mean_loss']:.4f}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "records": records, "summary": summaries}, stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"평가 저장: {args.output} | 체크포인트 저장·수정 없음")


if __name__ == "__main__":
    main()
PY
) && python smollm2_evaluate.py --device cuda
```

CPU로 평가할 때만 마지막 옵션을 `--device cpu`로 바꿉니다. CUDA 요청이 불가능하면 CPU로 자동 전환하지 않습니다. 저장 위치가 다르면 `--sft 경로 --lora 경로`를 지정합니다. 평가 결과 파일의 부모 폴더는 이미 존재해야 합니다.

## 4. 코드 읽기

1. **ROWS / classify / summarize**
   - 문항·정답·그룹을 실행 전 고정합니다. 판정은 실제 생성 문자열로 하고, 그룹별 분모를 유지합니다.
   - 오류 분류는 정답률과 별도로 JSON에 남깁니다. 정확도를 높이려고 사후에 정답이나 판정 규칙을 바꾸지 않습니다.
2. **example()**
   - 17·18장의 대화 템플릿·응답/eos 전용 labels 구성을 이어 씁니다. 과거 스크립트는 import 시 학습/복원을 실행하므로 import하지 않고 필요한 짧은 데이터 구성을 이 파일에 둡니다.
   - ids와 labels는 `(1, 전체길이)`, 생성 입력은 `(1, 질문과 assistant 시작까지의 길이)`입니다. CausalLM이 내부에서 정답을 이동하므로 labels를 다시 한 칸 옮기지 않습니다.
3. **원본·SFT·LoRA 순차 적재**
   - 원본은 고정 revision의 로컬 캐시, SFT는 수정된 전체 가중치, LoRA는 같은 원본과 어댑터를 결합합니다. 한 번에 세 모델을 GPU에 올리지 않습니다.
   - 후보 토크나이저의 어휘/ID·eos·대화 템플릿과 LoRA 원본/revision을 검사합니다. 이 검사는 학습 이력의 진위나 품질 인증은 아닙니다.
4. **eval / requires_grad_(False) / inference_mode**
   - 평가 모드로 바꾸고 gradient 계산·가중치 학습을 하지 않습니다. 과거의 `reference.pt`에 담긴 정답 예측을 읽는 대신, 모델에서 새로 loss와 생성을 계산합니다.
5. **JSON**
   - `metadata`: 버전·실행 장치·dtype·고정 revision·생성/채점 설정·후보 경로.
   - `records`: 모델별8개, 합계24개 문항의 질문·정답·생성 원문·오류 분류·정답 여부·loss.
   - `summary`: 모델별 train/probe/diagnostic의 정답 수·분모·정답률·평균 loss.
   - 결과에 체크포인트 가중치는 포함하지 않습니다. 파일 작성 도중 중단되면 새 JSON이 불완전할 수 있으므로 확인 후 별도로 처리합니다.

## 5. 검증 결과와 실행 결과 구분

**실제 CPU 통합 검사는 실행했지만, 아래 검사는 학습된 SFT·LoRA의 새 품질 비교가 아닙니다.** Windows·Python3.12·PyTorch2.9.0+cpu·Transformers4.57.1·PEFT0.17.1에서 원본과 동일한 미학습 전체모델 복사본, 변경분이0인 LoRA를 임시 폴더에 만들어 로딩 경로를 검사했습니다. 학습은 수행하지 않았고 이 fixture를 학습 모델로 보고하지 않습니다.

확인한 내용:

- 실제 모델의24개 문항 실행·JSON 생성과 동일 가중치 세 경로의 결과 동등성.
- 실제 logits로 독립 계산한 logsumexp cross-entropy와 응답/eos loss 일치.
- 문항/그룹 집계, 공백·대소문자·빈 응답·다른 라벨·추가 문구 처리.
- 입력 후보 파일 불변, 기존 JSON 덮어쓰기 거부, CUDA 없는 환경의 명시 실패.

위 검사는 코드 경로 검증입니다. 아래의 **기존 학습 후보를 사용한 CUDA 실행 예시**와 구분합니다. 각 환경의 성공 여부와 수치는 자신의 출력으로 확인하며, 이전 네 질문의 결과를 새로운 네 질문의 결과로 대신하지 않습니다.

개발 환경에서 검사를 재실행하려면 저장소 루트에서 실행합니다. 첫 명령은 표준 라이브러리만 사용합니다. 두 번째 명령은 실제 PyTorch/Transformers/PEFT와 원본 캐시가 필요하고, 임시 미학습 모델을 저장할 디스크 공간을 사용합니다. 기존 학습 폴더를 사용하거나 변경하지 않습니다.

```bash
python source/19-model-evaluation/test_evaluation.py
python source/19-model-evaluation/test_evaluation.py --integration
```

검사 소스: [test_evaluation.py](source/19-model-evaluation/test_evaluation.py).

### 기존 학습 후보의 CUDA 실행 예시

Ubuntu VM의 제공된 실행 로그에서 세 모델 모두 CUDA·float32로8문항씩 평가하고 새 JSON을 저장한 결과를 확인했습니다. 재학습 없이17장 전체 SFT와18장 LoRA 저장물을 읽은 실행입니다. 위의 미학습 fixture 결과가 아닙니다. 아래는 출력된 그룹 평균을 그대로 옮겼으며, 반올림된 문항 loss로 평균을 다시 계산한 값과 마지막 자리가 다를 수 있습니다.

| 모델 | 학습 질문 정답 / 평균 loss | 기존 probe 정답 / 평균 loss | 추가 진단 정답 / 평균 loss |
|---|---|---|---|
| 원본 | 0/2 · 12.7998 | 0/2 · 9.7420 | 0/4 · 10.5820 |
| 전체 SFT | 2/2 · 0.0138 | 2/2 · 0.1679 | 3/4 · 0.5150 |
| LoRA | 0/2 · 10.0753 | 0/2 · 7.3542 | 0/4 · 8.3529 |

<details>
<summary>문항별 실제 CUDA 출력 펼치기</summary>

```text
장치: cuda | float32 | 재학습 없음 | 모델당 8문항
[base/train/1] other | 정답=failed | 생성='Device GPU status: 1' | loss=14.4325
[base/train/2] other | 정답=healthy | 생성='Device network status:\n\n- Device' | loss=11.1671
[base/probe/3] other | 정답=failed | 생성="The GPU status is currently '0'." | loss=12.3041
[base/probe/4] other | 정답=healthy | 생성='The network status is currently stable and functioning' | loss=7.1800
[base/diagnostic/5] other | 정답=failed | 생성='GPU status:\n\nGPU status:' | loss=16.2342
[base/diagnostic/6] other | 정답=healthy | 생성='Network status: Unavailable' | loss=10.6410
[base/diagnostic/7] extra_text | 정답=failed | 생성='The GPU status is "failed".' | loss=9.3431
[base/diagnostic/8] other | 정답=healthy | 생성='Network status: "Network Status: Status' | loss=6.1097
[base/train] 정답 0/2 | 평균 loss=12.7998
[base/probe] 정답 0/2 | 평균 loss=9.7420
[base/diagnostic] 정답 0/4 | 평균 loss=10.5820
[sft/train/1] correct | 정답=failed | 생성='failed' | loss=0.0138
[sft/train/2] correct | 정답=healthy | 생성='healthy' | loss=0.0138
[sft/probe/3] correct | 정답=failed | 생성='failed' | loss=0.2292
[sft/probe/4] correct | 정답=healthy | 생성='healthy' | loss=0.1066
[sft/diagnostic/5] correct | 정답=failed | 생성='failed' | loss=0.5903
[sft/diagnostic/6] correct | 정답=healthy | 생성='healthy' | loss=0.0215
[sft/diagnostic/7] correct | 정답=failed | 생성='failed' | loss=0.3245
[sft/diagnostic/8] wrong_label | 정답=healthy | 생성='failed' | loss=1.1236
[sft/train] 정답 2/2 | 평균 loss=0.0138
[sft/probe] 정답 2/2 | 평균 loss=0.1679
[sft/diagnostic] 정답 3/4 | 평균 loss=0.5150
[lora/train/1] other | 정답=failed | 생성='Device GPU status: 0\n\n' | loss=11.8943
[lora/train/2] other | 정답=healthy | 생성='Device network status: Disconnected' | loss=8.2563
[lora/probe/3] other | 정답=failed | 생성="The GPU status is currently '0'." | loss=8.6117
[lora/probe/4] other | 정답=healthy | 생성='The network status is currently stable and functioning' | loss=6.0968
[lora/diagnostic/5] other | 정답=failed | 생성='GPU status:\n\nGPU status:' | loss=14.1535
[lora/diagnostic/6] other | 정답=healthy | 생성='Network status: Unavailable' | loss=7.7694
[lora/diagnostic/7] extra_text | 정답=failed | 생성='The GPU status is "failed".' | loss=6.1346
[lora/diagnostic/8] other | 정답=healthy | 생성='Network status: "Network Status: Status' | loss=5.3541
[lora/train] 정답 0/2 | 평균 loss=10.0753
[lora/probe] 정답 0/2 | 평균 loss=7.3542
[lora/diagnostic] 정답 0/4 | 평균 loss=8.3529
평가 저장: ../smollm2_evaluation_v1.json | 체크포인트 저장·수정 없음
```

</details>

- **SFT의 추가 진단은3/4, 75%입니다.** 마지막 `Report the network status in one word.`에서 정답 healthy 대신 failed를 생성했습니다. 학습 질문·기존 probe를 모두 맞혀도 표현 변경에서 다른 라벨을 고를 수 있음을 보여줍니다. 로그만으로 그 내부 원인을 확정할 수는 없습니다.
- **LoRA는 세 그룹의 평균 loss가 모두 원본보다 낮지만 완전 일치 정답 수는 그대로0입니다.** 정답에 주는 점수 개선과 실제 생성 성공을 분리해서 읽습니다. 이번 설정의 결과이지 LoRA 방식 전체의 한계가 아닙니다.
- 원본과 LoRA의7번 응답 `The GPU status is "failed".`에는 정답 단어가 있지만, 한 단어 응답 기준을 어겨 extra_text 오답입니다. SFT의8번은 단어 형식은 맞지만 상태가 틀린 wrong_label입니다.
- JSON 저장 완료는 실행 로그로 확인한 범위입니다. 이 로그는 별도의 JSON 파일 내용 검사·실행 속도·GPU 메모리·독립 최종 평가의 증거는 아닙니다.

## 6. 출력 읽기와 오류 분석

실제 실행 후 모델별 세 줄의 그룹 요약과 문항별 생성 문자열을 확인합니다. 마지막에 `평가 저장: ... | 체크포인트 저장·수정 없음`이 표시되고 JSON에24개 records가 있어야 합니다. 이 문구는 품질 합격을 뜻하지 않습니다.

| 관찰 | 해석 | 피해야 할 결론 |
|---|---|---|
| train 정답률만 높음 | 학습 질문 적합 | 새로운 표현도 해결됨 |
| probe/diagnostic에서 하락 | 표현 변경에 취약할 가능성; 원문 확인 | 실서비스 전체 성능 수치 |
| loss 감소, 정답률 그대로 | 정답 토큰 점수와 실제 선택이 다름 | 코드가 반드시 잘못됨 |
| extra_text | 정답 단어가 있어도 한 단어 형식 위반 | 부분 문자열이 있으니 정답 |
| wrong_label | 다른 합성 라벨로 완전 일치 | 모든 의미 오류가 이 분류로 탐지됨 |

추가 진단 결과가 좋더라도 이 작은 합성 과제 밖의 지식·안전성·기존 응답 능력 보존을 검증하지 않습니다. 17장과18장은 학습 대상과 학습 장치가 다를 수 있으며, 같은 평가 장치가 학습 조건까지 통제하지는 않습니다. 이 결과를 전체 SFT 대 LoRA의 보편적인 우열·속도·메모리 벤치마크로 해석하지 않습니다.

오답은 정상적인 평가 결과입니다. 이번 묶음에서는 재학습·학습률 튜닝·QLoRA·서빙·자동 배포를 추가하지 않습니다. 결과를 보고 설정을 고치면 이 진단 문항은 개발에 사용한 데이터가 되며, 독립 최종 평가에는 별도 설계가 필요합니다.

## 7. 복습

- [ ] 학습 질문·기존 probe·추가 진단의 결과를 따로 읽는다.
- [ ] 정답 응답을 넣는 loss와 질문만 주는 생성을 구분한다.
- [ ] 정확도 기준과 오류 분류를 출력 확인 전에 정한다.
- [ ] 원본·전체 SFT·원본+어댑터의 적재 차이를 설명한다.
- [ ] 저장·복원 성공, 평가 실행 성공, 품질 개선을 구분한다.

## 공식 참고자료

- [Transformers4.57.1 GenerationConfig와 생성](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)
- [PEFT 모델 적재·is_trainable](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model)

[← 이전: GPU·LoRA](18-lora-gpu.md) · [목차](README.md) · [다음: QLoRA →](20-qlora.md)
