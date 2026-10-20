# 17 · 작은 데이터로 SFT: 기본 모델 비교부터 저장·복원까지

[← 이전: 사전학습 LLM 추론](16-pretrained-inference.md) · [목차](README.md) · [다음: GPU·LoRA →](18-lora-gpu.md)

## 1. 완성할 결과물과 개념

16장은 기존 가중치를 읽기만 했습니다. 이번에는 **같은 모델에 질문·정답 응답을 가르치고 → 같은 질문으로 전후 비교하고 → 새 폴더에 저장하고 → 새 Python 프로세스에서 복원**합니다. 이것을 하나의 통합 실습으로 실행합니다.

**SFT(Supervised Fine-Tuning)**는 사전학습된 모델에 입력과 정답 출력을 주고 가중치를 추가 학습하는 방법입니다. 이번에는 모델 전체를 학습합니다. LoRA는 일부 저차원 추가 파라미터를 학습하는 별도 방식이며 이번에는 사용하지 않습니다. Trainer·TRL·PEFT 등 새 프레임워크도 설치하지 않고, 기존 PyTorch·Transformers·SGD를 사용합니다.

### 실습용 합성 규칙과 데이터를 먼저 고정

실제 장비 조회가 아닙니다. 교육용으로 `gpu → failed`, `network → healthy`라는 정답을 지정합니다. 현실에서 항상 GPU가 고장이라는 뜻이 아닙니다.

| 용도 | 질문 | 정답 |
|---|---|---|
| train | `Device gpu status? Reply with one word.` | `failed` |
| train | `Device network status? Reply with one word.` | `healthy` |
| probe | `What is the gpu status? Reply with one word.` | `failed` |
| probe | `What is the network status? Reply with one word.` | `healthy` |

두 train만 가중치 갱신에 사용합니다. probe는 문구를 바꾼 비교 입력이며 train loss에 넣지 않습니다. 그러나 같은 기기·정답·템플릿을 공유하고 각 두 예제뿐이므로 **독립 최종 평가나 넓은 일반화 검증이 아닙니다**. probe를 보고 설정을 조정하면 개발용 데이터가 됩니다. 여기서는 학습률·갱신 수를 고정하고 모델을 선택하거나 재조정하지 않습니다.

## 2. Ubuntu VM에서 전체 실행

실행 위치는 **기존 Ubuntu VM 터미널의 `~/llm-lab/source`**, 가상환경은 `.venv`입니다. 16장의 PyTorch·Transformers 4.57.1과 모델 캐시를 사용합니다. 캐시가 없으면 같은 고정 revision을 다운로드하므로 인터넷이 필요합니다.

- 새 코드: `source/smollm2_sft.py`.
- 새 결과: `~/llm-lab/smollm2_sft_v1/`의 가중치·토크나이저·설정·복원 기준.
- 기존 작은 모델 체크포인트와 원본 Hugging Face 캐시는 덮어쓰지 않습니다.
- CPU·float32·전체 파라미터 학습입니다. 추론보다 RAM이 더 필요하고 느릴 수 있습니다. GPU·CUDA 설정은 변경하지 않습니다.
- 새 폴더는 약 540 MB의 float32 가중치 외에도 토크나이저·기준 파일을 담으므로 여유 공간이 필요합니다. 원본 캐시는 별도로 남습니다.
- 코드나 출력 폴더가 이미 있으면 덮어쓰지 않고 중단합니다. 실패·중단으로 새 폴더에 일부 파일만 남을 수 있으므로 확인 없이 삭제·재사용하지 않습니다.

소스: [01-sft-compare.py](source/17-sft/01-sft-compare.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
(set -C; cat > smollm2_sft.py <<'PY'
"""작은 합성 데이터로 SFT·기본 모델 비교·별도 저장·새 프로세스 복원을 실행합니다."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["train", "check"])
parser.add_argument("output", nargs="?", type=Path, default=Path("../smollm2_sft_v1"))
args = parser.parse_args()
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
train_rows = [("Device gpu status? Reply with one word.", "failed"),
              ("Device network status? Reply with one word.", "healthy")]
probe_rows = [("What is the gpu status? Reply with one word.", "failed"),
              ("What is the network status? Reply with one word.", "healthy")]
torch.set_num_threads(1)
torch.manual_seed(0)
if args.mode == "train":
    args.output.mkdir(parents=True, exist_ok=False)
    location, options = model_id, {"revision": revision}
else:
    location, options = args.output, {"local_files_only": True}
tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=False, **options)
model = AutoModelForCausalLM.from_pretrained(
    location, trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
    attn_implementation="eager", **options,
).to("cpu")


def example(prompt, answer=None):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if answer is None:
        return torch.tensor([prefix])
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True)
    assert full[:len(prefix)] == prefix
    end = full.index(tokenizer.eos_token_id, len(prefix)) + 1
    ids = torch.tensor([full[:end]])
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    assert torch.all(labels[:, :len(prefix)] == -100)
    assert torch.equal(labels[:, len(prefix):], ids[:, len(prefix):])
    assert (labels != -100).any()
    return ids, labels


@torch.inference_mode()
def report(name):
    model.eval()
    records, scores = [], []
    for split, rows in (("train", train_rows), ("probe", probe_rows)):
        for prompt, answer in rows:
            ids, labels = example(prompt, answer)
            loss = model(ids, labels=labels, use_cache=False).loss.item()
            inputs = example(prompt)
            logits = model(inputs).logits[:, -1, :]
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                       max_new_tokens=8, do_sample=False, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
            assert torch.isfinite(logits).all()
            print(f"[{name}/{split}] {answer} | loss={loss:.4f} | 생성={text!r}")
            records.append((split, answer, text))
            scores.append(logits)
    return records, torch.cat(scores)


if args.mode == "train":
    report("base")
    before = model.get_input_embeddings().weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    # ponytail: CPU·전체 파라미터·두 예제·20회 갱신; 확장 시 GPU/LoRA를 별도 비교합니다.
    for step in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model(ids, labels=labels, use_cache=False).loss
                   for ids, labels in (example(*row) for row in train_rows)) / len(train_rows)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert not torch.equal(before, model.get_input_embeddings().weight)
    del before
    records, logits = report("sft")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    torch.save({"records": records, "logits": logits}, args.output / "reference.pt")
    print("추가 학습·별도 저장 완료: smollm2_sft_v1 | 원본 캐시 저장 없음")
else:
    saved = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    records, logits = report("restored")
    assert records == saved["records"], "생성 결과 불일치"
    assert torch.allclose(logits, saved["logits"], atol=1e-5, rtol=1e-5), "logits 불일치"
    print("복원 검사 통과: logits·생성 결과 일치·재학습 없음")
PY
) && python smollm2_sft.py train && python smollm2_sft.py check
```

`train`과 `check`는 별도 Python 프로세스입니다. `&&`는 앞 단계가 성공한 경우에만 다음을 실행합니다. 이미 저장한 후보의 복원만 다시 검사하려면 같은 폴더·환경에서 `python smollm2_sft.py check`만 실행합니다. assert를 사용하므로 `python -O`는 사용하지 않습니다.

## 3. 입력과 정답: 질문을 지우지 않고 loss에서 제외

`example()`은 모델의 대화 템플릿을 재사용합니다. 질문 뒤에 assistant의 정답을 붙인 전체 시퀀스를 만들고, **응답을 시작하기 전 위치의 labels만 `-100`**으로 설정합니다.

- `input_ids`: system·질문·assistant 표시·정답·eos 전체. 모델은 질문을 문맥으로 볼 수 있습니다.
- `labels`: 질문 쪽은 `-100`, 응답과 eos 쪽은 실제 토큰 ID.
- `-100`: 이 위치를 loss 계산에서 무시한다는 뜻입니다. 토큰을 입력에서 삭제하거나 Attention을 막는 값이 아닙니다.
- `prefix`: 응답 생성 시 입력할 질문 부분입니다. 전체 시퀀스가 이 prefix로 시작하는지 확인합니다.

이 모델의 `labels=...` 경로는 **내부에서 예측과 정답을 한 칸 이동**합니다. 마지막 질문/assistant 접두 토큰 위치의 logits로 첫 응답 토큰을, 마지막 응답 토큰 위치의 logits로 eos를 맞힙니다. 코드를 보고 labels를 다시 한 칸 옮기면 정답이 두 번 이동하므로 그렇게 하지 않습니다. eos 뒤 템플릿 줄바꿈은 학습 대상에서 제외합니다.

두 예제를 따로 순전파하므로 padding·배치 합치기·문장 사이 연결이 필요 없습니다. 각 예제의 응답 토큰 평균 loss를 계산한 뒤 두 예제 loss를 평균합니다. 응답 길이가 다르면 전체 토큰을 한데 모은 평균과 가중 방식이 다릅니다.

## 4. 실제로 바뀌는 것과 비교 방식

같은 고정 revision의 전체 파라미터를 `SGD(lr=0.01, momentum=0, weight_decay=0)`로 20회 갱신합니다. 매회 두 예제의 평균 loss를 새로 계산하고 `zero_grad → backward → gradient norm을 최대1로 제한 → step`을 수행합니다. clipping이 적용되면 원래 gradient가 그대로 적용되는 것은 아닙니다. 학습되는 입력 임베딩이 실제로 달라졌는지도 확인합니다.

`report()`는 `eval()`·`inference_mode()`에서 전후를 비교합니다. loss는 **정답 응답과 eos를 제공한 teacher-forcing 평균 cross-entropy**입니다. 생성은 정답을 주지 않고 질문만으로 `generate()`를 실행합니다. 따라서 loss와 자유 생성은 별도의 관찰입니다.

두 시점 모두 같은 질문·정답·최대 새 토큰8개·greedy 조건을 사용합니다. 원본에서 먼저 결과를 기록한 뒤 같은 메모리 모델의 가중치를 갱신하므로 모델 두 개를 동시에 상주시킬 필요가 없습니다. 원본 가중치는 원래 캐시에서 다시 불러올 수 있으며 이번 코드가 그 캐시에 저장하지 않습니다.

생성 문자열은 정답 단어만 나오기를 기대하지만 코드가 결과를 정답 문자열로 치환하거나 강제로 제한하지 않습니다. 최대8개까지 생성하므로 기본 모델의 긴 응답은 잘릴 수 있습니다. 생성 품질·eos 종료를 전부 통과한 것으로 해석하지 않습니다.

### 별도 저장과 복원

`save_pretrained(..., safe_serialization=True)`는 학습 후 모델 가중치·설정을, `tokenizer.save_pretrained()`는 대응하는 토크나이저를 새 폴더에 저장합니다. `reference.pt`에는 비교 질문의 저장 전 logits와 실제 생성 문자열을 기록합니다. 비교용 저장 값은 정답 점수가 아니라 **저장 시점 모델의 결과**입니다.

`check`는 원본을 다시 학습하지 않고 새 폴더에서 `local_files_only=True`로 모델·토크나이저를 읽습니다. 실제 순전파·생성을 다시 실행해 네 질문의 마지막 위치 전체 어휘 logits와 생성 문자열을 비교합니다. 저장된 문장을 그냥 답으로 출력하는 방식이 아닙니다. 파일은 자신이 만든 신뢰하는 것만 읽고, `reference.pt`는 `weights_only=True`로 읽습니다.

이 저장은 **추론용 체크포인트**입니다. optimizer/RNG 상태가 없어 임의 시점의 정확한 학습 재개를 보장하지 않습니다. `check` 통과는 학습 후 모델이 복원됐다는 뜻이지, 모델이 좋은 답만 한다는 뜻이 아닙니다.

## 5. 실제 로컬 실행 예시

Python 3.12·PyTorch 2.9.0+cpu·Transformers 4.57.1의 격리된 Windows 로컬 환경에서 실제 전체 파라미터 추가 학습과 별도 프로세스 복원을 실행했습니다. 아래는 실제 stdout이며 **학습자 Ubuntu VM 실행 완료를 뜻하지 않습니다**. 장치·버전·계산 조건에 따라 수치·문장이 달라질 수 있습니다.

```text
[base/train] failed | loss=14.4325 | 생성='Device GPU status: 1'
[base/train] healthy | loss=11.1671 | 생성='Device network status:\n\n- Device'
[base/probe] failed | loss=12.3040 | 생성="The GPU status is currently '0'."
[base/probe] healthy | loss=7.1799 | 생성='The network status is currently stable and functioning'
[sft/train] failed | loss=0.0137 | 생성='failed'
[sft/train] healthy | loss=0.0138 | 생성='healthy'
[sft/probe] failed | loss=0.2291 | 생성='failed'
[sft/probe] healthy | loss=0.1066 | 생성='healthy'
추가 학습·별도 저장 완료: smollm2_sft_v1 | 원본 캐시 저장 없음
[restored/train] failed | loss=0.0137 | 생성='failed'
[restored/train] healthy | loss=0.0138 | 생성='healthy'
[restored/probe] failed | loss=0.2291 | 생성='failed'
[restored/probe] healthy | loss=0.1066 | 생성='healthy'
복원 검사 통과: logits·생성 결과 일치·재학습 없음
```

기본 모델은 이 합성 규칙을 따로 배우지 않았으므로 두 단어를 답하지 못한다고 일반적인 영어 능력이 없다는 뜻은 아닙니다. 이번 작은 실험에서는 추가 학습 후 train뿐 아니라 probe의 응답도 두 정답 단어와 일치했고, 해당 정답·eos의 loss가 낮아졌습니다.

그러나 같은 두 기기에 대한 표현 변경 네 질문의 결과입니다. 새로운 기기·복잡한 질문·실제 상태 조회·사실성 평가를 증명하지 않습니다. 16장의 일반 GPU 설명 질문을 포함한 **넓은 생성 능력의 유지·회귀는 이번에 평가하지 않았습니다**. 작은 과제에 맞춘 추가 학습은 다른 응답을 악화시킬 수 있으므로 모델을 운영용 개선판으로 선언하지 않습니다.

## 6. 검사·복습·문제 대응

개발 환경의 저장소 루트에서 [test_sft.py](source/17-sft/test_sft.py)를 실행할 수 있습니다. 실제 모델과 동일한 패키지가 필요하고, 임시 폴더에서 실제 추가 학습을 수행합니다. VM 학습자가 반드시 한 번 더 실행해야 하는 단계는 아닙니다.

```bash
python source/17-sft/test_sft.py
```

응답/eos만 남는 labels의 실제 값, 원본 가중치 파일 해시 불변, 새 모델의 실제 저장·별도 프로세스 복원, 기존 폴더 재저장 거부, 정상 복원 뒤 후보 파일 해시 불변을 확인합니다. 임시 복원 기준 logits를 의도적으로 바꾸면 복원 검사가 실패하는지도 확인합니다. 의도적 변조는 테스트 임시 폴더 안에서만 수행합니다.

기존 출력 폴더 보호 오류가 나면 저장 완료 여부를 먼저 확인하고, 완료된 후보에는 `check`만 실행합니다. 메모리 부족·다운로드·모듈 오류는 마지막 오류 메시지로 구분하며 원본을 삭제하거나 SSL 검증을 끄지 않습니다. 실패한 새 폴더가 완전한 모델이라고 가정하지 않습니다.

- [ ] SFT에서 질문은 문맥으로 유지하고 응답만 loss에 넣는 이유를 설명한다.
- [ ] CausalLM의 내부 정답 이동과 직접 이동을 중복하지 않는다.
- [ ] teacher-forcing loss와 질문만 주는 자유 생성을 구분한다.
- [ ] 작은 합성 과제 성공을 일반적 기술지원 품질로 확대하지 않는다.
- [ ] 원본 캐시와 별도 학습 모델을 구분하고 재학습 없이 복원한다.

## 공식 참고자료

- [SmolLM2 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Transformers 4.57.1 LlamaForCausalLM](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llama)
- [Transformers 4.57.1 대화 템플릿](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)

[← 이전: 사전학습 LLM 추론](16-pretrained-inference.md) · [목차](README.md) · [다음: GPU·LoRA →](18-lora-gpu.md)
