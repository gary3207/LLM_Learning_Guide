# 16 · 사전학습 LLM 불러오기: 토큰화부터 응답 생성까지

[← 이전: 작은 모델 마무리](15-finish-tiny-model.md) · [목차](README.md) · [다음: 작은 SFT →](17-sft-compare.md)

## 1. 이번에 완성할 결과물

**이미 학습된 모델과 토크나이저를 불러와 → 질문을 숫자로 바꾸고 → 다음 토큰 점수를 확인하고 → 응답을 생성**하는 하나의 Python 프로그램을 만듭니다. 새로운 학습·LoRA·서버 구축은 하지 않습니다. 기존 작은 모델의 파일도 읽거나 바꾸지 않습니다.

사용할 모델은 `HuggingFaceTB/SmolLM2-135M-Instruct`입니다. 최신·최고 성능 모델을 고르는 실습이 아니라, 작은 공개 모델로 입력부터 출력까지 관찰하는 실습입니다. 모델 카드에서 영어 모델임을 확인할 수 있어 영어 질문으로 시작합니다. 다운로드와 CPU 추론을 실제로 확인한 revision과 패키지 버전을 고정합니다.

| 개념 | 앞선 작은 모델 | 이번 모델 |
|---|---|---|
| 가중치 | 두 합성 예문에서 직접 학습 | 배포자가 이미 학습한 가중치를 다운로드 |
| 어휘·토크나이저 | 공백으로 나눈 일곱 토큰 | 모델과 함께 배포된 토크나이저 |
| 입력 형식 | `gpu` 등의 토큰 목록 | 역할·본문·특수 토큰을 포함한 대화 형식 |
| 다음 토큰 | 마지막 위치 logits의 argmax | 같은 원리, `generate()`가 반복을 담당 |
| 이번에 바뀌는 가중치 | 이전 장은 실제 추가 학습 | 없음: 추론만 실행 |

**사전학습**은 배포 전에 많은 텍스트로 가중치를 학습하는 과정입니다. **Instruct**는 지시·대화 응답에 맞춘 추가 학습을 거친 배포형입니다. 둘 다 이번 프로그램이 수행하는 학습이 아닙니다. `from_pretrained()`는 새 모델을 처음부터 훈련하는 명령이 아닙니다.

## 2. Ubuntu VM에서 전체 실행

**기존 Ubuntu VM 터미널**, `~/llm-lab/source`의 `.venv`에서 실행합니다. PyTorch는 기존 설치를 유지하고 Transformers만 지정 버전으로 설치합니다. 이 실습은 CPU·float32이며 GPU 드라이버나 CUDA를 바꾸지 않습니다.

- 새 코드: `~/llm-lab/source/pretrained_inference.py`. 기존 파일은 덮어쓰지 않습니다.
- 새 모델 파일: Hugging Face 캐시에 다운로드됩니다. 기본 위치는 `~/.cache/huggingface/`이며 `HF_HOME` 설정이 있으면 그 위치를 사용합니다. 저장소에 넣지 않습니다.
- 최초 실행에는 인터넷과 다운로드·메모리 공간이 필요합니다. 가중치 파일은 실제 확인값 약 269 MB이며, 설치·캐시·실행에는 그보다 여유 있는 공간이 필요합니다.
- 기존 `tiny_transformer_v1.pt`와 `tiny_transformer_augmented_v1.pt`는 그대로 둡니다. 새 `.pt` 파일은 만들지 않습니다.
- 아래 `&&`는 앞 명령이 실패하면 뒤 명령을 실행하지 않습니다. `(set -C; ...)`는 기존 코드 파일 덮어쓰기를 거부합니다.

소스: [01-tokenize-generate.py](source/16-pretrained/01-tokenize-generate.py).

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python --version &&
python -m pip install 'transformers==4.57.1' &&
(set -C; cat > pretrained_inference.py <<'PY'
"""사전학습된 작은 Instruct 모델로 토큰화·점수·반복 생성을 관찰합니다. 학습은 하지 않습니다."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
prompt = "Explain what a GPU is in one short sentence."
torch.set_num_threads(1)
# ponytail: 첫 추론은 CPU·float32·단일 입력; GPU 성능 비교는 별도 실습입니다.
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=False,
    use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
).to("cpu").eval()
print(f"모델: {model_id}")
print(f"revision: {revision}")
print(f"장치: {model.device} | dtype: {model.dtype} | 파라미터: {model.num_parameters():,}")
plain_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"질문: {prompt}")
print("질문 토큰:", list(zip(plain_ids, tokenizer.convert_ids_to_tokens(plain_ids))))
assert tokenizer.decode(plain_ids, clean_up_tokenization_spaces=False) == prompt
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
)
print("대화 형식:", repr(tokenizer.decode(inputs["input_ids"][0])))
print("input_ids shape:", tuple(inputs["input_ids"].shape))
with torch.inference_mode():
    logits = model(**inputs).logits
    output = model.generate(
        **inputs, max_new_tokens=48, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
assert torch.isfinite(logits).all()
new_ids = output[0, inputs["input_ids"].shape[1]:]
assert 1 <= new_ids.numel() <= 48
assert new_ids[0].item() == logits[0, -1].argmax().item()
answer = tokenizer.decode(new_ids, skip_special_tokens=True)
print("logits shape:", tuple(logits.shape))
print("첫 예측 토큰:", new_ids[0].item(), repr(tokenizer.decode(new_ids[:1])))
print("응답:", answer)
print(f"새 토큰 수: {new_ids.numel()} | 마지막이 eos: {new_ids[-1].item() == tokenizer.eos_token_id}")
print("추론 완료: 재학습 없음·기존 작은 모델 파일 사용/변경 없음")
PY
) && python pretrained_inference.py
```

이미 코드를 작성했다면 내용을 확인한 뒤 `python pretrained_inference.py`만 다시 실행합니다. 기존 파일을 삭제하거나 보호 옵션을 제거할 필요가 없습니다. assert 검사가 있으므로 `python -O`는 사용하지 않습니다.

## 3. 코드와 데이터 흐름

### 모델과 토크나이저를 한 쌍으로 읽기

`AutoTokenizer`는 문자를 토큰 ID로, `AutoModelForCausalLM`은 토큰 ID를 다음 토큰 점수로 바꿉니다. 같은 모델 ID와 revision으로 둘을 맞춥니다. 서로 다른 어휘의 토크나이저를 임의로 섞으면 같은 ID가 다른 의미가 됩니다.

`revision`은 다운로드할 모델 저장소의 커밋입니다. `trust_remote_code=False`로 원격 사용자 정의 Python 코드 실행을 허용하지 않고, 가중치는 `use_safetensors=True`로 읽습니다. `.to("cpu")`는 계산 장치를, `dtype=torch.float32`는 가중치 자료형을 지정합니다. `eval()`은 평가 모드, `inference_mode()`는 기울기 기록 없이 추론하는 문맥입니다. optimizer·backward·step은 없습니다.

### 질문 토큰과 실제 입력 토큰 구분

`plain_ids`는 질문 본문만 토큰화한 결과입니다. 출력의 `Ġ`는 이 토크나이저에서 앞 공백을 나타내는 표기이며, 본문에 그런 글자를 입력한 것이 아닙니다. 토큰은 일반적으로 단어와 일대일이 아닙니다. 다른 단어·언어는 여러 조각으로 나뉠 수 있습니다.

`apply_chat_template()`는 역할과 대화 구분 토큰을 붙여 **모델이 배운 대화 형식**으로 바꿉니다. 이 모델의 템플릿은 system 메시지를 기본으로 추가합니다. 그 문구가 출력되어도 별도 모델이 실행되거나 이번에 학습된 것이 아닙니다. `add_generation_prompt=True`는 assistant 응답을 시작할 위치를 표시합니다. `tokenize=True`로 바로 수치화해 특수 토큰을 다시 중복 추가하지 않습니다.

`inputs`에는 `input_ids`와 `attention_mask`가 들어갑니다. 이 단일 입력에서 mask는 실제 입력 위치를 나타내며 모델 내부의 causal mask와 역할이 다릅니다. 실제 입력은 질문 본문보다 길어집니다.

### 점수 하나에서 반복 생성으로

실제 예시의 `input_ids`는 `(1, 40)`: 문장 한 개·대화 형식 포함 40토큰입니다. `logits`는 `(1, 40, 49152)`: 문장·입력 위치·어휘 후보 점수입니다. 마지막 위치 `logits[0, -1]`에서 가장 큰 후보가 첫 새 토큰입니다. softmax 전 점수이며 정답 확률이나 사실성 점수가 아닙니다.

`do_sample=False, num_beams=1`은 무작위 샘플링 대신 greedy 선택을 사용합니다. 첫 토큰이 별도 순전파의 argmax와 같은지 검사합니다. `generate()`는 그 토큰을 문맥에 붙이고 같은 방식으로 다음을 예측합니다. 별도 logits 계산은 원리 관찰용이라 첫 입력 순전파가 한 번 중복됩니다.

출력에는 입력 토큰도 포함됩니다. 그래서 `output[0, 입력길이:]`로 **새로 생성한 부분만** 잘라 응답으로 디코딩합니다. `skip_special_tokens=True`는 종료 토큰 등을 표시에서 제외할 뿐, 생성 내용을 교정하지 않습니다. `max_new_tokens=48`은 입력 길이가 아닌 새 생성량의 상한입니다. eos가 먼저 나오면 더 일찍 끝나며, 길이 상한에 닿았다고 답이 완성된 것은 아닙니다.

## 4. 실제 로컬 실행 예시와 해석

Python 3.12·PyTorch 2.9.0+cpu·Transformers 4.57.1의 격리된 로컬 Windows 환경에서 모델 다운로드와 CPU 추론을 실행했습니다. 같은 캐시에서 오프라인 재실행도 통과했습니다. **Ubuntu VM 실행 완료를 뜻하지 않습니다.** 패키지·장치·계산 조건이 다르면 출력 문장이 달라질 수 있습니다.

```text
모델: HuggingFaceTB/SmolLM2-135M-Instruct
revision: 12fd25f77366fa6b3b4b768ec3050bf629380bac
장치: cpu | dtype: torch.float32 | 파라미터: 134,515,008
질문: Explain what a GPU is in one short sentence.
질문 토큰: [(36971, 'Explain'), (732, 'Ġwhat'), (253, 'Ġa'), (29593, 'ĠGPU'), (314, 'Ġis'), (281, 'Ġin'), (582, 'Ġone'), (1890, 'Ġshort'), (6330, 'Ġsentence'), (30, '.')]
대화 형식: '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nExplain what a GPU is in one short sentence.<|im_end|>\n<|im_start|>assistant\n'
input_ids shape: (1, 40)
logits shape: (1, 40, 49152)
첫 예측 토큰: 49 'A'
응답: A GPU (Graphics Processing Unit) is a specialized computer component that processes and displays images, videos, and other visual content.
새 토큰 수: 27 | 마지막이 eos: True
추론 완료: 재학습 없음·기존 작은 모델 파일 사용/변경 없음
```

질문 본문은 10토큰이지만 대화 형식까지 포함하면 40토큰입니다. 첫 새 토큰은 ID49, `A`였고 총 27개 새 토큰을 생성했습니다. 새 토큰 수에는 화면에서 숨긴 eos도 포함됩니다. 응답은 **모델이 생성한 관찰 결과**입니다. GPU의 활용을 그래픽 밖으로 설명하지 않는 등 범위가 좁고, 한 문장이 자연스럽다는 사실은 기술지원 정확도·지시 준수·한국어 성능의 증명이 아닙니다.

일곱 토큰의 작은 모델에서 없던 영어 문장을 얻었지만, 같은 데이터·정답·계산량으로 비교한 품질 실험은 아닙니다. 이번 목표는 기존 가중치와 토크나이저로 추론 경로를 실행하는 것입니다. 이 모델도 환각·반복·부정확한 답을 만들 수 있으며 생성문을 실제 장비 상태로 믿어서는 안 됩니다.

## 5. 문제 발생 시와 다시 실행

| 상황 | 확인할 내용 |
|---|---|
| `python` 또는 모듈을 못 찾음 | `cd ~/llm-lab/source`, `source .venv/bin/activate` 후 `python --version`, `python -m pip show torch transformers` 확인 |
| 다운로드 실패 | 실제 오류와 네트워크·디스크를 확인. SSL 검증 해제나 토큰 입력으로 우회하지 않음 |
| 기존 코드 파일 보호 오류 | 기존 내용을 확인하고 코드가 이미 맞으면 `python pretrained_inference.py`만 실행 |
| 느리거나 길이 상한으로 종료 | 이번 CPU 실습의 실행 조건을 확인. 즉시 GPU·서빙·양자화 설정을 추가하지 않음 |

캐시가 완전히 준비된 뒤 네트워크 없이 재실행하려면 기존 폴더·가상환경에서 `HF_HUB_OFFLINE=1 python pretrained_inference.py`를 사용합니다. 캐시가 없으면 실패하는 것이 정상이며 첫 다운로드를 생략하는 명령이 아닙니다. 설치와 다운로드까지 인터넷이 불필요하다는 의미도 아닙니다.

개발자가 저장소 루트에서 실행할 수 있는 검사는 [test_pretrained.py](source/16-pretrained/test_pretrained.py)입니다. torch·지정 Transformers와 실제 모델이 필요하며 최초에는 다운로드할 수 있습니다.

```bash
python source/16-pretrained/test_pretrained.py
```

토큰 왕복·출력의 입력 접두 보존·첫 argmax·생성 길이·유한 logits·gradient 없음·응답 부분 디코딩을 확인합니다. 자연어 응답의 사실성이나 운영 품질을 채점하는 검사는 아닙니다.

- [ ] 다운로드·복원과 학습을 구분한다.
- [ ] 질문 본문과 대화 형식을 포함한 실제 입력 길이를 구분한다.
- [ ] ID·logits·새 토큰·디코딩의 연결을 설명한다.
- [ ] eos 종료와 생성 길이 상한을 구분한다.
- [ ] 응답 생성 성공을 사실성 평가 통과로 해석하지 않는다.

다음 [17장](17-sft-compare.md)은 작은 데이터로 사전학습 모델을 추가 학습하고 기본 모델과 같은 기준으로 비교한 뒤 별도 저장·복원합니다. 이 16장에서는 SFT·LoRA·RAG·서빙을 설치하거나 실행하지 않습니다.

## 공식 참고자료

- [모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Transformers 4.57.1 대화 템플릿](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)
- [Transformers 4.57.1 생성 설정](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)

[← 이전: 작은 모델 마무리](15-finish-tiny-model.md) · [목차](README.md) · [다음: 작은 SFT →](17-sft-compare.md)
