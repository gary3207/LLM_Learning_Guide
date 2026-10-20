# 11 · 작은 언어 모델 저장·복원·생성

[← 이전: 임베딩과 다음 토큰 학습](10-embedding-next-token.md) · [목차](README.md)

## 1. 목표와 선행 조건

10장의 모델은 메모리에만 있고 Python 명령이 끝나면 사라집니다. 이번에는 **같은 모델을 별도 파일에 저장 → 새로운 Python 프로세스로 복원 → 자신의 예측을 다음 입력에 붙여 생성**합니다.

앞 명령의 메모리를 되살리는 것은 아닙니다. seed 7·CPU·float64·같은 어휘와 구조·SGD 200회로 재현 학습해 저장합니다. 이후 generate 명령은 학습하지 않습니다. 두 예문에 맞춘 결과이며 새 문장 성능 검증은 아닙니다.

```text
save:     같은 설정으로 재현 학습 → state_dict·어휘·구조 ID·검사용 logits 저장
                                           ↓ 파일
새 프로세스 generate: 같은 구조 생성 → 가중치 복원 → 저장 전후 점수 비교
                                           ↓
gpu → current → status → failed → <eos> 종료
```

## 2. 저장 정보와 생성 방식

| 정보·연산 | 역할 |
|---|---|
| state_dict | 토큰·위치 임베딩, LayerNorm, QKV, FFN, 출력층의 학습된 숫자 |
| vocab | ID와 단어를 연결하는 순서; 순서가 바뀌면 같은 ID도 다른 단어 |
| architecture | 이 예제의 고정 구조 버전; 범용 모델 자동 구성 정보는 아님 |
| reference_logits | 두 학습 입력의 저장 전 점수 2×4×7; 복원 점검용, 정답/생성 입력 아님 |
| model.eval() | 평가 모드; 미분을 끄는 기능과는 별개 |
| no_grad | 생성·점검에서 미분 그래프를 만들지 않음 |
| logits[0, -1] | 현재 문맥의 마지막 위치에서 다음 토큰 후보 7개의 점수 |
| argmax | 가장 높은 점수의 토큰 하나 선택; 확률 샘플링 아님 |

PyTorch 공식 저장·복원 튜토리얼의 state_dict → load_state_dict 방식입니다. 코드의 모델 정의도 함께 보관해야 합니다. 전체 Python 모델 객체나 optimizer 상태는 저장하지 않습니다. 따라서 **추론용 체크포인트이지 임의 시점 학습 재개용 체크포인트는 아닙니다**.

10장은 정답 문장의 접두 문맥을 넣었습니다. 이번에는 gpu만 넣고 예측 current를 붙인 뒤 gpu current를 다시 넣습니다. 매회 문맥 전체를 계산하며 KV cache는 없습니다. 참조 logits는 복원 검사에서만 사용하고 generate는 정답 배열을 받지 않습니다.

## 3. 실행 위치와 전체 명령

**Ubuntu VM 터미널**, 기존 `~/llm-lab/source/.venv`를 사용합니다. 설치·GPU 사용·기존 모델 변경은 없습니다.

- 새 코드: `~/llm-lab/source/tiny_transformer_checkpoint.py`
- 새 체크포인트: `~/llm-lab/tiny_transformer_v1.pt`
- `cat`은 아래 코드를 파일에 기록합니다. 괄호 안 `set -C`는 기존 코드 파일의 덮어쓰기를 거부합니다. 기존 파일이 있으면 내용부터 확인하며 자동 삭제하지 않습니다.
- 체크포인트도 `open("xb")`로 새 파일만 생성합니다. 같은 파일이 있으면 **학습 전에 실패**합니다. 학습·저장 중 중단되면 새로 만든 빈 파일이나 불완전 파일이 남을 수 있습니다. 기존 정상 파일은 자동 삭제하지 않습니다.
- `python -O`는 assert를 생략하므로 사용하지 않습니다. 자신이 만든 신뢰할 수 있는 체크포인트만 읽습니다. weights_only=True도 임의 파일의 안전성을 보장하지 않습니다.

소스: [03-checkpoint-generate.py](source/09-transformer/03-checkpoint-generate.py). 앞 장을 수정하지 않고 같은 작은 모델 정의를 독립 실행 예제에 유지합니다.

### 3-1. 코드 배치와 재현 학습·별도 저장

아래 전체를 한 번에 실행합니다. 코드 파일을 기록하는 괄호 명령이 성공했을 때만 `&&` 뒤 학습 명령이 실행됩니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python --version

(set -C; cat > tiny_transformer_checkpoint.py <<'PY'
"""10장과 같은 모델을 재현해 별도 저장하거나, 재학습 없이 복원·생성합니다."""
import argparse
from math import sqrt
from pathlib import Path
import torch
from torch import nn

vocab = ["<eos>", "current", "failed", "gpu", "healthy", "network", "status"]
to_id = {token: i for i, token in enumerate(vocab)}
architecture = "tiny-preln-d4-ff8-context4-singlehead-v1"


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        options = {"dtype": torch.float64, "device": "cpu"}
        self.token = nn.Embedding(len(vocab), 4, **options)
        self.position = nn.Embedding(4, 4, **options)
        self.norm1 = nn.LayerNorm(4, eps=1e-5, **options)
        self.norm2 = nn.LayerNorm(4, eps=1e-5, **options)
        self.q = nn.Linear(4, 4, bias=False, **options)
        self.k = nn.Linear(4, 4, bias=False, **options)
        self.v = nn.Linear(4, 4, bias=False, **options)
        self.ff = nn.Sequential(nn.Linear(4, 8, **options), nn.ReLU(),
                                nn.Linear(8, 4, **options))
        self.head = nn.Linear(4, len(vocab), bias=False, **options)

    def forward(self, ids):
        # ponytail: 최대4토큰·단일head·padding 없음; 긴 문맥은 후속 범위입니다.
        assert ids.ndim == 2 and ids.shape[0] > 0 and 1 <= ids.shape[1] <= 4
        assert ids.dtype == torch.long and torch.all((ids >= 0) & (ids < len(vocab)))
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.token(ids) + self.position(positions)
        normalized = self.norm1(embedded)
        Q, K, V = self.q(normalized), self.k(normalized), self.v(normalized)
        scores = (Q @ K.transpose(-2, -1)) / sqrt(Q.shape[-1])
        future = torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool, device=ids.device).triu(1)
        attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        residual = embedded + attention @ V
        hidden = residual + self.ff(self.norm2(residual))
        return self.head(hidden), attention


def save(path):
    # xb는 이미 있는 파일·심볼릭 링크를 덮어쓰지 않습니다. 학습 전에 확보합니다.
    with path.open("xb") as stream:
        torch.manual_seed(7)
        sentences = ["gpu current status failed", "network current status healthy"]
        sequences = [s.split() + ["<eos>"] for s in sentences]
        x = torch.tensor([[to_id[t] for t in seq[:-1]] for seq in sequences])
        y = torch.tensor([[to_id[t] for t in seq[1:]] for seq in sequences])
        model = TinyLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
        loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
        model.train()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            reference_logits, _ = model(x)
            final_loss = loss_fn(reference_logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
        assert torch.isfinite(reference_logits).all()
        assert torch.equal(reference_logits.argmax(-1), y)
        torch.save({"architecture": architecture, "vocab": vocab,
                    "state_dict": model.state_dict(), "reference_logits": reference_logits}, stream)
    print(f"200회 재현 학습 후 loss: {final_loss:.4f} | 학습 위치: 8/8")
    print(f"별도 저장 완료: {path.name}")


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != architecture or payload.get("vocab") != vocab:
        raise ValueError("이 실습의 모델 구조·어휘 순서와 다른 체크포인트입니다")
    model = TinyLanguageModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("가중치에 NaN 또는 무한대가 있습니다")
    model.eval()
    with torch.no_grad():
        probes = torch.tensor([[to_id[t] for t in s.split()] for s in
                              ["gpu current status failed", "network current status healthy"]])
        logits, _ = model(probes)
    if not torch.allclose(logits, payload["reference_logits"], rtol=0.0, atol=1e-12):
        raise ValueError("저장 전·복원 후 logits가 다릅니다")
    return model


@torch.no_grad()
def generate(model, prompt, max_new_tokens=4):
    if not isinstance(prompt, list) or not 1 <= len(prompt) <= 4 or any(
            not isinstance(t, str) or t not in to_id or t == "<eos>" for t in prompt):
        raise ValueError("시작 문맥은 eos를 제외한 등록 토큰 1~4개의 리스트여야 합니다")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4:
        raise ValueError("최대 생성 수는 정수 1~4여야 합니다")
    tokens = prompt.copy()
    for _ in range(max_new_tokens):
        # ponytail: 학습 위치는0~3뿐; 전체 문맥4개를 넘으면 중단, sliding window는 미검증입니다.
        if len(tokens) > 4:
            return tokens, "context"
        ids = torch.tensor([[to_id[t] for t in tokens]], dtype=torch.long)
        logits, _ = model(ids)
        if not torch.isfinite(logits).all():
            raise ValueError("생성 점수가 유한하지 않습니다")
        next_token = vocab[logits[0, -1].argmax().item()]
        tokens.append(next_token)
        if next_token == "<eos>":
            return tokens, "eos"
    return tokens, "limit"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["save", "generate"])
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "save":
        save(args.checkpoint)
    else:
        model = restore(args.checkpoint)
        print("복원 검사 통과: 같은 구조·어휘·저장 전후 logits 일치. 재학습 없음")
        for start, expected in [("gpu", ["gpu", "current", "status", "failed", "<eos>"]),
                                ("network", ["network", "current", "status", "healthy", "<eos>"])]:
            tokens, reason = generate(model, [start])
            assert tokens == expected and reason == "eos"
            assert generate(model, [start], 1) == ([start, "current"], "limit")
            print(f"{' '.join(tokens)} | 종료: {reason}")
        print("검사 통과: 자기 예측 이어 붙이기·양쪽 eos 종료·최대 생성 수 제한")
PY
) && python tiny_transformer_checkpoint.py save ../tiny_transformer_v1.pt
```

`save`는 아래 순서로 실행합니다.
1. 출력 파일을 새 파일로만 확보합니다.
2. 10장과 같은 입력·한 칸 이동 정답·모델을 재현해 200회 학습합니다.
3. 학습 위치 8/8과 유한한 점수를 검사하고 state_dict 등을 저장합니다.

### 3-2. 새 프로세스에서 복원·생성

**3-1의 저장 성공을 확인한 다음**, 같은 VM 터미널에서 실행합니다. 이 명령으로 새 Python 프로세스가 시작됩니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
python tiny_transformer_checkpoint.py generate ../tiny_transformer_v1.pt
```

`restore`는 CPU로 읽고 구조 ID·어휘 순서를 확인한 뒤 `strict=True`로 키·크기가 맞는 가중치를 복원합니다. 저장 전·복원 후 logits를 절대 허용오차 1e-12로 비교합니다. 이 값은 환경 간 비트 단위 동일성을 보장하는 조건이 아닙니다. 오차가 크면 traceback과 환경을 확인하고 검사를 없애지 않습니다.

`generate`는 입력을 복사하고, 예측 토큰을 붙인 뒤 다음 입력으로 사용합니다. 입력 logits는 1×현재길이×7입니다. 정답 배열이나 reference_logits로 토큰을 고르지 않습니다. eos가 나오면 그 토큰을 포함해 반환합니다.

## 4. 실제 로컬 실행 예시와 해석

실제 CPU PyTorch 격리 환경에서 실행한 출력입니다. VM 실행 완료를 뜻하지 않습니다. 같은 seed여도 버전·계산 환경에 따라 숫자가 달라질 수 있습니다.

저장 프로세스:

```text
200회 재현 학습 후 loss: 0.1662 | 학습 위치: 8/8
별도 저장 완료: tiny_transformer_v1.pt
```

다른 프로세스의 복원·생성:

```text
복원 검사 통과: 같은 구조·어휘·저장 전후 logits 일치. 재학습 없음
gpu current status failed <eos> | 종료: eos
network current status healthy <eos> | 종료: eos
검사 통과: 자기 예측 이어 붙이기·양쪽 eos 종료·최대 생성 수 제한
```

`<eos>`는 종료용 토큰이며 실제 문장 끝을 표시합니다. 생성 루프가 학습 문장의 단어들을 스스로 이어 붙이고 멈췄다는 증거입니다. 새로운 문장 이해·일반화 성능의 증거는 아닙니다. 로컬 검증 환경에서는 NumPy 미설치 경고가 있었지만 실제 텐서 계산과 모든 검사가 통과했습니다.

## 5. 종료 조건·검사·한계

| 종료 이유 | 조건 |
|---|---|
| eos | 모델이 eos를 예측했음 |
| limit | 지정한 새 토큰 수만큼 생성; eos 예측을 뜻하지 않음 |
| context | 다음 계산에 필요한 전체 문맥이 4개를 초과함 |

입력은 eos를 제외한 등록 토큰 1~4개, 최대 생성 수는 정수 1~4만 허용합니다. 한 번에 새 토큰 1개만 허용하는 검사도 실행하여 두 시작 단어 모두 current를 만든 뒤 limit로 멈추는지 확인합니다. **입력 4개에서 다음 토큰을 예측하는 것은 가능**하지만, eos가 아닌 다섯 번째 토큰을 다시 입력할 위치 임베딩은 없습니다. 긴 문맥을 임의로 잘라 위치를 다시 매기는 sliding window는 이번 학습에서 검증하지 않아 추가하지 않았습니다.

[독립 검사](source/09-transformer/test_checkpoint.py)는 저장소 루트의 개발용 PyTorch 환경에서 `python source/09-transformer/test_checkpoint.py`로 실행합니다. 임시 디렉터리에서 실제 save와 generate를 서로 다른 프로세스로 실행하고, 이전 실습의 모든 가중치와 일치·기존 파일 보호·입력 거부·어휘 불일치 거부를 확인합니다. 이 검사는 실제 재현 학습을 실행하며 과거 체크포인트는 읽거나 쓰지 않습니다.

- 단일-head·한 블록·입력 최대4개·4좌표·두 학습 예문 전용입니다.
- 별도 평가 데이터·sampling·긴 문맥·학습 재개·서빙은 포함하지 않습니다.
- 복원 성공과 생성 성공은 별개입니다. 저장 출력만으로 생성까지 완료했다고 판단하지 않습니다.
- 다음 확장에서는 별도 데이터 평가나 더 긴 문맥을 목표에 맞게 정합니다.

## 6. 복습

- [ ] 메모리의 모델과 파일에 저장한 state_dict를 구분한다.
- [ ] 가중치뿐 아니라 어휘 순서와 모델 코드가 필요한 이유를 설명한다.
- [ ] 재현 학습·저장과 재학습 없는 복원을 구분한다.
- [ ] teacher forcing과 자신의 예측을 이어 붙이는 생성을 구분한다.
- [ ] eos·최대 생성 수·위치 길이 제한을 구분한다.
- [ ] 두 학습 문장 생성이 일반화 평가가 아닌 이유를 설명한다.

공식 참고: [PyTorch 저장·복원 튜토리얼](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

[← 이전: 임베딩과 다음 토큰 학습](10-embedding-next-token.md) · [목차](README.md)
