# 34. 작은 Transformer 구조 비교

> 목표: 기존 단일-head·단일층 블록을 여러 head/층으로 확장하고, 보이는 문맥을 줄였을 때 무엇이 달라지는지 비교한다. 학습·진단 평가·새 저장·새 프로세스 복원을 한 묶음으로 실행한다.

[← 이전: 학습 재개](33-training-resume.md) · [목차](README.md)

## 1. 무엇을 바꾸나

이번에는 사전학습 SmolLM2/QLoRA가 아니라 [11장 직접 만든 Transformer](11-checkpoint-generation.md)로 돌아온다. 기존 구현의 토큰/위치 임베딩, Pre-LN, QKV, causal mask, 두 잔차, FFN, 어휘 출력층을 이어 사용한다. 원본 소스와 저장 모델은 수정하지 않는다. 새 정본은 [구조 비교 코드](source/34-structure-comparison/01-compare-structure.py), 검사는 [실제 Torch 테스트](source/34-structure-comparison/test_structure.py)다.

| 이름 | Attention head | 블록 수 | 최대 보이는 토큰 | 고정되는 것 |
|---|---:|---:|---:|---|
| baseline | 1 | 1 | 4 | 폭4, FFN8, 위치 테이블4 |
| heads2 | 2 | 1 | 4 | 기준에서 head만 변경 |
| layers2 | 1 | 2 | 4 | 기준에서 블록만 추가 |
| context2 | 1 | 1 | 2 | 기준에서 보이는 마지막 토큰 수만 변경 |

- **Head:** 폭4의 Q/K/V를 head1이면4차원 하나, head2이면2차원 둘로 나눈다. 각 head가 자기 Attention 비중을 계산하고 결과를 이어 붙여 다시 폭4로 만든다. 여기서 `head`라는 어휘 출력층 이름과 Attention head 개수는 다르다.
- **층:** Attention/FFN 잔차 블록을 순서대로 두 번 통과한다. 입력 토큰을 두 번 읽는 별도 학습 반복이 아니다.
- **문맥:** `prompt.split()[-context:]`로 입력을 잘라 모델에 전달한다. 단순히 메모리 한도 설정만 줄이는 것이 아니라, 모델이 실제로 볼 수 있는 토큰을 바꾼다. 위치는 잘린 입력의0부터 다시 매긴다. 생성용 sliding window나 KV cache는 구현하지 않는다.

모든 조건에서 폭/어휘/위치 테이블4·seed7·CPU float64·단일스레드·SGD lr0.1·200회·마지막 위치 loss·배치4개를 고정한다. 공통 이름의 초기 가중치는 기준 모델에서 실제 복사한다. 두 번째 블록만 새 초기값이다. head 수 증가가 폭이나 전체 파라미터 수 증가를 뜻하지 않으며, 깊이가 늘면 연산량과 파라미터 수는 달라진다. 같은 갱신 수는 같은 계산 비용이 아니다.

이전 모델과 마찬가지로 Attention 출력 projection은 항등으로 생략했고 최종 LayerNorm/dropout도 없다. 표준 LLM 전체 구조나 논문 재현이 아니라 기존 블록의 최소 확장이다. 같은 seed의 기준 모델은 이전11장 초기 logits와 실제로 일치하는지 검사했다. **이전 학습 체크포인트에서 이어 학습하는 것은 아니다.**

## 2. 데이터와 평가 의미

학습4개는15장의 기존 보완 학습 입력과 같다.

| 학습 입력 | 마지막 위치의 다음 토큰 정답 |
|---|---|
| gpu current status | failed |
| network current status | healthy |
| gpu status current | failed |
| network status current | healthy |

진단 probe4개는 `gpu/network current current status`, `gpu/network status status current`다. 기존 어휘를 반복한 아주 작은 합성 진단이며 독립 test·실제 장애 진단·넓은 일반화 검증이 아니다. 여기서는 모든 모델을 평가만 하고 최고 모델 선택이나 probe 기반 재튜닝을 하지 않는다.

**학습 목적 변경:** 10/11장의 모든 위치 다음 토큰 평균 loss가 아니라,15장처럼 마지막 입력 위치에서 `failed/healthy`를 맞히는 평균 cross-entropy다. 정답은7개 전체 어휘 중 하나다. 따라서 기존11장 loss 수치나 반복 생성 성능과 직접 비교할 수 없다. 생성은 이번 범위가 아니다.

문맥2에서는 `gpu current status`와 `network current status`가 모두 `current status`가 된다. 입력이 같은데 정답은 반대라 어느 결정적 모델도 둘 다 맞힐 수 없다. 다른 문장 틀 쌍도 같아서 **학습4개와 probe4개 각각 최대2개 정답**이다. 코드의 `collision_bound`는 동일 입력 그룹마다 가장 많은 정답 수를 더한 이 데이터의 결정적 분류 상한이다. head나 층을 늘려도 입력에서 사라진 장치 이름은 복구할 수 없다. 상한과 실제 학습이 상한에 도달하는지는 별개다.

## 3. 실행 위치와 파일 보호

Ubuntu VM의 기존 `~/llm-lab/source/.venv`를 그대로 사용한다. **의도적으로 CPU 실행**이며 GPU·양자화·새 패키지가 필요하지 않다. 작은 폭의 실험에서 이전 직접 만든 모델의 float64 조건을 유지하기 위한 선택이고 GPU 속도/메모리 비교가 아니다.

```bash
cd ~/llm-lab/source && source .venv/bin/activate
```

아래 블록은 새 `compare_structure.py`만 배타 생성한다. 같은 파일이 이미 있으면 덮어쓰지 않고 오류가 난다. 새 결과 폴더 `../tiny_structure_v1`도 이미 있으면 학습 전에 거부한다. 기존 tiny Transformer/QLoRA/품질 비교/재개 결과/레지스트리를 열거나 변경하지 않는다. 오류가 나면 기존 파일을 삭제하지 말고 원인을 확인한다.

## 4. 전체 코드 생성

```bash
python - <<'PY'
from pathlib import Path
code = r'''"""CPU float64: change heads, depth or visible context; save and restore separately."""
import argparse
from collections import Counter, defaultdict
import json
from math import sqrt
from pathlib import Path
import torch
from torch import nn

VOCAB = ['<eos>', 'current', 'failed', 'gpu', 'healthy', 'network', 'status']
TO_ID = {token: i for i, token in enumerate(VOCAB)}
TRAIN = [('gpu current status', 'failed'), ('network current status', 'healthy'),
         ('gpu status current', 'failed'), ('network status current', 'healthy')]
PROBE = [('gpu current current status', 'failed'), ('network current current status', 'healthy'),
         ('gpu status status current', 'failed'), ('network status status current', 'healthy')]
CONFIGS = {'baseline': (1, 1, 4), 'heads2': (2, 1, 4),
           'layers2': (1, 2, 4), 'context2': (1, 1, 2)}
OPTIONS = {'dtype': torch.float64, 'device': 'cpu'}
ARCHITECTURE = 'tiny-preln-d4-ff8-maxpos4-identity-output-multihead-v1'
STEPS, LR = 200, 0.1


class Block(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(4, eps=1e-5, **OPTIONS)
        self.norm2 = nn.LayerNorm(4, eps=1e-5, **OPTIONS)
        self.q = nn.Linear(4, 4, bias=False, **OPTIONS)
        self.k = nn.Linear(4, 4, bias=False, **OPTIONS)
        self.v = nn.Linear(4, 4, bias=False, **OPTIONS)
        self.ff = nn.Sequential(nn.Linear(4, 8, **OPTIONS), nn.ReLU(), nn.Linear(8, 4, **OPTIONS))

    def forward(self, x):
        b, t, width = x.shape
        normalized = self.norm1(x)
        q, k, v = [projection(normalized).reshape(b, t, self.heads, width // self.heads).transpose(1, 2)
                   for projection in (self.q, self.k, self.v)]
        scores = q @ k.transpose(-2, -1) / sqrt(width // self.heads)
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        attention = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
        mixed = (attention @ v).transpose(1, 2).contiguous().reshape(b, t, width)
        # ponytail: identity output projection, as in chapter11; learned W_o is a later architecture change.
        residual = x + mixed
        return residual + self.ff(self.norm2(residual)), attention


class TinyLanguageModel(nn.Module):
    def __init__(self, heads=1, layers=1, context=4):
        super().__init__()
        if any(type(n) is not int for n in (heads, layers, context)) or heads not in (1, 2) or layers not in (1, 2) or context not in (2, 4):
            raise ValueError('heads/layers=1|2, context=2|4 required')
        self.context = context
        self.token = nn.Embedding(len(VOCAB), 4, **OPTIONS)
        # Keep the same four position vectors even when only two tokens are visible.
        self.position = nn.Embedding(4, 4, **OPTIONS)
        self.blocks = nn.ModuleList([Block(heads) for _ in range(layers)])
        self.head = nn.Linear(4, len(VOCAB), bias=False, **OPTIONS)

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[0] == 0 or not 1 <= ids.shape[1] <= self.context or ids.dtype != torch.long or not torch.all((ids >= 0) & (ids < len(VOCAB))):
            raise ValueError('invalid CPU token IDs or context overflow')
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        weights = []
        for block in self.blocks:
            x, attention = block(x)
            weights.append(attention)
        return self.head(x), weights


def batch(rows, context):
    return (torch.tensor([[TO_ID[t] for t in prompt.split()[-context:]] for prompt, _ in rows]),
            torch.tensor([TO_ID[target] for _, target in rows]))


def collision_bound(x, y):
    groups = defaultdict(Counter)
    for tokens, target in zip(x.tolist(), y.tolist()):
        groups[tuple(tokens)][target] += 1
    return sum(max(counts.values()) for counts in groups.values())


@torch.no_grad()
def evaluate(model, rows):
    x, y = batch(rows, model.context)
    logits = model(x)[0][:, -1]
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    return {'correct': int((logits.argmax(-1) == y).sum()), 'total': len(rows),
            'loss': nn.functional.cross_entropy(logits, y).item(),
            'predictions': [VOCAB[i] for i in logits.argmax(-1).tolist()],
            'collision_bound': collision_bound(x, y)}, logits


def train(output):
    # Exclusive directory: old tiny models, QLoRA artifacts and registry are never opened.
    output.mkdir(parents=False, exist_ok=False)
    torch.manual_seed(7)
    baseline = TinyLanguageModel()
    shared = baseline.state_dict()
    report = {'architecture': ARCHITECTURE, 'torch': str(torch.__version__), 'device': 'cpu',
              'dtype': 'float64', 'seed': 7, 'steps': STEPS, 'lr': LR,
              'train': TRAIN, 'probe': PROBE, 'results': {}}
    for name, config in CONFIGS.items():
        torch.manual_seed(7)
        model = TinyLanguageModel(*config)
        # Match shared initial weights exactly, including the vocabulary head in deeper models.
        state = model.state_dict()
        state.update(shared)
        model.load_state_dict(state, strict=True)
        initial = evaluate(model, TRAIN)[0]
        x, y = batch(TRAIN, model.context)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
        model.train()
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x)[0][:, -1], y)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            optimizer.step()
        model.eval()
        metrics, references = {}, {}
        for group, rows in [('train', TRAIN), ('probe', PROBE)]:
            metrics[group], references[group] = evaluate(model, rows)
        result = {'config': list(config), 'parameters': sum(p.numel() for p in model.parameters()),
                  'initial_train': initial, **metrics}
        report['results'][name] = result
        with (output / (name + '.pt')).open('xb') as stream:
            torch.save({'architecture': ARCHITECTURE, 'vocab': VOCAB, 'config': list(config),
                        'state_dict': model.state_dict(), 'reference_logits': references}, stream)
        print(f"{name} heads={config[0]} layers={config[1]} context={config[2]} params={result['parameters']} "
              f"train={metrics['train']['correct']}/4 loss={metrics['train']['loss']:.6f} "
              f"probe={metrics['probe']['correct']}/4 loss={metrics['probe']['loss']:.6f}", flush=True)
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('SAVED report.json + 4 checkpoints | CPU only; no old models read/changed; probe is diagnostic')


def check(output):
    destination = output / 'verify.json'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('existing verify.json preserved')
    restored = []
    for name, config in CONFIGS.items():
        payload = torch.load(output / (name + '.pt'), map_location='cpu', weights_only=True)
        if payload['architecture'] != ARCHITECTURE or payload['vocab'] != VOCAB or payload['config'] != list(config):
            raise ValueError('checkpoint architecture/vocabulary/config mismatch')
        model = TinyLanguageModel(*config)
        model.load_state_dict(payload['state_dict'], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite checkpoint parameters')
        model.eval()
        for group, rows in [('train', TRAIN), ('probe', PROBE)]:
            actual = evaluate(model, rows)[1]
            if not torch.allclose(actual, payload['reference_logits'][group], atol=1e-12, rtol=0):
                raise ValueError(f'{name}/{group} restored logits mismatch')
        restored.append(name)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump({'restored': restored, 'atol': 1e-12, 'rtol': 0, 'training_executed': False}, stream, indent=2)
    print('RESTORE=PASS (4 models, train/probe logits, fresh process) | SAVED verify.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'check'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(f'DEVICE=cpu DTYPE=float64 TORCH={torch.__version__}', flush=True)
    (train if args.mode == 'train' else check)(args.output)
'''
with Path('compare_structure.py').open('x', encoding='utf-8') as stream:
    stream.write(code)
print('CREATED compare_structure.py')
PY
```

## 5. 학습 후 새 프로세스 복원

```bash
python -B compare_structure.py train ../tiny_structure_v1 &&
python -B compare_structure.py check ../tiny_structure_v1
```

- `train`: 기준/세 변형을 각각 새 초기값에서200회 학습하고 train/probe 마지막 위치 정답 수·loss를 계산한다. `baseline.pt`, `heads2.pt`, `layers2.pt`, `context2.pt`와 `report.json`을 새 폴더에 저장한다.
- `check`: 별도 Python 프로세스에서 구조·어휘·config를 확인하고 `weights_only=True`로 읽는다. 모든 파라미터의 유한값, train/probe logits의 저장 전후 일치(`atol=1e-12`, `rtol=0`)를 확인한 뒤 새 `verify.json`을 쓴다. optimizer 갱신/새 학습은 없다.
- `&&`: 앞 명령이 성공했을 때만 다음 명령을 실행한다. `check`는 이미 있는 검증 보고서도 덮어쓰지 않는다.
- 새 폴더/체크포인트는 저장 도중 중단되면 일부만 생길 수 있다. 원자적 장애복구 시스템은 아니며, 자동 삭제·재학습하지 않는다. trusted local 실습용 저장물만 읽는다.

주요 shape는 `ids=(B,T)`, 임베딩/블록입출력`(B,T,4)`, 각 Q/K/V`(B,H,T,4/H)`, Attention`(B,H,T,T)`, logits`(B,T,7)`이다. loss에는 `logits[:, -1]`의`(B,7)`과 정답`(B,)`를 전달한다. 미래 위치를`-inf`로 막은 뒤 softmax하므로 앞 토큰은 뒤 입력을 보지 못한다. 모든 블록에서 이 마스크를 다시 적용한다.

## 6. 실제 로컬 CPU 실행 예시

다음은 Python3.12·PyTorch2.9.0+cpu의 실제 학습과 별도 프로세스 복원 결과다. **VM 실행 완료를 뜻하지 않는다.**

```text
DEVICE=cpu DTYPE=float64 TORCH=2.9.0+cpu
baseline heads=1 layers=1 context=4 params=212 train=4/4 loss=0.002856 probe=4/4 loss=0.044103
heads2 heads=2 layers=1 context=4 params=212 train=4/4 loss=0.003508 probe=4/4 loss=0.051355
layers2 heads=1 layers=2 context=4 params=352 train=4/4 loss=0.001738 probe=4/4 loss=0.016900
context2 heads=1 layers=1 context=2 params=212 train=2/4 loss=0.694796 probe=2/4 loss=0.694796
SAVED report.json + 4 checkpoints | CPU only; no old models read/changed; probe is diagnostic
DEVICE=cpu DTYPE=float64 TORCH=2.9.0+cpu
RESTORE=PASS (4 models, train/probe logits, fresh process) | SAVED verify.json
```

이 실행에서는 head를 늘려도 정답 수는 늘지 않았고, 두 층 모델의 loss는 낮았지만 정답 수는 기준과 같았다. 문맥2에서는 실제로 상한2/4에 도달했다. **층이 많으면 항상 좋다거나 head가 많으면 정답률이 오른다는 결론은 아니다.** 단일 seed·아주 작은 합성 입력·고정 학습량의 관찰이다. probe는 진단용이며 이 결과로 모델 선택을 하지 않았다. 장치·버전·수치 연산 차이로 마지막 loss 표시가 다를 수 있으므로 수치를 맞추려고 설정을 튜닝하지 않는다.

이 로컬 격리 환경에는 NumPy가 없어 Torch의 `Failed to initialize NumPy` 경고가 있었지만 NumPy 경로는 사용하지 않았고 실제 텐서 학습·검증은 종료0으로 완료됐다. 경고 제거를 위해 VM 패키지를 바꾸지 않는다.

실제 Torch 검사는 이전 기준 초기 logits 동등성, 각 head/층의 미래 비중0·합1, prefix 일치·미래 토큰 변경 시 앞 출력 불변, 유한 기울기, 입력 충돌 상한, stdlib logsumexp 기반 독립 loss 계산, 실제 학습/별도 프로세스 복원, 기존 출력 보존·중복 실행 거부·참조 logits 손상 거부를 통과했다. 테스트는 별도 임시 폴더를 사용한다.

```bash
python -B source/34-structure-comparison/test_structure.py
```

위 테스트 명령은 저장소 루트에서, Torch가 설치된 환경으로 실행한다. VM의 flat 단일 파일 배치와는 구분한다. VM에는 비교 코드 하나만 있으면 된다.

### 별도 Ubuntu VM 실행 예시

별도 VM에서도 PyTorch `2.11.0+cu128`, `DEVICE=cpu`, `DTYPE=float64`로 다음 출력과 셸 복귀를 확인했다. 패키지 버전의 CUDA 표기와 실제 실행 장치를 구분한다. 이번 실습은 GPU 실행이나 GPU 성능 측정이 아니다. 다음은 제공된 콘솔 출력이며 VM의 전체 보고서·체크포인트를 직접 읽은 결과는 아니다.

```text
CREATED compare_structure.py
DEVICE=cpu DTYPE=float64 TORCH=2.11.0+cu128
baseline heads=1 layers=1 context=4 params=212 train=4/4 loss=0.002856 probe=4/4 loss=0.044103
heads2 heads=2 layers=1 context=4 params=212 train=4/4 loss=0.003508 probe=4/4 loss=0.051355
layers2 heads=1 layers=2 context=4 params=352 train=4/4 loss=0.001738 probe=4/4 loss=0.016900
context2 heads=1 layers=1 context=2 params=212 train=2/4 loss=0.694796 probe=2/4 loss=0.694796
SAVED report.json + 4 checkpoints | CPU only; no old models read/changed; probe is diagnostic
DEVICE=cpu DTYPE=float64 TORCH=2.11.0+cu128
RESTORE=PASS (4 models, train/probe logits, fresh process) | SAVED verify.json
```

네 모델의 학습·진단 비교, 보고서와 체크포인트4개 저장, 새 프로세스의 train/probe logits 복원 검사와 `verify.json` 저장까지 완료됐다. 표시된 loss와 정답 수는 위 로컬 예시와 같지만 가중치나 전체 logits가 두 환경 사이에서 비트 단위로 같다는 뜻은 아니다. 이번 VM 실행에서는 head 수를 늘려 정답 수가 개선되지는 않았고, 두 층은 loss가 낮아도 정답 수는 기준과 같았다. 문맥2의2/4는 오류가 아니라 장치 이름을 제거한 입력 충돌 상한에 도달한 결과다. 진단 probe를 독립 test로 해석하거나 기존 QLoRA 품질 개선으로 확대하지 않는다.

## 7. 확인할 것

- 네 모델의 `heads/layers/context/params`, 학습/진단 정답 수와 loss를 비교한다.
- `SAVED report.json + 4 checkpoints`와 별도 프로세스의 `RESTORE=PASS`, `SAVED verify.json`을 확인한다.
- 문맥2의 입력이 왜 같아졌는지 읽고, 정보 부족과 optimizer/복원 오류를 구분한다.
- 이 실습의 저장은 추론용 가중치와 참조 출력이다.33장처럼 optimizer/scheduler/RNG를 담은 학습 상태 재개가 아니다.

기존 QLoRA의 정답률·품질 게이트·배포 상태는 이번 작은 모델의 결과와 별개로 유지한다. RAG·새 운영 프레임워크·추가 모델 튜닝은 포함하지 않는다.

[← 이전: 학습 재개](33-training-resume.md) · [목차](README.md)
