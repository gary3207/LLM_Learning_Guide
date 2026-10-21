"""CPU float64: change heads, depth or visible context; save and restore separately."""
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
