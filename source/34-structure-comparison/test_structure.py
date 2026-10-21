"""Real-tensor structure/causality checks and fresh-process save/restore."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import torch

script = Path(__file__).with_name('01-compare-structure.py')
assert script.is_file(), '구조 비교 실습이 아직 없습니다'
m = runpy.run_path(str(script))
torch.set_num_threads(1)
old = runpy.run_path(str(script.parent.parent / '09-transformer/03-checkpoint-generate.py'))
torch.manual_seed(7)
previous = old['TinyLanguageModel']()
torch.manual_seed(7)
base = m['TinyLanguageModel']()
x = torch.tensor([[old['to_id'][t] for t in 'gpu current status failed'.split()]])
assert torch.allclose(base(x)[0], previous(x)[0], atol=1e-12, rtol=0)
for heads, layers, context in m['CONFIGS'].values():
    model = m['TinyLanguageModel'](heads, layers, context)
    logits, weights = model(x[:, :context])
    changed = x[:, :context].clone()
    changed[:, -1] = old['to_id']['network']
    assert logits.shape == (1, context, len(m['VOCAB']))
    assert len(weights) == layers
    for a in weights:
        assert a.shape == (1, heads, context, context)
        assert torch.all(a.triu(1) == 0)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
    assert torch.allclose(logits[:, :-1], model(changed)[0][:, :-1], atol=1e-12, rtol=0)
    for length in range(1, context + 1):
        assert torch.allclose(model(x[:, :length])[0][:, -1], logits[:, length - 1], atol=1e-12, rtol=0)
    loss = model(x[:, :context])[0].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
for bad in [(3, 1, 4), (True, 1, 4), (1, 0, 4), (1, 1, 5)]:
    try:
        m['TinyLanguageModel'](*bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid config accepted')
for context, expected in [(2, 2), (4, 4)]:
    encoded, labels = m['batch'](m['TRAIN'], context)
    assert m['collision_bound'](encoded, labels) == expected
assert m['batch'](m['TRAIN'], 2)[0].tolist() == [[m['TO_ID'][t] for t in s.split()[-2:]] for s, _ in m['TRAIN']]
with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
    out = Path(directory) / 'comparison'
    def invoke(*args):
        return subprocess.run([sys.executable, '-B', str(script), *map(str, args)], capture_output=True, text=True)
    result = invoke('train', out)
    print(result.stdout, end='')
    assert result.returncode == 0, result.stderr
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    result = invoke('check', out)
    print(result.stdout, end='')
    assert result.returncode == 0, result.stderr
    assert json.loads((out / 'verify.json').read_text())['restored'] == list(m['CONFIGS'])
    report = json.loads((out / 'report.json').read_text())
    for name, result in report['results'].items():
        assert result['train']['correct'] <= result['train']['collision_bound']
        assert result['probe']['correct'] <= result['probe']['collision_bound']
        payload = torch.load(out / (name + '.pt'), weights_only=True)
        for group, rows in [('train', m['TRAIN']), ('probe', m['PROBE'])]:
            scores = payload['reference_logits'][group]
            labels = [m['TO_ID'][label] for _, label in rows]
            from math import exp, log
            losses = []
            for row, target in zip(scores.tolist(), labels):
                peak = max(row)
                losses.append(log(sum(exp(value - peak) for value in row)) - (row[target] - peak))
            assert abs(sum(losses) / len(losses) - result[group]['loss']) < 1e-12
            assert sum(int(row.argmax()) == target for row, target in zip(scores, labels)) == result[group]['correct']
    evidence = {'report': report, 'verify': json.loads((out / 'verify.json').read_text()), 'test_pass': False}
    assert all(hashlib.sha256((out / name).read_bytes()).hexdigest() == digest for name, digest in before.items())
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert invoke('train', out).returncode != 0
    assert invoke('check', out).returncode != 0
    assert after == {p.name: p.read_bytes() for p in out.iterdir()}
    payload = torch.load(out / 'baseline.pt', weights_only=True)
    payload['reference_logits']['train'][0, 0] += 1
    torch.save(payload, out / 'baseline.pt')
    (out / 'verify.json').unlink()
    assert invoke('check', out).returncode != 0
    assert not (out / 'verify.json').exists()
print('PASS historical baseline equivalence, multihead/stack causality/gradients, collision bound, real train/fresh restore, preservation/refusals/corruption')
if len(sys.argv) == 3 and sys.argv[1] == '--evidence':
    evidence['test_pass'] = True

    with Path(sys.argv[2]).open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
