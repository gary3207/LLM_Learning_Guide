"""Bounded single-writer deployment. systemd --user owns the process; no reboot/crash recovery.
Registry selection is a NEXT-candidate pointer and is never changed here.
Sequential restarts have downtime. Evidence hashes are not signatures or backups.
"""
import importlib.util
import json
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
path = HERE / 'smollm2_pipeline.py'
if not path.is_file():
    path = HERE.parent / '27-evaluation-pipeline/01-pipeline.py'
spec = importlib.util.spec_from_file_location('smollm2_pipeline', path)
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
settings = sys.pycache_prefix, sys.dont_write_bytecode
try:
    sys.pycache_prefix = str(HERE / ('.uncached-' + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    spec.loader.exec_module(pipeline)
finally:
    sys.pycache_prefix, sys.dont_write_bytecode = settings
tracker, registry = pipeline.tracker, pipeline.registry
selected = tracker.gate.selected
def sources():
    return dict(tracker.sources(), pipeline=dict(path=str(Path(pipeline.__file__).resolve()), **registry.fingerprint(Path(pipeline.__file__))),
                deployment=dict(path=str(Path(__file__).resolve()), **registry.fingerprint(Path(__file__))),
                template=dict(path=str(HERE / 'llm-lab-serving.service'), **registry.fingerprint(HERE / 'llm-lab-serving.service')))


LOADED = sources()


def validate(run, store):
    run, store = Path(run), Path(store)
    files = [run / name for name in ('pipeline.json', 'run.json', 'quality.json')]
    if run.is_symlink() or any(p.is_symlink() for p in files):
        raise ValueError('symlink evidence is unsupported')
    before = [registry.fingerprint(p) for p in files]
    decision, receipt, report = [registry.read_json(p) for p in files]
    tracker.validate_retained_receipt(receipt, run.name)
    tracker.validate_retained_report(report, receipt)
    approved = receipt['approved']
    stage, handoff, code = ('approved', 'ready_for_next_step', 0) if approved else ('blocked', 'halted_quality', 2)
    inputs = receipt['inputs']
    if (type(decision['schema']) is not int or decision['schema'] != 1 or
            decision['run_id'] != run.name or decision['decision'] != handoff or
            type(decision['exit_code']) is not int or decision['exit_code'] != code or
            decision['stages'] != dict(Evaluate=stage, Verify='passed', Decision=handoff) or
            decision['followup'] != 'skipped' or decision['deployed'] is not False or
            decision['run_file'] != before[1] or decision['quality_file'] != before[2] or
            receipt['quality_file'] != before[2] or decision['selection'] != inputs['selection'] or
            decision['pipeline_source'] != LOADED['pipeline'] or inputs['sources'] != tracker.LOADED_SOURCES or
            sources() != LOADED):
        raise ValueError('pipeline/evidence/source linkage mismatch')
    record = registry.verify(store, inputs['selection']['version'])
    if (record != inputs['registry_record'] or registry.fingerprint(registry.version_path(store, record['version']))['sha256'] != inputs['selection']['manifest_sha256'] or
            inputs['generation'] != tracker.gate.GENERATION):
        raise ValueError('explicit model manifest/config mismatch')
    import hashlib
    dataset = hashlib.sha256(json.dumps(tracker.gate.scorer.ROWS, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if inputs['dataset_sha256'] != dataset or before != [registry.fingerprint(p) for p in files] or sources() != LOADED:
        raise ValueError('inputs/evidence changed during validation')
    return approved, dict(run=str(run.resolve()), registry=str(store.resolve()), identity=inputs['selection'],
                          evidence=dict(zip(('pipeline.json', 'run.json', 'quality.json'), before)), sources=sources())


def control(*args):
    import subprocess
    try:
        return subprocess.run(['systemctl', '--user', *args], check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'{exc}: {(exc.stderr or "").strip()}') from exc


def write_json(path, value, exclusive=False):
    import os
    import tempfile
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def quote(argument):
    text = str(argument)
    if any(ord(c) < 32 or ord(c) == 127 for c in text):
        raise ValueError('control characters in systemd argument')
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def render(state):
    command = ' '.join(quote(arg) for arg in (Path(sys.executable).absolute(), '-B', Path(__file__).resolve(), 'serve', '--state', state))
    return (HERE / 'llm-lab-serving.service').read_text(encoding='utf-8').replace('@COMMAND@', command)


def options(device, port, unit, timeout):
    import math
    import re
    if (device not in ('cpu', 'cuda') or type(port) is not int or not 1 <= port <= 65535 or
            not isinstance(unit, str) or not re.fullmatch(r'llm-lab-serving(?:-test-[0-9a-f]{32})?\.service', unit) or
            type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('invalid device/port/dedicated unit/startup timeout')


def safe_state(state, release):
    state = Path(state)
    if any(p.is_symlink() for p in (state, *state.parents)):
        raise ValueError('symlink state path')
    state = state.resolve()
    record = registry.verify(Path(release['registry']), release['identity']['version'])
    source_root = HERE if path.parent == HERE else HERE.parent
    protected = [source_root, Path(release['run']).parent, Path(release['registry']),
                 Path(record['model_directory']), Path(record['report_path'])]
    if any(state.is_relative_to(p.resolve()) or p.resolve().is_relative_to(state) for p in protected):
        raise ValueError('state overlaps read-only inputs/source')
    if state.exists() and (not state.is_dir() or any(p.is_symlink() for p in state.rglob('*'))):
        raise ValueError('unsafe existing state')
    return state


def verified(release):
    approved, current = validate(Path(release['run']), Path(release['registry']))
    if not approved or any(current[key] != release[key] for key in current):
        raise ValueError('approved release evidence/source changed')
    return release


def check_http(release, timeout):
    import urllib.request
    base = 'http://127.0.0.1:' + str(release['port'])
    result = {}
    for endpoint, body in (('health', None), ('generate', {'prompt': 'Hello', 'max_new_tokens': 1})):
        request = urllib.request.Request(base + '/' + endpoint, data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get('model_version') != release['identity']:
            raise ValueError('HTTP full model identity mismatch')
        if endpoint == 'health' and data.get('status') != 'ok':
            raise ValueError('unhealthy model')
        if endpoint == 'generate' and (not isinstance(data.get('text'), str) or data.get('generated_tokens') != 1):
            raise ValueError('invalid generation smoke response')
        result[endpoint] = data
    return result


def start(state, config, release, timeout):
    import time
    import urllib.error
    verified(release)
    instance = uuid.uuid4().hex
    write_json(state / 'launch.json', dict(release, deployment_instance=instance))
    control('restart', config['unit'])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = control('show', config['unit'], '--property=ActiveState', '--value')
        if status in ('failed', 'inactive'):
            raise RuntimeError('owned unit failed during startup')
        try:
            response = check_http(release, min(2, max(.01, deadline - time.monotonic())))
            if any(data.get('deployment_instance') != instance for data in response.values()):
                raise ValueError('HTTP response is not from this new deployment instance')
            if time.monotonic() >= deadline:
                raise TimeoutError('startup deadline exceeded')
            verified(release)
            if control('show', config['unit'], '--property=ActiveState', '--value') != 'active':
                raise RuntimeError('owned service exited during smoke check')
            return response
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
    raise TimeoutError('startup health/generation deadline exceeded')


def deploy(run, store, state, device='cuda', port=8001, unit='llm-lab-serving.service', timeout=120):
    options(device, port, unit, timeout)
    approved, release = validate(run, store)
    if not approved:
        return 2
    release.update(device=device, port=port)
    state = safe_state(state, release)
    config = dict(schema=1, state=str(state), unit=unit, content=render(state), sources=sources(), device=device, port=port)
    if state.exists():
        if registry.read_json(state / 'owner.json') != config or (state / unit).read_text(encoding='utf-8') != config['content']:
            raise ValueError('unowned state or changed deployment code/unit/config')
    fragment = control('show', unit, '--property=FragmentPath', '--value')
    if fragment and (not state.exists() or Path(fragment).resolve() != state / unit):
        raise ValueError('refusing unrelated existing unit')
    previous = registry.read_json(state / 'active.json')['current'] if (state / 'active.json').exists() else None
    if previous is not None:
        verified(previous)
    else:
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))  # Refuse occupied listener; never kill/adopt it.
    if not state.exists():
        state.mkdir(parents=True)
        write_json(state / 'owner.json', config, exclusive=True)
        with (state / unit).open('x', encoding='utf-8') as stream:
            stream.write(config['content'])
    return transition(state, config, release, previous, timeout, 'deploy', not fragment)


def owner(state):
    state = Path(state)
    if any(p.is_symlink() for p in (state, *state.parents)) or any(p.is_symlink() for p in state.rglob('*')):
        raise ValueError('symlink deployment state')
    state = state.resolve()
    config = registry.read_json(state / 'owner.json')
    options(config['device'], config['port'], config['unit'], 1)
    expected = dict(schema=1, state=str(state), unit=config['unit'], content=render(state), sources=LOADED,
                    device=config['device'], port=config['port'])
    if config != expected or sources() != LOADED or (state / config['unit']).read_text(encoding='utf-8') != config['content']:
        raise ValueError('unowned state or changed deployment code/unit/config')
    return state, config


def rollback(state, timeout=120):
    state, config = owner(state)
    options(config['device'], config['port'], config['unit'], timeout)
    active = registry.read_json(state / 'active.json')
    current, previous = active['current'], active['previous']
    if previous is None:
        raise ValueError('no previous approved deployment available')
    verified(previous)
    safe_state(state, previous)
    if any(previous[k] != config[k] for k in ('device', 'port')):
        raise ValueError('previous release configuration mismatch')
    if Path(control('show', config['unit'], '--property=FragmentPath', '--value')).resolve() != state / config['unit']:
        raise ValueError('refusing unrelated/missing unit')
    return transition(state, config, previous, current, timeout, 'rollback')


class DeploymentHandler(selected.Handler):
    def send_json(self, status, data):
        if status == 200:
            data = dict(data, deployment_instance=self.server.deployment_instance)
        super().send_json(status, data)


def serve(state):
    state, config = owner(state)
    release = verified(registry.read_json(state / 'launch.json'))
    tracker.retained_hex(release['deployment_instance'], 32)
    safe_state(state, release)
    if any(release[k] != config[k] for k in ('device', 'port')):
        raise ValueError('launch configuration mismatch')
    record = registry.verify(Path(release['registry']), release['identity']['version'])
    api = selected.api
    if record['base_model'] != api.MODEL or record['revision'] != api.REVISION:
        raise ValueError('unsupported base model/revision')
    tokenizer, model, count = api.load_model(Path(record['model_directory']), release['device'])
    verified(release)  # Includes all source/model/evidence hashes after loading, before binding.
    with api.HTTPServer(('127.0.0.1', release['port']), DeploymentHandler) as server:
        server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
        server.identity, server.metrics = dict(release['identity']), False
        server.deployment_instance = release['deployment_instance']
        print('READY', json.dumps(dict(port=server.server_port, model_version=server.identity, quantized_modules=count)), flush=True)
        server.serve_forever()


def transition(state, config, candidate, previous, timeout, operation, install=False):
    # ponytail: one writer; no crash-safe multi-file transaction or zero-downtime handover.
    attempt = state / (uuid.uuid4().hex + '-attempt.json')
    write_json(attempt, dict(candidate=candidate, previous=previous, operation=operation), exclusive=True)
    result = dict(exit_code=1)
    try:
        if install:
            control('link', str(state / config['unit']))
            control('daemon-reload')
        result['response'] = start(state, config, candidate, timeout)
        verified(candidate)
        write_json(state / 'active.json', dict(current=candidate, previous=previous))
        result['exit_code'] = 0
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        try:
            if previous is None:
                control('stop', config['unit'])
                result['recovery'] = 'unavailable_no_previous'
            else:
                result['recovery_response'] = start(state, config, verified(previous), timeout)
                result['recovery'] = 'restored_previous'
        except Exception as recovery:
            result.update(recovery='failed', recovery_error=f'{type(recovery).__name__}: {recovery}')
            try:
                control('stop', config['unit'])
            except Exception as stop:
                result['stop_error'] = f'{type(stop).__name__}: {stop}'
    write_json(attempt.with_name(attempt.name.replace('-attempt', '-result')), result, exclusive=True)
    return result['exit_code']


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    deployment = commands.add_parser('deploy', help='consume one explicit chapter27 run; never reevaluate/train')
    deployment.add_argument('--run', type=Path, required=True)
    deployment.add_argument('--registry', type=Path, required=True)
    deployment.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    deployment.add_argument('--port', type=int, default=8001)
    deployment.add_argument('--unit', default='llm-lab-serving.service')
    for command in (deployment, commands.add_parser('rollback'), commands.add_parser('serve')):
        command.add_argument('--state', type=Path, default=Path('../serving_deployment_v1'))
        if command.prog.split()[-1] != 'serve':
            command.add_argument('--timeout', type=float, default=120)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        if args.command == 'serve':
            serve(args.state)
            return 0
        if args.command == 'rollback':
            return rollback(args.state, args.timeout)
        return deploy(args.run, args.registry, args.state, args.device, args.port, args.unit, args.timeout)
    except Exception as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
