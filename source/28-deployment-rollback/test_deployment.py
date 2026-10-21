"""Stdlib checks; synthetic approvals are NOT learned quality or model inference."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get('TMPDIR') or tempfile.gettempdir())
SCRATCH.mkdir(parents=True, exist_ok=True)
script = HERE / '01-deploy-rollback.py'
assert script.is_file(), 'chapter28 implementation missing'
spec = importlib.util.spec_from_file_location('deployment', script)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
pipeline, tracker, registry = deploy.pipeline, deploy.tracker, deploy.registry
gate = tracker.gate
previous = HERE.parent / '26-run-tracking/test_tracking.py'
tree = ast.parse(previous.read_text(encoding='utf-8'))
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'hashes', 'fixture', 'synthetic', 'flat_copy'}]
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(previous), 'exec'))
hashes, fixture, synthetic, flat_copy = (globals()[name] for name in ('hashes', 'fixture', 'synthetic', 'flat_copy'))


def make_run(root, store, approved=False):
    def boundary(*args, **kwargs):
        report = synthetic(*args, **kwargs)
        if approved:
            for row in report['rows']:
                row['text'] = row['answer']
            report['summary'], report['approved'] = gate.decide(report['rows'])
            args[2].write_text(json.dumps(report), encoding='utf-8')
        return report
    with patch.object(gate, 'run', side_effect=boundary):
        result = pipeline.run(root / 'runs', store, 'cpu')
    return root / 'runs' / result['run_id']


def invoke(run, store, state, **kwargs):
    return deploy.deploy(run, store, state, 'cpu', kwargs.get('port', 18001), kwargs.get('unit', 'llm-lab-serving.service'), kwargs.get('timeout', 2))


def test_scratch_location():
    assert SCRATCH.resolve() == Path(os.environ.get('TMPDIR') or tempfile.gettempdir()).resolve(), 'test scratch must follow TMPDIR/platform temp, not a personal host path'
    print('PASS portable test scratch follows TMPDIR/platform temp')


def test_render_keeps_virtual_environment():
    if os.name != 'posix':
        return  # Actual symlinked Python environment is checked by the Linux run.
    import subprocess
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        environment = root / 'venv with spaces'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)], check=True)
        python = environment / 'bin/python'
        assert python.is_symlink(), 'Linux regression requires the ordinary symlinked venv executable'
        probe = subprocess.run([str(python), '-c', 'import sys; print(sys.prefix)'], capture_output=True, text=True, check=True)
        assert Path(probe.stdout.strip()).resolve() == environment.resolve()
        with patch.object(sys, 'executable', str(python)):
            command = deploy.render(root / 'state')
        assert deploy.quote(python.absolute()) in command, 'unit resolved venv Python to the package-less base interpreter'
    print('PASS native Linux unit preserves symlinked Python virtual environment')


def test_blocked_no_changes():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        _, store, _ = fixture(root)
        run = make_run(root, store)
        before = hashes(root)
        with patch.object(deploy, 'control', side_effect=AssertionError('service control forbidden')) as ctl:
            assert invoke(run, store, root / 'state') == 2
            ctl.assert_not_called()
        assert hashes(root) == before and not (root / 'state').exists()
    print('PASS synthetic blocked quality: exit2, no state/unit/control/input changes')


class TransportService:
    """TEST ONLY subprocess adapter, not production lifecycle or real systemd."""
    def __init__(self, state, real_model=False):
        self.state, self.process, self.fragment = state, None, ''
        self.real_model = real_model
        self.fail = set()
        self.calls = []

    def __call__(self, *args):
        import subprocess
        self.calls.append(args)
        if args[0] == 'show':
            if 'FragmentPath' in args[-2]:
                return self.fragment
            return 'active' if self.process and self.process.poll() is None else 'failed'
        if args[0] == 'link':
            self.fragment = args[1]
        if args[0] in ('stop', 'restart'):
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.process = None
        if args[0] == 'restart':
            launch = registry.read_json(self.state / 'launch.json')
            if launch['identity']['version'] not in self.fail:
                command = [sys.executable, '-B', str(script), 'serve', '--state', str(self.state)] if self.real_model else [sys.executable, '-B', str(Path(__file__).resolve()), '--fixture-server', str(self.state)]
                import uuid
                with (self.state / ('test-process-' + uuid.uuid4().hex + '.log')).open('xb') as log:
                    self.process = subprocess.Popen(command, stdout=log, stderr=log)
        return ''


def fixture_server(state):
    """TEST ONLY stdlib identity transport: no ML loader or inference."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    launch = json.loads((Path(state) / 'launch.json').read_text())
    failure = Path(state) / 'fail-versions.json'
    if '--fail-startup' in sys.argv or (failure.exists() and launch['identity']['version'] in json.loads(failure.read_text())):
        raise RuntimeError('TEST ONLY requested transport startup failure')
    delay = Path(state) / 'test-startup-delay.json'
    if delay.exists():
        import time
        time.sleep(json.loads(delay.read_text()))  # TEST ONLY: Type=exec active before bind.
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond({'status': 'ok', 'model_version': launch['identity']})
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert request == {'prompt': 'Hello', 'max_new_tokens': 1}
            self.respond({'text': 'TEST ONLY transport', 'generated_tokens': 1, 'model_version': launch['identity']})
        def respond(self, data):
            body = json.dumps(dict(data, deployment_instance=launch['deployment_instance'])).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    HTTPServer(('127.0.0.1', launch['port']), Handler).serve_forever()


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_approved_http():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        _, store, _ = fixture(root)
        run = make_run(root, store, True)
        before = hashes(root)
        state, port = root / 'owned state $ %', free_port()
        service = TransportService(state)
        with patch.object(deploy, 'control', side_effect=service):
            try:
                assert invoke(run, store, state, port=port) == 0
                active = registry.read_json(state / 'active.json')
                assert active['current']['identity'] == registry.read_json(run / 'pipeline.json')['selection']
                assert active['previous'] is None
                assert deploy.check_http(active['current'], 2)['generate']['generated_tokens'] == 1
                wrong = dict(active['current'], identity=dict(active['current']['identity'], manifest_sha256='0' * 64))
                try:
                    deploy.check_http(wrong, 2)
                except ValueError:
                    pass
                else:
                    raise AssertionError('same version with wrong full identity accepted')
                assert all(hashes(root)[name] == digest for name, digest in before.items())
            finally:
                service('stop', 'llm-lab-serving.service')
    print('PASS synthetic approval + real stdlib HTTP full identity; TEST adapter lifecycle, NOT systemd/ML')


def test_failed_candidate_recovery():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        model, store, performance = fixture(root)
        run_a = make_run(root, store, True)
        registry.register(store, 'v2', model, performance)
        registry.select(store, 'v2')
        run_b = make_run(root, store, True)
        before = hashes(root)
        state, port = root / 'state', free_port()
        service = TransportService(state)
        with patch.object(deploy, 'control', side_effect=service):
            try:
                assert invoke(run_a, store, state, port=port) == 0
                original = (state / 'active.json').read_bytes()
                service.fail.add('v2')
                assert invoke(run_b, store, state, port=port) == 1
                assert (state / 'active.json').read_bytes() == original
                active = registry.read_json(state / 'active.json')['current']
                assert deploy.check_http(active, 2)['health']['model_version']['version'] == 'v1'
                results = [registry.read_json(p) for p in state.glob('*-result.json')]
                failed = next(x for x in results if x['exit_code'] == 1)
                assert failed['recovery'] == 'restored_previous' and failed['error']
                assert all(hashes(root)[name] == digest for name, digest in before.items())
            finally:
                service('stop', 'llm-lab-serving.service')
        initial = root / 'initial failure'
        service = TransportService(initial)
        service.fail.add('v2')
        with patch.object(deploy, 'control', side_effect=service):
            assert invoke(run_b, store, initial, port=free_port()) == 1
            assert not (initial / 'active.json').exists()
            failed = registry.read_json(next(initial.glob('*-result.json')))
            assert failed['recovery'] == 'unavailable_no_previous' and service.process is None
    print('PASS TEST transport candidate failure exit1 + verified previous restoration; initial failure no fabricated rollback')


def test_explicit_rollback():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        model, store, performance = fixture(root)
        run_a = make_run(root, store, True)
        registry.register(store, 'v2', model, performance)
        registry.select(store, 'v2')
        run_b = make_run(root, store, True)
        before = hashes(root)
        state, port = root / 'state', free_port()
        service = TransportService(state)
        with patch.object(deploy, 'control', side_effect=service):
            try:
                assert invoke(run_a, store, state, port=port) == 0
                assert invoke(run_b, store, state, port=port) == 0
                assert deploy.rollback(state, 2) == 0
                active = registry.read_json(state / 'active.json')
                assert active['current']['identity']['version'] == 'v1'
                assert active['previous']['identity']['version'] == 'v2'
                assert deploy.check_http(active['current'], 2)['health']['model_version'] == active['current']['identity']
                assert all(hashes(root)[name] == digest for name, digest in before.items())
                assert len(list(state.glob('*-attempt.json'))) == 3
                assert len(list(state.glob('*-result.json'))) == 3
            finally:
                service('stop', 'llm-lab-serving.service')
    print('PASS synthetic A->B->explicit rollback A, full HTTP identities, unchanged registry selection, exclusive journals')


def test_rollback_from_corrupt_current():
    import shutil
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        model, store, performance = fixture(root)
        run_a = make_run(root, store, True)
        model_b = root / 'model-b'
        shutil.copytree(model, model_b)
        registry.register(store, 'v2', model_b, performance)
        registry.select(store, 'v2')
        run_b = make_run(root, store, True)
        state, port = root / 'state', free_port()
        service = TransportService(state)
        with patch.object(deploy, 'control', side_effect=service):
            try:
                assert invoke(run_a, store, state, port=port) == 0
                assert invoke(run_b, store, state, port=port) == 0
                (model_b / 'adapter_model.safetensors').write_text('corrupt current model')
                assert deploy.rollback(state, 2) == 0
                active = registry.read_json(state / 'active.json')
                assert active['current']['identity']['version'] == 'v1'
                assert deploy.check_http(active['current'], 2)['health']['model_version']['version'] == 'v1'
            finally:
                service('stop', 'llm-lab-serving.service')
    print('PASS rollback validates previous A, not corrupt currently-active B; B cannot be reused without revalidation')


def test_serve_rechecks_before_bind():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        model, store, _ = fixture(root)
        run = make_run(root, store, True)
        state, port = root / 'state', free_port()
        service = TransportService(state)
        with patch.object(deploy, 'control', side_effect=service):
            try:
                assert invoke(run, store, state, port=port) == 0
            finally:
                service('stop', 'llm-lab-serving.service')
        def mutate(*args):
            (model / 'adapter_model.safetensors').write_text('changed', encoding='utf-8')
            return None, None, 210
        with patch.object(deploy.selected.api, 'load_model', side_effect=mutate) as loader, patch.object(deploy.selected.api, 'HTTPServer', side_effect=AssertionError('bound changed inputs')) as server:
            try:
                deploy.serve(state)
            except ValueError:
                pass
            else:
                raise AssertionError('model mutation accepted')
            loader.assert_called_once_with(model.resolve(), 'cpu')
            server.assert_not_called()
    print('PASS TEST load boundary model mutation rejected before bind; actual API loader is production-only path')


def test_unrelated_same_release_listener():
    import subprocess
    import time
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        _, store, _ = fixture(root)
        run = make_run(root, store, True)
        state, port = root / 'state', free_port()
        owned, unrelated = TransportService(state), TransportService(state)
        try:
            with patch.object(deploy, 'control', side_effect=owned):
                assert invoke(run, store, state, port=port) == 0
            original = (state / 'active.json').read_bytes()
            release = registry.read_json(state / 'active.json')['current']
            owned('stop', 'llm-lab-serving.service')
            unrelated('restart', 'llm-lab-serving.service')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    assert deploy.check_http(release, 1)['health']['model_version'] == release['identity']
                    break
                except OSError:
                    time.sleep(.02)
            else:
                raise AssertionError('unrelated transport did not start')
            def still_loading(*args):
                if args[0] == 'restart':
                    owned('stop', args[1])
                    owned.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return ''
                return owned(*args)
            with patch.object(deploy, 'control', side_effect=still_loading):
                assert invoke(run, store, state, port=port) == 1, 'unrelated same-release HTTP accepted as new owned instance'
            assert (state / 'active.json').read_bytes() == original
            assert unrelated.process is not None and unrelated.process.poll() is None, 'unrelated listener was stopped'
            assert owned.process is None
        finally:
            owned('stop', 'llm-lab-serving.service')
            unrelated('stop', 'llm-lab-serving.service')
    print('PASS unrelated same-release listener cannot satisfy a loading owned service; no false commit or unrelated stop')


def test_control_diagnostic():
    import subprocess
    error = subprocess.CalledProcessError(1, ['systemctl', '--user', 'restart', 'llm-lab-serving.service'], stderr='TEST native manager root cause')
    with patch.object(subprocess, 'run', side_effect=error):
        try:
            deploy.control('restart', 'llm-lab-serving.service')
        except Exception as exc:
            assert 'TEST native manager root cause' in str(exc), 'native service cause was lost'
        else:
            raise AssertionError('native service failure accepted')
    print('PASS native service failures preserve captured stderr cause')


def test_cli():
    import subprocess
    for args, code in ((['--help'], 0), (['--force'], 1), (['deploy', '--device', 'bad'], 1), (['rollback'], 1)):
        result = subprocess.run([sys.executable, '-B', str(script), *args], capture_output=True, text=True)
        assert result.returncode == code, (args, result.returncode, result.stderr)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        _, store, _ = fixture(root)
        run = make_run(root, store)
        before = hashes(root)
        result = subprocess.run([sys.executable, '-B', str(script), 'deploy', '--run', str(run), '--registry', str(store), '--state', str(root / 'state')], capture_output=True, text=True)
        assert result.returncode == 2, result.stdout + result.stderr
        assert hashes(root) == before
        for invalid in (['--port', '0'], ['--timeout', 'nan'], ['--unit', 'unowned.service']):
            result = subprocess.run([sys.executable, '-B', str(script), 'deploy', '--run', str(run), '--registry', str(store), '--state', str(root / 'state'), *invalid], capture_output=True, text=True)
            assert result.returncode == 1, (invalid, result.returncode)
        assert hashes(root) == before
    print('PASS CLI help0/badargs1/quality2; no force option or service control for blocked run')


def test_invalid_inputs_and_ownership():
    import socket
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        model, store, performance = fixture(root)
        run = make_run(root, store, True)
        for target in (run / 'pipeline.json', run / 'run.json', run / 'quality.json', model / 'adapter_model.safetensors', performance, registry.version_path(store, 'v1')):
            original = target.read_bytes()
            target.write_bytes(original + b' changed')
            before = hashes(root)
            with patch.object(deploy, 'control', side_effect=AssertionError('control before validation')) as ctl:
                try:
                    invoke(run, store, root / 'state')
                except (ValueError, KeyError):
                    pass
                else:
                    raise AssertionError('tamper accepted: ' + str(target))
                ctl.assert_not_called()
            assert hashes(root) == before and not (root / 'state').exists()
            target.write_bytes(original)
        # Re-link all hashes after falsifying a scored answer: validators, not hashes alone, reject it.
        originals = {p: p.read_bytes() for p in (run / 'quality.json', run / 'run.json', run / 'pipeline.json')}
        quality = registry.read_json(run / 'quality.json')
        quality['rows'][0]['text'] = 'wrong'
        (run / 'quality.json').write_text(json.dumps(quality), encoding='utf-8')
        receipt = registry.read_json(run / 'run.json')
        receipt['quality_file'] = registry.fingerprint(run / 'quality.json')
        (run / 'run.json').write_text(json.dumps(receipt), encoding='utf-8')
        handoff = registry.read_json(run / 'pipeline.json')
        handoff.update(run_file=registry.fingerprint(run / 'run.json'), quality_file=receipt['quality_file'])
        (run / 'pipeline.json').write_text(json.dumps(handoff), encoding='utf-8')
        with patch.object(deploy, 'control', side_effect=AssertionError('forged approval controlled service')) as ctl:
            try:
                invoke(run, store, root / 'state')
            except ValueError:
                pass
            else:
                raise AssertionError('rehashed false approval accepted')
            ctl.assert_not_called()
        for target, original in originals.items():
            target.write_bytes(original)
        before = hashes(root)
        for state in (model / 'new', store / 'new', run / 'new', performance, HERE / 'new'):
            with patch.object(deploy, 'control', side_effect=AssertionError('unsafe state service control')) as ctl:
                try:
                    invoke(run, store, state)
                except ValueError:
                    pass
                else:
                    raise AssertionError('unsafe state accepted')
                ctl.assert_not_called()
        assert hashes(root) == before
        with patch.object(deploy, 'control', return_value='/unrelated.service') as ctl:
            try:
                invoke(run, store, root / 'state')
            except ValueError:
                pass
            else:
                raise AssertionError('unowned unit adopted')
            assert [x.args[0] for x in ctl.call_args_list] == ['show']
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            with patch.object(deploy, 'control', return_value='') as ctl:
                try:
                    invoke(run, store, root / 'state', port=sock.getsockname()[1])
                except OSError:
                    pass
                else:
                    raise AssertionError('unowned listener adopted')
                assert [x.args[0] for x in ctl.call_args_list] == ['show']
            assert sock.fileno() >= 0
        assert hashes(root) == before and not (root / 'state').exists()
        for bad in ('x\n', 'x\0', 'x\t'):
            try:
                deploy.quote(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('control character accepted')
        assert deploy.quote('space $ % "') == '"space $ %% \\""'
    print('PASS tampered pipeline/run/quality/model/report/manifest rejected before control; unsafe paths, unrelated unit/listener untouched; unit argument escaping')


def test_flat_vm_state():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        _, store, _ = fixture(root)
        flat = flat_copy(root)
        copied_script = flat / 'smollm2_deploy.py'
        copied_script.write_bytes(script.read_bytes())
        (flat / 'smollm2_pipeline.py').write_bytes(Path(pipeline.__file__).read_bytes())
        (flat / 'llm-lab-serving.service').write_bytes((HERE / 'llm-lab-serving.service').read_bytes())
        spec = importlib.util.spec_from_file_location('flat_deployment', copied_script)
        assert spec is not None and spec.loader is not None
        copied = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copied)
        def approval(*args, **kwargs):
            report = synthetic(*args, **kwargs)
            for row in report['rows']:
                row['text'] = row['answer']
            report['summary'], report['approved'] = gate.decide(report['rows'])
            args[2].write_text(json.dumps(report), encoding='utf-8')
            return report
        with patch.object(copied.tracker.gate, 'run', side_effect=approval):
            result = copied.pipeline.run(root / 'runs', store, 'cpu')
        run = root / 'runs' / result['run_id']
        state, port = root / 'serving_deployment_v1', free_port()
        service = TransportService(state)
        with patch.object(copied, 'control', side_effect=service):
            try:
                assert copied.deploy(run, store, state, 'cpu', port, timeout=2) == 0
                assert registry.read_json(state / 'active.json')['current']['run'] == str(run.resolve())
                for unsafe in (flat / 'new', root):
                    try:
                        copied.safe_state(unsafe, copied.validate(run, store)[1])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('flat source or its ancestor accepted as writable state')
            finally:
                service('stop', 'llm-lab-serving.service')
    print('PASS flat VM source/ + sibling default state: approved path works; sources/ancestors remain protected')


def test_fresh_source_import():
    import py_compile
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        root = Path(tmp)
        (root / 'code').mkdir()
        flat = flat_copy(root / 'code')
        copied_script = flat / script.name
        copied_script.write_bytes(script.read_bytes())
        (flat / 'smollm2_pipeline.py').write_bytes(Path(pipeline.__file__).read_bytes())
        (flat / 'llm-lab-serving.service').write_bytes((HERE / 'llm-lab-serving.service').read_bytes())
        for target in (flat / 'smollm2_pipeline.py', flat / 'smollm2_quality_gate.py'):
            target.write_bytes(target.read_bytes() + b'\nCACHE_SENTINEL = 1\n')
            stat = target.stat()
            py_compile.compile(str(target), doraise=True)
            target.write_bytes(target.read_bytes().replace(b'CACHE_SENTINEL = 1', b'CACHE_SENTINEL = 2'))
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        before, settings = hashes(flat), (sys.pycache_prefix, sys.dont_write_bytecode)
        spec = importlib.util.spec_from_file_location('copied_deployment', copied_script)
        copied = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copied)
        assert copied.pipeline.CACHE_SENTINEL == copied.tracker.gate.CACHE_SENTINEL == 2
        assert hashes(flat) == before and settings == (sys.pycache_prefix, sys.dont_write_bytecode)
        # Change code after import; approval must not allow a restart.
        _, store, _ = fixture(root)
        with patch.object(copied.tracker.gate, 'run', side_effect=lambda *a, **kw: synthetic(*a, **kw)):
            receipt = copied.pipeline.run(root / 'runs', store, 'cpu')
        run = root / 'runs' / receipt['run_id']
        copied_script.write_bytes(copied_script.read_bytes() + b'\n# changed\n')
        with patch.object(copied, 'control', side_effect=AssertionError('changed source control')) as ctl:
            try:
                copied.deploy(run, store, root / 'state', 'cpu')
            except ValueError:
                pass
            else:
                raise AssertionError('changed source accepted')
            ctl.assert_not_called()
    print('PASS root+nested same-size/mtime stale pyc ignored, caches/settings unchanged; source mutation fails before control')


def clean_native(unit, state):
    import subprocess
    fragment = deploy.control('show', unit, '--property=FragmentPath', '--value')
    if fragment:
        assert Path(fragment).resolve() == state / unit
        deploy.control('stop', unit)
        deploy.control('disable', unit)
        deploy.control('daemon-reload')
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True)
    result = dict(load=deploy.control('show', unit, '--property=LoadState', '--value'), active=deploy.control('show', unit, '--property=ActiveState', '--value'), fragment=deploy.control('show', unit, '--property=FragmentPath', '--value'))
    assert result == dict(load='not-found', active='inactive', fragment=''), result
    return result


def test_native_systemd():
    import subprocess
    import uuid
    assert os.name == 'posix', '--systemd requires native Linux systemd --user'
    root = Path(tempfile.mkdtemp(prefix='chapter28-native-', dir=SCRATCH))
    model, store, performance = fixture(root)
    run_a = make_run(root, store, True)
    registry.register(store, 'v2', model, performance)
    registry.select(store, 'v2')
    run_b = make_run(root, store, True)
    blocked = make_run(root, store)
    before = hashes(root)
    unit = 'llm-lab-serving-test-' + uuid.uuid4().hex + '.service'
    state, port = root / 'owned state $ %', free_port()
    def transport_unit(target, fail=False):
        args = (sys.executable, '-B', Path(__file__).resolve(), '--fixture-server', *(('--fail-startup',) if fail else ()), target)
        command = ' '.join(deploy.quote(arg) for arg in args)
        return (HERE / 'llm-lab-serving.service').read_text().replace('@COMMAND@', command)
    evidence = {'scope': 'SYNTHETIC approval + stdlib HTTP fixture ONLY; REAL Ubuntu systemd --user lifecycle; NOT model inference or learner deployment', 'root': str(root), 'unit': unit, 'port': port}
    try:
        with patch.object(deploy, 'render', side_effect=transport_unit):
            assert invoke(blocked, store, root / 'blocked state', unit=unit, port=port) == 2
            assert not (root / 'blocked state').exists()
            assert invoke(run_a, store, state, unit=unit, port=port, timeout=20) == 0
            verify = subprocess.run(['systemd-analyze', '--user', 'verify', str(state / unit)], capture_output=True, text=True)
            evidence['unit_verify'] = dict(exit_code=verify.returncode, stdout=verify.stdout, stderr=verify.stderr)
            assert verify.returncode == 0, verify.stderr
            assert invoke(run_b, store, state, unit=unit, port=port, timeout=20) == 0
            evidence['B'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 2)
            assert deploy.rollback(state, 20) == 0
            evidence['rollback_A'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 2)
            original = (state / 'active.json').read_bytes()
            (state / 'fail-versions.json').write_text('["v2"]')
            assert invoke(run_b, store, state, unit=unit, port=port, timeout=20) == 1
            assert (state / 'active.json').read_bytes() == original
            evidence['recovered_A'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 2)
            assert evidence['B']['health']['model_version']['version'] == 'v2'
            assert evidence['rollback_A']['health']['model_version']['version'] == 'v1'
            assert evidence['recovered_A']['health']['model_version']['version'] == 'v1'
            assert all(hashes(root)[name] == digest for name, digest in before.items())
            evidence['input_files_unchanged'] = True
            # A real native unit can be active while loading; a different same-release
            # listener must not make that unbound instance look ready.
            deploy.control('stop', unit)
            unrelated = TransportService(state)
            try:
                import time
                unrelated('restart', unit)
                release = registry.read_json(state / 'active.json')['current']
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        deploy.check_http(release, 1)
                        break
                    except OSError:
                        time.sleep(.02)
                else:
                    raise AssertionError('unrelated native-test transport did not start')
                (state / 'test-startup-delay.json').write_text('5')
                native_control, observed = deploy.control, []
                def observe_control(*args):
                    value = native_control(*args)
                    if args[0] == 'show' and '--property=ActiveState' in args:
                        observed.append(value)
                    return value
                old_results = set(state.glob('*-result.json'))
                with patch.object(deploy, 'control', side_effect=observe_control):
                    assert invoke(run_a, store, state, unit=unit, port=port, timeout=20) == 1
                assert 'active' in observed
                assert (state / 'active.json').read_bytes() == original
                assert unrelated.process is not None and unrelated.process.poll() is None
                new_results = set(state.glob('*-result.json')) - old_results
                assert len(new_results) == 1
                failure = registry.read_json(new_results.pop())
                assert 'new deployment instance' in failure['error'] and failure['recovery'] == 'failed'
                evidence['same_release_unrelated_listener_rejected'] = dict(
                    native_active_observed=True, unrelated_listener_preserved=True, result=failure)
            finally:
                unrelated('stop', unit)
                (state / 'test-startup-delay.json').unlink(missing_ok=True)
            evidence['results'] = [registry.read_json(p) for p in sorted(state.glob('*-result.json'))]
    finally:
        evidence['cleanup'] = clean_native(unit, state)
        (root / 'native-evidence.json').write_text(json.dumps(evidence, indent=2))
    initial_unit = 'llm-lab-serving-test-' + uuid.uuid4().hex + '.service'
    initial = root / 'initial failure'
    try:
        with patch.object(deploy, 'render', side_effect=lambda target: transport_unit(target, True)):
            assert invoke(run_b, store, initial, unit=initial_unit, port=free_port(), timeout=20) == 1
            assert not (initial / 'active.json').exists()
            evidence['initial_failure'] = registry.read_json(next(initial.glob('*-result.json')))
            assert evidence['initial_failure']['recovery'] == 'unavailable_no_previous'
    finally:
        evidence['initial_cleanup'] = clean_native(initial_unit, initial)
        (root / 'native-evidence.json').write_text(json.dumps(evidence, indent=2))
    print('PASS REAL native systemd A->B->rollback A->failed B recovery A + initial failure/no rollback; owned units inactive/not-found; SYNTHETIC transport only')
    print('EVIDENCE', root / 'native-evidence.json')


def test_real_nf4():
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # Trusted chapter26 construction only: zero-B untrained adapter, NO training/evaluation.
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'api' for t in n.targets))
    end = next(i for i, n in enumerate(tree.body[start:], start) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'registry' and n.value.func.attr == 'select')
    namespace = dict(globals(), **locals(), script=previous.with_name('01-track-run.py'))
    with patch.dict(os.environ, {'TMPDIR': str(SCRATCH), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}):
        exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(previous), 'exec'), namespace)
    root, store, model_dir, performance, base = (namespace[k] for k in ('root', 'store', 'model_dir', 'performance', 'base'))
    run_a = make_run(root, store, True)
    registry.register(store, 'zero-fixture-v2', model_dir, performance)
    registry.select(store, 'zero-fixture-v2')
    run_b = make_run(root, store, True)
    before = dict(base=hashes(base), model=hashes(model_dir), registry=hashes(store), runs=hashes(root / 'runs'), sources=deploy.sources())
    state, port = root / 'nf4 state', free_port()
    service = TransportService(state, real_model=True)
    evidence = dict(scope='REAL CPU NF4 production serve + HTTP inference; SYNTHETIC approvals; TEST subprocess lifecycle adapter NOT native systemd; zero-B untrained fixture NOT learner approved deployment', root=str(root), port=port)
    try:
        with patch.object(deploy, 'control', side_effect=service):
            assert invoke(run_a, store, state, port=port, timeout=120) == 0
            assert invoke(run_b, store, state, port=port, timeout=120) == 0
            evidence['B'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 5)
            assert deploy.rollback(state, 120) == 0
            evidence['rollback_A'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 5)
            service.fail.add('zero-fixture-v2')
            active_bytes = (state / 'active.json').read_bytes()
            assert invoke(run_b, store, state, port=port, timeout=120) == 1
            assert (state / 'active.json').read_bytes() == active_bytes
            evidence['recovered_A'] = deploy.check_http(registry.read_json(state / 'active.json')['current'], 5)
            for key in ('B', 'rollback_A', 'recovered_A'):
                assert evidence[key]['health']['quantized_modules'] == 210
                assert evidence[key]['generate']['generated_tokens'] == 1
            assert evidence['B']['health']['model_version']['version'] == 'zero-fixture-v2'
            assert evidence['rollback_A']['health']['model_version']['version'] == evidence['recovered_A']['health']['model_version']['version'] == 'zero-fixture-v1'
            assert before == dict(base=hashes(base), model=hashes(model_dir), registry=hashes(store), runs=hashes(root / 'runs'), sources=deploy.sources())
            evidence['inputs_unchanged'] = True
            evidence['results'] = [registry.read_json(p) for p in sorted(state.glob('*-result.json'))]
    finally:
        service('stop', 'llm-lab-serving.service')
        evidence['owned_process_stopped'] = service.process is None
        (root / 'deployment-nf4-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('PASS REAL CPU NF4 210 modules + actual production serve/HTTP A->B->rollback A->failed B recovery A; synthetic approvals/test adapter only')
    print('EVIDENCE', root / 'deployment-nf4-evidence.json')


if __name__ == '__main__':
    if '--fixture-server' in sys.argv:
        fixture_server(sys.argv[-1])
    elif '--systemd' in sys.argv:
        test_native_systemd()
    elif '--integration' in sys.argv:
        test_real_nf4()
    else:
        test_scratch_location()
        test_render_keeps_virtual_environment()
        test_flat_vm_state()
        test_blocked_no_changes()
        test_approved_http()
        test_failed_candidate_recovery()
        test_explicit_rollback()
        test_rollback_from_corrupt_current()
        test_serve_rechecks_before_bind()
        test_unrelated_same_release_listener()
        test_control_diagnostic()
        test_cli()
        test_invalid_inputs_and_ownership()
        test_fresh_source_import()
