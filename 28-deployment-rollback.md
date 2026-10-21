# 28. 승인 근거로 배포하고 실패하면 이전 버전 복구하기

[← 이전: 평가 파이프라인](27-evaluation-pipeline.md) · [목차](README.md) · [다음: 운영 관찰·부하 검사 →](29-operations-load.md)

## 1. 목표와 경계

LLMOps 여섯 묶음 중 **4/6 배포·롤백**입니다.27장의 명시한 실행 폴더를 읽고 승인 근거가 유효한 경우에만 전용 로컬 서비스를 전환합니다. 시작 후HTTP 식별 정보를 확인하고, 실패하면 이전에 검증한 배포로 복구합니다.

새 평가·재학습·등록·모델 선택 변경은 하지 않습니다. 기존 모델·레지스트리·보고서·실행 기록을 보존합니다. 프로세스 시작·종료는 Ubuntu 기본 **systemd 사용자 서비스**가 맡으며 별도 데몬·프록시·배포 플랫폼을 만들지 않습니다. RAG는 제외합니다.

> 현재 CUDA 예시의 모델은0/8·품질 미달입니다. 이 실행을 넣으면 **종료2, 서비스/상태 변경 없음**이 정상입니다. 합격을 만들기 위해 기준을 낮추거나 승인 파일을 고치지 않습니다. 실제 승인 배포·학습 모델 롤백은 아직 미확인입니다.

## 2. 선택·배포·롤백의 차이

| 항목 | 의미 |
|---|---|
| selected.json |23장 레지스트리의 다음 후보 선택. 현재 서비스가 무엇을 로딩했는지와 다름 |
| 지정한27장 실행 폴더 | pipeline/run/quality 결과·지문·모델 식별 정보로 연결한 승인 또는 차단 근거 |
| active.json | 이 실행기가 마지막으로 HTTP 점검까지 성공해 확정한 배포와 이전 배포 |
| launch.json | 전용 unit이 다음 시작 때 읽는 명시적 모델 버전·설정 |

이 장은`registry.verify(store, version)`으로 **명시적 버전**을 검증합니다. 현재selected 포인터를 바꾸거나 따라가지 않으므로, 롤백할A와 지금 선택된 후보B가 달라도 A를 검증할 수 있습니다.

active.json은 마지막 성공 기록이지 현재 생존 보장이 아닙니다. 수동 정지·복구 실패 뒤에는 unit 상태와 결과 기록을 따로 확인해야 합니다. 운영 관찰은 다음 묶음에서 다룹니다.

## 3. 전환 절차와 중단 조건

1. **근거 확인:** 명시한 실행의 스키마·점수·판정·run/quality/소스 지문·등록 모델을 검증합니다. 전체 과거 이력이나 현재selected 포인터는 검사 대상으로 강제하지 않습니다.
   - 정상 품질 미달이면 종료2. unit 조회·제어·상태 폴더 생성 전 중단합니다.
   - 손상·불일치면 종료1. 승인으로 취급하지 않습니다.
2. **승인된 전환:** 전용 unit/상태 소유권을 확인하고 새 attempt 기록을 만든 뒤 명시한 버전으로 재시작합니다.
   - 기본은loopback `127.0.0.1:8001`입니다. 이전 실습의8000 리스너나 다른 unit을 종료·채택하지 않습니다.
   - 최초 배포의8001이 이미 사용 중이면 거부합니다. 기존 단위 설정이 이 실행기의 것과 다르면 덮어쓰지 않습니다.
3. **HTTP 점검과 확정:** `/health`와 `/generate`의 **전체model_version**뿐 아니라 **이번 시작의deployment_instance**가 모두 일치하는지 확인합니다. 한 토큰 실제 생성도 확인합니다.
   - HTTP200·같은 모델 버전·unit의active 상태만으로 합격시키지 않습니다. 시작마다 새 토큰을launch.json에 넣고 서비스가 두 응답에 돌려주는지 비교합니다. 다른 동일 버전 리스너가 로딩 중인 새 프로세스 대신 점검을 통과할 수 없습니다. 점검과 입력 재검증 뒤에만 active.json을 확정합니다.
4. **실패 복구:** 이전에 검증된 배포가 있으면 그 버전으로 재시작하고 HTTP 식별 정보를 확인합니다.
   - 복구가 성공해도 후보 배포 실패는 종료1로 유지합니다.
   - 최초 배포 실패라면 복구 대상이 없음을 기록하고 전용 unit을 정지합니다. 가짜 롤백 성공을 만들지 않습니다.

인스턴스 토큰은 이번 프로세스의 준비 상태를 연결하는 값이지 비밀번호·서명·접근 제어가 아닙니다. 로컬 파일을 악의적으로 함께 고치는 공격자를 막는 기능으로 해석하지 않습니다.

이 절차는 **중단이 있는 순차 재시작**입니다. 무중단 전환·동시 작성자·재부팅 지속·강제 종료 후 자동 트랜잭션 복구를 보장하지 않습니다. 코드/모델/환경을 백업하지 않으며 변경되거나 사라진 이전 아티팩트를 복원하지 못합니다.

## 4. 실행 위치·입력·상태

Ubuntu VM의 `~/llm-lab/source`, 기존 `.venv`에서 실행합니다.27장까지의 여섯 의존 파일과smollm2_pipeline.py를 그대로 사용합니다. 새 코드는smollm2_deploy.py와unit 템플릿llm-lab-serving.service 두 파일입니다.

기본 상태는 `../serving_deployment_v1`입니다. 소스·입력 실행·모델·레지스트리·기존 성능 보고서와 겹치는 경로는 거부합니다. 모든 변경은 이 실행기가 소유한 상태와 전용 사용자 unit으로 제한합니다.

```text
serving_deployment_v1/
  owner.json                 소유한 코드·설정·unit 내용
  llm-lab-serving.service     실제 명령으로 렌더링한 unit
  launch.json                다음 시작 입력·새 인스턴스 토큰
  active.json                마지막 성공 배포와 이전 배포
  <새 ID>-attempt.json       시도 전 입력 기록
  <같은 ID>-result.json      성공/실패·HTTP 결과·복구 결과
```

품질 미달이면 이 상태 폴더를 새로 만들지 않습니다. 승인 경로에서는`systemctl --user link`, `daemon-reload`, `restart`/`stop`을 사용합니다. **sudo·기존 unit 변경·enable·linger 설정은 하지 않습니다.** 사용자 서비스 관리자가 동작하는 로그인 세션을 전제로 하며, 로그아웃/재부팅 뒤 지속 서빙은 이번 범위가 아닙니다.

## 5. 전체 코드·unit 생성과 실행

정본: [01-deploy-rollback.py](source/28-deployment-rollback/01-deploy-rollback.py), [llm-lab-serving.service](source/28-deployment-rollback/llm-lab-serving.service). 새 파일은 exclusive 모드로 생성합니다. 두 이름 중 하나라도 이미 있으면 멈추고 기존 파일을 보존합니다.

아래 실행 중에는 **27장 출력의 실제run_id**를 입력합니다. 이미 차단된 실행은 다시 평가하지 않고 근거를 읽어 종료2로 거부합니다.

```bash
cd ~/llm-lab/source &&
source .venv/bin/activate &&
python - <<'CREATE' &&
from pathlib import Path
files = {
    'smollm2_deploy.py': r'''"""Bounded single-writer deployment. systemd --user owns the process; no reboot/crash recovery.
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
''',
    'llm-lab-serving.service': r'''[Unit]
Description=LLM lab bounded serving (single local user)

[Service]
Type=exec
ExecStart=:@COMMAND@
Restart=no
TimeoutStopSec=15
'''
}
if any(Path(name).exists() or Path(name).is_symlink() for name in files):
    raise FileExistsError('새 코드/unit 파일이 이미 있습니다. 덮어쓰지 않습니다.')
for name, body in files.items():
    with Path(name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)
    print('CREATED', name, Path(name).stat().st_size, 'bytes')
CREATE
{
read -r -p '27장 출력의 run_id: ' RUN_ID &&
python smollm2_deploy.py deploy \
  --run "../llmops_runs_v1/$RUN_ID" \
  --registry ../model_registry_v1 \
  --state ../serving_deployment_v1 \
  --device cuda --port 8001 --timeout 120
result=$?
printf 'DEPLOY_EXIT_CODE=%s\n' "$result"
}
```

## 6. 코드 해설·결과 확인

- 초기화: 기존27장 실행기와26장 검증 함수를 재사용합니다. 캐시 경로 격리로 낡은bytecode 대신 소스를 읽고 설정을 복원합니다.
- `validate`: 지정 실행의 문항·집계·파일 지문·판정·명시적 모델 버전·소스를 연결합니다. 단순한approved 값이나 HTTP200만 믿지 않습니다.
- `safe_state`: 저장소 배치와 VM의 평평한 파일 배치를 구분하고 실제 소스 영역을 보호합니다. VM의source/ 옆 상태 폴더를 잘못 막지 않도록 승인 경로도 검사합니다.
- `render`/unit: Python 실행 파일·스크립트·상태 경로를 systemd 인자로 인용합니다. 가상환경의Python 심볼릭 링크를 원본 시스템Python으로 풀지 않아 기존 패키지 환경을 유지합니다. 환경변수 자동 치환을 끄고 공백·달러·퍼센트 등을 구분하며 제어문자는 거부합니다. 셸을 거치지 않습니다.
- `serve`: 기존 원본/revision 제한과NF4 로더·HTTP Handler를 재사용합니다. 로딩 뒤 재검증하고loopback에bind하며 시작 identity를 고정합니다.
- `transition`: 새attempt/result 파일은 덮어쓰지 않습니다. launch/active 쓰기는 임시 파일과replace를 사용하지만 여러 파일 전체의 원자적 트랜잭션은 아닙니다.

| 종료 | 의미 | 상태 변경 |
|---:|---|---|
| 2 | 근거가 정상인 품질 미달 | 서비스 제어·새 상태 쓰기 없음 |
| 0 | 승인된 전환 또는 명시적 롤백의 HTTP 점검 완료 | 마지막 성공 배포 확정 |
| 1 | 근거/설정/서비스/HTTP/기록 오류 또는 후보 실패 | 결과 기록과recovery 필드 확인. 복구 성공도 후보 실패를 성공으로 바꾸지 않음 |

일반CLI는 종료 코드로 결과를 전달합니다. 승인 경로의 세부 진단은 새`*-result.json`의error/recovery/recovery_error·HTTP 응답과 해당 unit의로그를 확인합니다. 전환 전 오류는stderr에 출력합니다. 파일 쓰기 실패나 강제 종료라면 일부 기록만 있을 수 있습니다.

### 별도 CUDA VM의 배포 거부 실행 예

27장에서 품질 미달로 판정한 기존 실행을 지정한 실제 출력입니다. 새 평가나 모델 로딩 없이 보관 근거를 검사하고 배포를 거부했습니다.

```text
CREATED smollm2_deploy.py 17054 bytes
CREATED llm-lab-serving.service 134 bytes
DEPLOY_EXIT_CODE=2
DEPLOY_STATE_EXISTS=False
```

`DEPLOY_STATE_EXISTS`는 실행 뒤 `Path('../serving_deployment_v1').exists()`를 출력한 값입니다. 종료2는 품질 미달에 따른 정상 거부이고, False는 새 배포 상태 폴더가 없음을 뜻합니다. 다운로드된 `.service`는 템플릿이며 등록된 unit이 아닙니다. 승인 배포·롤백·새 CUDA 추론의 실행 증거가 아니며 기존 수동 서버의 종료도 뜻하지 않습니다. 이 차단 경로를 완료한 뒤에는 품질 기준 완화나 재학습 없이 다음 운영 관찰 묶음으로 넘어갈 수 있습니다.

승인 전환 뒤 확인 명령입니다. 품질 미달로 상태가 생성되지 않았다면 아래active.json 읽기를 할 필요가 없습니다.

```bash
systemctl --user status llm-lab-serving.service --no-pager
journalctl --user -u llm-lab-serving.service -n 30 --no-pager
python - <<'STATE'
import json
from pathlib import Path
root = Path('../serving_deployment_v1')
for name in ('active.json',):
    print(name, json.dumps(json.loads((root / name).read_text(encoding='utf-8')), ensure_ascii=False, indent=2))
STATE
```

이전 성공 배포가 실제로 존재할 때의 명시적 롤백 명령입니다. **현재0/8 차단만 확인한 경우에는 실행하지 않습니다.** 이전 배포가 없으면 종료1이며 임의의 과거 수동 서버를 복구 대상으로 채택하지 않습니다.

```bash
python smollm2_deploy.py rollback --state ../serving_deployment_v1 --timeout 120
result=$?
printf 'ROLLBACK_EXIT_CODE=%s\n' "$result"
```

## 7. 실제 검증 범위와 미확인 부분

| 근거 | 범위 |
|---|---|
| 로컬 표준 라이브러리·합성 승인 | 품질 차단 무변경·파일/모델/소스 변조 거부·소유권/경로·상태/HTTP 식별 정보·동일 버전의 다른 리스너 거부·오류/복구·평평한 VM 승인 경로·테스트 임시 폴더 이식성 검사 |
| 실제 Ubuntu systemd 사용자 unit | 고유 테스트 unit의A→B·명시적A롤백·실패B의A복구·최초 실패/복구 대상 없음·active이나 아직bind하지 않은 서비스 대신 다른 동일 버전 리스너를 성공으로 받아들이지 않음·정리 후inactive/not-found. HTTP 앱과승인은 **합성 transport fixture**이며 모델 추론이 아님 |
| 실제 CPU NF4 HTTP | 실제 운영용 로더/Handler·210개 양자화 모듈·한 토큰 생성·A→B→A롤백·실패B에서A복구. **승인은 합성**, 수명관리는 테스트용 subprocess adapter이며 native systemd와 결합한 ML 실행이 아님 |
| 별도 CUDA VM의 품질 차단 | 기존27장0/8 실행의 배포 종료2·DEPLOY_STATE_EXISTS=False·셸 복귀 확인. 새 모델 로딩·CUDA 추론·서비스 등록 없음 |
| 별도 VM의 학습 모델 승인 배포·롤백 | 아직 미실행. 품질 차단 확인과 구분 |

두 통합 검사의 coverage를 합쳐 **학습 모델의 실제 승인 배포**로 해석하지 않습니다. 일반 실행에품질 우회 옵션이나 합성 서버는 없습니다. 기준 변경·재학습·가짜 승인 파일 생성도 하지 않습니다.

검증 스크립트는 [test_deployment.py](source/28-deployment-rollback/test_deployment.py)입니다. 기본 검사는 실제loopback 테스트 프로세스를 시작하고 종료하지만 systemd/모델 추론은 사용하지 않습니다. `--systemd`는 Linux의사용자 관리자에서 고유 테스트 unit만 만들고 정리합니다. `--integration`은 기존 분리된 ML 검증 환경에서 실제NF4를 사용합니다. 테스트승인과학습 모델품질을 분리합니다.

```bash
python source/28-deployment-rollback/test_deployment.py
python source/28-deployment-rollback/test_deployment.py --systemd
python source/28-deployment-rollback/test_deployment.py --integration
python source/check_guide.py
```

파일 지문은서명·권한 통제·아티팩트 백업이 아닙니다. 코드·템플릿·과거 모델·근거가 변경되면 복구도 거부할 수 있습니다. 이 실행기는 새 승인으로 현재 파일을 덮거나 손상된 과거 파일을 자동 복원하지 않습니다. 복구 실패 시unit을 정지하고 오류를 남기므로active.json만 보고 서비스가 살아 있다고 판단하지 않습니다.

## 8. 복습과 다음 묶음

- [ ] 선택 포인터와 실제 배포 버전이 다른 이유를 설명한다.
- [ ] 품질 차단 시 서비스 제어 전에 멈추는 경계를 설명한다.
- [ ] 전체model_version 식별 정보 비교와HTTP200의 차이를 설명한다.
- [ ] 후보 실패·복구 성공·전체 종료1이 함께 나올 수 있음을 설명한다.
- [ ] 최초 실패에서 복구 대상 없음과 이전 버전 복구를 구분한다.
- [ ] 검증한transport/systemd/모델 추론과 실제 학습 모델 승인 배포를 구분한다.

다음 묶음은 **운영 관찰·부하 검사**입니다. 이번 장은 단일 작성자·단일 전용loopback 서비스의 순차 전환이며, 운영 관찰·동시 부하·Docker 패키징은 후속 범위입니다.

[← 이전: 평가 파이프라인](27-evaluation-pipeline.md) · [목차](README.md) · [다음: 운영 관찰·부하 검사 →](29-operations-load.md)
