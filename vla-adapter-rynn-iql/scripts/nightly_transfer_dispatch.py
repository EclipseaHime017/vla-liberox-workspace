"""R21 one-window discovery and serial verified download/publication/evaluation.

Reuses the deployed network_reference transfer and memory-only password broker.
No backend or training-process changes. Source/allowlist/config are locally pinned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tempfile
import urllib.parse
import urllib.request

import yaml
from nightly_transfer_source import canonical, sha, safe, token, file_sha, immutable, POLICY_FILES, TRAIN_FILES
from nightly_transfer_windows import parse_windows, active_window

SCHEDULE = '7a82d7067d2620e8c02d1f2afbaac7822b91db801de36cb2cb0c4ee9e28a8ba1'
TERMINAL = ('EVALUATION_QUEUED', 'REJECTED', 'EVALUATION_AMBIGUOUS', 'BLOCKED_RETRY_BUDGET', 'BLOCKED_AUTHENTICATION')


class IdentityError(ValueError):
    pass


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('Window timestamps require explicit timezone')
    return parsed.timestamp()


def atomic(path, value):
    path = safe(path)
    tmp = path.with_name(path.name + '.tmp')
    safe(tmp)
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def module(path, digest, name):
    if file_sha(safe(path)) != digest:
        raise IdentityError('Pinned module SHA mismatch')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def allowlisted_jobs(raw):
    if raw.get('kind') == 'r21_nscc_pbs_static_allowlist':
        return [{**e, 'request_sha256': raw['request_canonical_sha256'],
                 'max_steps': e['evaluation_max_steps'], 'run_directory': e['run']}
                for e in raw['entries']]
    return raw['jobs']


def validate_manifest(manifest, reference, allowed, source_id):
    if sha(canonical(manifest)) != reference['manifest_sha256'] or manifest.get('source_id') != source_id:
        raise IdentityError('Frozen manifest SHA/source mismatch')
    if manifest.get('kind') != 'r21_final_model' or len(manifest.get('models', [])) != 1:
        raise IdentityError('Expected one final model')
    model = manifest['models'][0]
    token(model['policy_id'])
    expected = allowed.get(model['job_id'])
    if expected is None:
        raise IdentityError('Unallowlisted training job')
    fields = ('job_id', 'request_sha256', 'task_id', 'max_steps', 'training_seed', 'algorithm', 'training_steps', 'priority')
    if any(model.get(k) != expected[k] for k in fields):
        raise IdentityError('Model differs from frozen job contract')
    if any(model[k] != reference[k] for k in ('policy_id', 'artifact_content_sha256', 'job_id', 'priority')):
        raise IdentityError('Catalog identity conflict')
    entries = manifest['files']
    required = set(POLICY_FILES) | {'training_records/' + n for n in TRAIN_FILES}
    if len(entries) != 9 or {e['relative'] for e in entries} != required:
        raise IdentityError('Incomplete or excessive artifact paths')
    for i, e in enumerate(entries):
        if (e['index'] != i or e['policy_id'] != model['policy_id'] or type(e['size']) is not int or e['size'] < 0
                or len(e['sha256']) != 64 or any(c not in '0123456789abcdef' for c in e['sha256'])):
            raise IdentityError('Malformed file identity')
    policy_entries = [e for e in entries if e['relative'] in POLICY_FILES]
    if [e['relative'] for e in policy_entries] != list(POLICY_FILES):
        raise IdentityError('Unexpected canonical policy file order')
    content = sha(canonical([{k: e[k] for k in ('relative', 'size', 'sha256')} for e in policy_entries]))
    if content != model['artifact_content_sha256']:
        raise IdentityError('Artifact content identity mismatch')
    terminal = manifest['terminal']
    if source_id == 'nscc':
        if (terminal.get('kind') != 'pbs' or terminal.get('pbs_job_id') != expected['pbs_job_id']
                or terminal['pbs'].get('job_state') != 'F' or terminal['pbs'].get('Exit_status') != 0
                or terminal.get('request_sha256') != expected['request_sha256']
                or terminal.get('run') != expected['run_directory']
                or terminal['summary'].get('status') != 'COMPLETED'
                or terminal['summary'].get('step') != model['training_steps']):
            raise IdentityError('PBS final evidence conflicts with frozen run')
        config_entry = next(e for e in entries if e['relative'] == 'training_records/effective_config.yaml')
        if config_entry['sha256'] != expected['effective_config_sha256']:
            raise IdentityError('PBS effective configuration SHA mismatch')
    elif (terminal['job']['id'] != model['job_id'] or terminal['job']['status'] != 'COMPLETED'
            or terminal['job']['request_sha256'] != model['request_sha256']
            or any(x.get('status') != 'COMPLETED' or x.get('step') != model['training_steps']
                   for x in (terminal['summary'], terminal['job']['result']))):
        raise IdentityError('Source has not completed final training')
    return model


class Scheduler:
    def __init__(self, config):
        self.config = config
        self.root = safe(config['root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.windows = parse_windows(config)
        self.start, self.end = self.windows[0][0], self.windows[-1][1]
        self.lock = threading.RLock()
        self.state_path = self.root / 'scheduler-state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'config_sha256': sha(canonical(config)), 'models': {}, 'source_errors': {}, 'last_scan': None}
        if self.state['config_sha256'] != sha(canonical(config)):
            raise IdentityError('Existing scheduler belongs to another frozen config')
        self.allowed = {}
        for name, source in config['sources'].items():
            raw = safe(source['allowlist']).read_bytes()
            if sha(raw) != source['allowlist_sha256']:
                raise IdentityError('Allowlist SHA mismatch')
            self.allowed[name] = {j['job_id']: j for j in allowlisted_jobs(json.loads(raw))}
        self.save()

    def save(self):
        with self.lock:
            atomic(self.state_path, self.state)

    def scan_due(self, now):
        window = active_window(self.windows,now)
        return window is not None and (self.state['last_scan'] is None or now-self.state['last_scan'] >= 1800
                or (now >= window[1]-60 and self.state['last_scan'] < window[1]-60))

    def discover(self, now, remote_json, *, smoke=False):
        if not smoke and not self.scan_due(now):
            return
        for source in self.config['sources']:
            try:
                catalog = remote_json(source, ['catalog'])
                if catalog.get('schema_version') != 1 or catalog.get('source_id') != source:
                    raise IdentityError('Invalid source catalog')
                for ref in catalog['models']:
                    if ref['job_id'] not in self.allowed[source]:
                        raise IdentityError('Catalog contains unknown job')
                    manifest = remote_json(source, ['manifest', token(ref['policy_id'])])
                    model = validate_manifest(manifest, ref, self.allowed[source], source)
                    pid = model['policy_id']
                    old = self.state['models'].get(pid)
                    if old and (old['artifact_content_sha256'] != model['artifact_content_sha256'] or old['manifest_sha256'] != ref['manifest_sha256']):
                        raise IdentityError('Same policy ID has different frozen content')
                    immutable(self.root / 'models' / pid / 'manifest.json', manifest)
                    if old is None:
                        with self.lock:
                            self.state['models'][pid] = {**model, 'manifest_sha256': ref['manifest_sha256'],
                                'source_id': source, 'phase': 'PENDING', 'discovered_at': now, 'retry_at': 0}
                with self.lock:
                    self.state['source_errors'].pop(source, None)
            except Exception as error:
                with self.lock:
                    self.state['source_errors'][source] = str(error)
        if not smoke:
            self.state['last_scan'] = now
        self.save()

    def next_model(self, now):
        if now < self.start:
            return None
        self.expire_retries(now)
        candidates = [m for m in self.state['models'].values() if m['phase'] not in TERMINAL and m.get('retry_at', 0) <= now]
        if active_window(self.windows,now) is None:
            candidates = [m for m in candidates if active_window(self.windows,m.get('transfer_started_at',-1))]
        return min(candidates, key=lambda m: (m['priority'], m['discovered_at'], m['policy_id'])) if candidates else None

    def expire_retries(self, now):
        with self.lock:
            for model in self.state['models'].values():
                deadline = min(model.get('started_at', now) + 86400,
                               self.config['sources'][model['source_id']].get('expires_at', float('inf')))
                if now >= deadline and model['phase'] not in TERMINAL:
                    model.update(phase='BLOCKED_RETRY_BUDGET', error='24h retry/source capability deadline reached; partial retained')
            self.save()

    def active_retry_pending(self, now):
        """Do not exit just because an in-window transfer's retry is 60s away."""
        with self.lock:
            self.expire_retries(now)
            return any(active_window(self.windows,m.get('transfer_started_at',-1)) is not None
                and m['phase'] not in TERMINAL
                for m in self.state['models'].values())

    def close_report(self):
        with self.lock:
            self.state['window_closed'] = True
            self.state['unprocessed'] = [{'policy_id': m['policy_id'], 'priority': m['priority'],
                'phase': m['phase'], 'reason': m.get('error') or ('WINDOW_CLOSED_NOT_STARTED' if 'transfer_started_at' not in m else 'INCOMPLETE')}
                for m in self.state['models'].values() if m['phase'] != 'EVALUATION_QUEUED']
            self.state['status'] = 'NEEDS_FOLLOWUP' if self.state['unprocessed'] or self.state['source_errors'] else 'ALL_DISCOVERED_MODELS_QUEUED'
            self.save()

    def open_window(self,now):
        window = active_window(self.windows,now)
        if window is None or self.state.get('current_window_start') == window[0]: return
        with self.lock:
            if self.state.get('window_closed'):
                self.state.setdefault('close_report_history',[]).append({
                    'previous_window_start':self.state.get('current_window_start'),
                    'status':self.state.get('status'), 'unprocessed':self.state.get('unprocessed',[]),
                    'closed_windows':list(self.state.get('closed_windows',[]))})
            self.state.update(current_window_start=window[0],current_window_end=window[1],
                              window_closed=False,unprocessed=[],status='WINDOW_OPEN')
            self.save()


def api(config, path, data=None):
    req = urllib.request.Request(config['api_url'] + path, data=None if data is None else canonical(data),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def evaluation(config, model, root, call):
    request = {'task_id': model['task_id'], 'policy_id': model['policy_id'], 'trials': 100,
               'max_steps': model['max_steps'], 'open_loop_steps': 10, 'realtime': False,
               'init_state_indices': list(range(10)), 'base_seed': 0, 'seed_count': 10, 'schedule_seed': 0}
    preview = call('/api/evaluations/preview', request)
    if preview['schedule_sha256'] != SCHEDULE:
        raise IdentityError('Evaluation schedule changed')
    request.update(schedule_sha256=SCHEDULE, policy_content_sha256=preview['policy_content_sha256'])
    immutable(root / 'evaluation-request.json', request)
    history = call('/api/evaluations?' + urllib.parse.urlencode({'policy_id': model['policy_id']}))
    queue = call('/api/evaluations/queue')
    identifiers = {e['id'] for e in history}
    identifiers.update(j['parameters']['evaluation_id'] for j in queue['jobs']
                       if j.get('parameters', {}).get('policy_id') == model['policy_id']
                       and j.get('parameters', {}).get('evaluation_id'))
    for identifier in sorted(identifiers):
        detail = call('/api/evaluations/' + urllib.parse.quote(identifier, safe=''))
        policy = detail.get('policy_snapshot') or detail.get('policy') or {}
        task = detail.get('task_snapshot') or detail.get('task') or {}
        if policy.get('policy_id') != model['policy_id']:
            continue
        if policy.get('content_sha256') != request['policy_content_sha256']:
            raise IdentityError('Existing evaluation has different policy content')
        if task.get('task_id') == model['task_id'] and detail.get('schedule_sha256') == SCHEDULE and detail.get('config') == preview['config']:
            if detail['status'] in ('FAILED', 'CANCELED', 'CANCELLED'):
                raise IdentityError('Existing matching evaluation failed; manual review required')
            return {'request': request, 'evaluation_id': identifier, 'reconciled': True}
        raise IdentityError('Existing policy evaluation uses a different configuration')
    # Persist intent before POST. If response is lost, only reconciliation is allowed.
    if (root / 'evaluation-intent.json').exists():
        raise RuntimeError('EVALUATION_AMBIGUOUS: prior POST intent has no confirmed matching job')
    immutable(root / 'evaluation-intent.json', request)
    queued = call('/api/evaluations/queue', request)
    return {'request': request, 'queued': queued, 'reconciled': False}


def process_model(scheduler, model, network_env):
    config, pid = scheduler.config, model['policy_id']
    root = safe(scheduler.root / 'models' / pid)
    manifest = json.loads((root / 'manifest.json').read_text())
    source = config['sources'][model['source_id']]
    def mark(phase, **kw):
        with scheduler.lock:
            model.update(phase=phase, **kw)
            scheduler.save()
    def network_command(command):
        pieces = shlex.split(command)
        if pieces[0] == 'get' and 'transfer_started_at' not in model:
            now = time.time()
            if active_window(scheduler.windows,now) is None:
                raise RuntimeError('WINDOW_CLOSED: refusing a new model transfer')
            mark('DOWNLOADING', transfer_started_at=now)
        remote = [pieces[0], pid, *pieces[1:]]
        return ['ssh', '-F', source['ssh_config'], source['ssh_alias'],
                source.get('remote_prefix', '') + ' ' + shlex.join(remote)]
    try:
        with safe(root / 'worker.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N = module(config['network_reference'], config['network_reference_sha256'], 'r21_network_worker')
            N.ROOT = root
            N.ssh = network_command
            N.SSH_ENV = network_env if model['source_id'] == 'nscc' else os.environ.copy()
            def target(entry):
                return safe(root / 'files' / entry['relative'])
            N.target = target
            N.total_downloaded = lambda m: sum(target(e).stat().st_size if target(e).exists() else (target(e).with_name(target(e).name+'.part').stat().st_size if target(e).with_name(target(e).name+'.part').exists() else 0) for e in m['files'])
            N.save = lambda **kw: atomic(root / 'download-progress.json', kw)
            verified = root / 'VERIFIED.json'
            mark('DOWNLOADING')
            if not verified.exists():
                for i in sorted(range(len(manifest['files'])), key=lambda i: manifest['files'][i]['relative'] == 'actor.pt'):
                    N.transfer(i, manifest['files'][i], manifest)
                mark('DOWNLOADING', authentication_failures=0)
            if verified.exists() and json.loads(verified.read_text()) != {'manifest_sha256': model['manifest_sha256'], 'artifact_content_sha256': model['artifact_content_sha256']}:
                raise IdentityError('Verified receipt conflicts')
            for e in manifest['files']:
                if target(e).stat().st_size != e['size'] or file_sha(target(e)) != e['sha256']:
                    raise IdentityError('Downloaded artifact SHA mismatch')
            immutable(verified, {'manifest_sha256': model['manifest_sha256'], 'artifact_content_sha256': model['artifact_content_sha256']})
            mark('VERIFIED')
            publish(config, model, root, manifest)
            mark('PUBLISHED')
            receipt = root / 'evaluation.json'
            if not receipt.exists():
                result = evaluation(config, model, root, lambda path, data=None: api(config, path, data))
                immutable(receipt, result)
            mark('EVALUATION_QUEUED', error=None)
    except Exception as error:
        phase = 'REJECTED' if isinstance(error, (IdentityError, ValueError, N.InvalidArtifact if 'N' in locals() else ValueError)) else 'ERROR'
        if 'EVALUATION_AMBIGUOUS' in str(error):
            phase = 'EVALUATION_AMBIGUOUS'
        if 'N' in locals() and isinstance(error, N.AuthenticationFailure):
            model['authentication_failures'] = model.get('authentication_failures', 0) + 1
            if model['authentication_failures'] >= 6:
                phase = 'BLOCKED_AUTHENTICATION'
        mark(phase, error=str(error), retry_at=time.time()+60)


def publish(config, model, root, manifest):
    source = root / 'files'
    raw = yaml.safe_load((source / 'policy.yaml').read_text())
    for k, expected in (('policy_id', model['policy_id']), ('algorithm', model['algorithm']), ('training_step', model['training_steps'])):
        if raw.get(k) != expected:
            raise IdentityError('Downloaded policy differs from allowlist')
    if raw['parent']['id'] != config['parent_id'] or raw['parent']['revision'] != config['parent_revision'] or raw['parent']['contract'] != config['parent_contract']:
        raise IdentityError('Downloaded parent identity mismatch')
    if (source / 'pi05_identity.json').read_bytes() != safe(Path(config['base_checkpoint']) / 'pi05_identity.json').read_bytes():
        raise IdentityError('Local base identity mismatch')
    if yaml.safe_load((source / 'training_records/effective_config.yaml').read_text())['training']['seed'] != model['training_seed']:
        raise IdentityError('Downloaded training seed mismatch')
    module_root = safe(config['export_module_root'])
    if file_sha(module_root / 'vla_rynn_iql/model_artifacts.py') != config['export_module_sha256']:
        raise IdentityError('Publication module changed')
    sys.path.insert(0, str(module_root))
    from vla_rynn_iql.model_artifacts import export_policy, validate_policy, decode_child
    decode_child(raw)
    dest = safe(Path(config['policy_registry']) / model['policy_id'])
    if not dest.exists():
        export_policy(source / 'policy.yaml', dest, config['base_checkpoint'])
    deployed = yaml.safe_load(safe(dest / 'policy.yaml').read_text())
    validate_policy(dest / 'policy.yaml', deployed)
    compare_deployed_identity(raw, deployed)
    for e in manifest['files']:
        if e['relative'].startswith('training_records/'):
            target = safe(dest / e['relative'])
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                copy_new(source / e['relative'], target, e)
            if target.stat().st_size != e['size'] or file_sha(target) != e['sha256']:
                raise IdentityError('Existing training record differs')
    immutable(root / 'publication.json', {'policy_id': model['policy_id'], 'artifact_content_sha256': model['artifact_content_sha256'],
                                         'deployed_policy_sha256': file_sha(dest / 'policy.yaml')})


def compare_deployed_identity(raw, deployed):
    # export_policy rewrites machine-local checkpoint paths, not model semantics.
    for key in ('policy_id', 'family', 'algorithm', 'training_step', 'component_sha256',
                'action_horizon', 'action_dim', 'proprio_dim', 'stats_key',
                'schema_version', 'dataset_sha256', 'reward_sha256'):
        if deployed.get(key) != raw.get(key):
            raise IdentityError('Existing deployed policy differs: ' + key)
    for key in ('id', 'family', 'revision', 'source', 'source_revision', 'contract', 'stats_key'):
        if deployed['parent'].get(key) != raw['parent'].get(key):
            raise IdentityError('Existing deployed parent differs: ' + key)
    # Parent sha256 is path-dependent and is checked by validate_policy/decode_child.
    source_config = {k: v for k, v in raw['model_config'].items() if k != 'base_checkpoint'}
    deployed_config = {k: v for k, v in deployed['model_config'].items() if k != 'base_checkpoint'}
    if source_config != deployed_config:
        raise IdentityError('Existing deployed model configuration differs')


def copy_new(source, target, entry):
    """Never leave a half-written immutable final record on process interruption."""
    fd, name = tempfile.mkstemp(prefix='.r21-record-', dir=target.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as out, source.open('rb') as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        if tmp.stat().st_size != entry['size'] or file_sha(tmp) != entry['sha256']:
            raise IdentityError('Copied training record failed SHA')
        try:
            os.link(tmp, target)  # new temporary inode, NOT a hardlink to the source
        except FileExistsError:
            if target.stat().st_size != entry['size'] or file_sha(target) != entry['sha256']:
                raise IdentityError('Concurrent final training record differs')
    finally:
        tmp.unlink(missing_ok=True)


def run(config, password_fd=None, *, smoke=False):
    root = safe(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    # Lock before Scheduler reads/writes persistent state, including metadata smoke.
    with safe(root / 'scheduler.lock').open('a+') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = Scheduler(config)
        env = os.environ.copy()
        if 'nscc' in config['sources']:
            if password_fd is None or password_fd < 3:
                scheduler.state['status'] = 'BLOCKED_CREDENTIAL'
                scheduler.save()
                raise RuntimeError('BLOCKED_CREDENTIAL: memory-only NSCC password pipe required')
            N = module(config['network_reference'], config['network_reference_sha256'], 'r21_broker')
            N.ROOT = safe(config['credential_tools_root'])
            saved = sys.argv
            try:
                sys.argv = ['broker', '--password-fd', str(password_fd)]
                password = N.password_from_pipe()
            finally:
                sys.argv = saved
            broker = stack.enter_context(N.CredentialBroker(password))
            password[:] = b'\0' * len(password)
            env = broker.environment()
        def remote_json(source_id, args):
            source = config['sources'][source_id]
            remote = source.get('remote_prefix', '') + ' ' + shlex.join(args)
            p = subprocess.run(['ssh', '-F', source['ssh_config'], source['ssh_alias'], remote],
                capture_output=True, stdin=subprocess.DEVNULL, timeout=180, env=env if source_id == 'nscc' else os.environ.copy())
            if p.returncode:
                raise RuntimeError('Source metadata unavailable: ' + p.stderr.decode(errors='replace')[-1000:])
            return json.loads(p.stdout)
        if smoke:
            scheduler.discover(time.time(), remote_json, smoke=True)
            return scheduler.state
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        active = None
        while True:
            now = time.time()
            if scheduler.scan_due(now):
                scheduler.discover(now, remote_json)
            closed = scheduler.state.setdefault('closed_windows',[])
            ended = [end for _,end in scheduler.windows if now >= end and end not in closed]
            if ended:
                # Metadata only: records final arrivals without starting them outside the window.
                scheduler.discover(now, remote_json, smoke=True)
                closed.extend(ended)
                scheduler.close_report()
            scheduler.open_window(now)
            scheduler.state['next_window_start'] = next((start for start,_ in scheduler.windows if start > now),None)
            if now < scheduler.end and active is None:
                scheduler.state['status'] = 'WINDOW_OPEN' if active_window(scheduler.windows,now) else 'WAITING_NEXT_WINDOW'
            scheduler.save()
            if active is not None and active.done():
                active.result()
                active = None
            if active is None:
                model = scheduler.next_model(now)
                if model:
                    model.setdefault('started_at', now)
                    scheduler.save()
                    active = executor.submit(process_model, scheduler, model, env)
            if now >= scheduler.end and active is None and not scheduler.active_retry_pending(now):
                scheduler.close_report()
                return scheduler.state
            time.sleep(5)


def migrate_config(old_path, new_path, old_digest, new_digest):
    """Locked audited schedule-only migration. Model trees and receipts are untouched."""
    old_bytes, new_bytes = safe(old_path).read_bytes(),safe(new_path).read_bytes()
    if sha(old_bytes) != old_digest or sha(new_bytes) != new_digest:
        raise IdentityError('Reviewed old/new configuration byte SHA mismatch')
    old, new = json.loads(old_bytes),json.loads(new_bytes)
    previous, upcoming = parse_windows(old),parse_windows(new)
    if 'windows' not in new: raise IdentityError('Migration requires explicit finite R22 nights')
    schedule_keys = {'window_start','window_end','windows'}
    if {k:v for k,v in old.items() if k not in schedule_keys} != {k:v for k,v in new.items() if k not in schedule_keys}:
        raise IdentityError('Migration may only change schedule, not sources/allowlist/identity')
    if any(not any(a <= start and end <= b for a,b in upcoming) for start,end in previous):
        raise IdentityError('Migration cannot drop an old authorized window')
    root = safe(old['root'])
    with safe(root/'scheduler.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path = safe(root/'scheduler-state.json')
        before_bytes = state_path.read_bytes()
        state = json.loads(before_bytes)
        key = sha(canonical({'old':old_digest,'new':new_digest}))
        record = root/'migrations'/key
        if state['config_sha256'] == sha(canonical(new)):
            proof = json.loads(safe(record/'intent.json').read_text())
            if proof['new_config_sha256'] != new_digest: raise IdentityError('Migration identity mismatch')
            if not (record/'APPLIED.json').exists():
                immutable(record/'APPLIED.json',{'state_sha256':sha(before_bytes),'migration':key})
            return {'migration':key,'already_applied':True}
        if state['config_sha256'] != sha(canonical(old)):
            raise IdentityError('Active state does not match reviewed old configuration')
        # Independently check current pinned allowlist bytes, without constructing/saving Scheduler.
        for source in new['sources'].values():
            if file_sha(safe(source['allowlist'])) != source['allowlist_sha256']:
                raise IdentityError('Pinned allowlist changed')
        immutable(record/'before-state.json',state)
        immutable(record/'intent.json',{'old_config_sha256':old_digest,'new_config_sha256':new_digest,
            'original_state_byte_sha256':sha(before_bytes),'old_config':old,'new_config':new})
        state['config_sha256'] = sha(canonical(new))
        state.setdefault('migration_history',[]).append(key)
        # All prior fields/model phases/POST intents are retained, including historical close reports.
        state['status'] = 'SCHEDULE_MIGRATED_PENDING_RESTART'
        atomic(state_path,state)
        immutable(record/'APPLIED.json',{'state_sha256':file_sha(state_path),'migration':key})
        return {'migration':key,'already_applied':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--password-fd', type=int)
    parser.add_argument('--metadata-smoke', action='store_true')
    parser.add_argument('--migrate-from',type=Path)
    parser.add_argument('--old-config-sha256')
    parser.add_argument('--new-config-sha256')
    args = parser.parse_args()
    if args.migrate_from:
        if not args.old_config_sha256 or not args.new_config_sha256 or args.metadata_smoke or args.password_fd is not None:
            parser.error('Migration requires both reviewed config SHA values and no runtime action')
        print(json.dumps(migrate_config(args.migrate_from,args.config,args.old_config_sha256,args.new_config_sha256)))
    else:
        print(json.dumps(run(json.loads(safe(args.config).read_text()), args.password_fd, smoke=args.metadata_smoke)))


if __name__ == '__main__':
    main()
