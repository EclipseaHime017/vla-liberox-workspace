"""R21 allowlisted final-artifact catalog. Independent operations process; no training writes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tempfile
import time

import yaml

TRAIN_FILES = ('metrics.jsonl', 'effective_config.yaml', 'summary.json', 'provenance.json', 'pipeline.json', 'progress.json')
POLICY_FILES = ('actor.pt', 'pi05_identity.json', 'policy.yaml')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def safe(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink path rejected')
    return path


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', value):
        raise ValueError('Unsafe identity token')
    return value


def signature(path):
    st = safe(path).stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError('Expected regular file')
    return {'size': st.st_size, 'inode': st.st_ino, 'mtime_ns': st.st_mtime_ns, 'device': st.st_dev}


def file_sha(path):
    before = signature(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    if signature(path) != before:
        raise ValueError('Source changed during hashing')
    return h.hexdigest()


def immutable(path, value):
    raw = canonical(value)
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Immutable publication identity conflict')
        return
    fd, name = tempfile.mkstemp(prefix='.r21-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_from_job(contract, expected, job):
    if job['status'] != 'COMPLETED':
        return None
    if job['id'] != expected['job_id'] or job['request_sha256'] != expected['request_sha256']:
        raise ValueError('Queue job identity mismatch')
    run = safe(expected['run_directory'])
    if safe(job['run_directory']) != run:
        raise ValueError('Run directory mismatch')
    summary = json.loads(safe(run / 'summary.json').read_text())
    result = job['result']
    if any(x.get('status') != 'COMPLETED' or x.get('step') != expected['training_steps'] for x in (summary, result)):
        raise ValueError('Final step/status mismatch')
    policy = safe(result['policy_overlay'])
    registry = safe(expected['policy_registry_root'])
    if policy != safe(summary['policy_overlay']) or policy.name != 'policy.yaml' or policy.parent.parent != registry:
        raise ValueError('Final policy outside allowlisted registry or summary mismatch')
    raw = yaml.safe_load(policy.read_text())
    pid = token(raw['policy_id'])
    parent = raw['parent']
    if parent['contract'] != contract['parent_contract'] or parent.get('sha256') != sha(canonical({k: v for k, v in parent.items() if k != 'sha256'})):
        raise ValueError('Frozen parent contract/hash mismatch')
    if (policy.parent.name != pid or raw['training_step'] != expected['training_steps']
            or raw['algorithm'] != expected['algorithm'] or raw['family'] != 'pi05'
            or parent['id'] != contract['parent_id'] or parent['revision'] != contract['parent_revision']
            or parent['contract']['observation'] != {'image_flip_horizontal': False, 'image_flip_vertical': True}
            or raw['actor'] != 'actor.pt' or raw['base_identity'] != 'pi05_identity.json'):
        raise ValueError('Final policy identity mismatch')
    training = yaml.safe_load(safe(run / 'effective_config.yaml').read_text())
    if training['training']['seed'] != expected['training_seed']:
        raise ValueError('Training seed mismatch')
    files = []
    for path, relative in [(policy.parent / n, n) for n in POLICY_FILES] + [(run / n, 'training_records/' + n) for n in TRAIN_FILES]:
        path = safe(path)
        before = signature(path)
        digest = file_sha(path)
        if signature(path) != before:
            raise ValueError('Source changed during publication')
        files.append({'index': len(files), 'policy_id': pid, 'source': str(path), 'relative': relative,
                      'size': before['size'], 'sha256': digest, 'stat': before})
    components = {e['relative']: e['sha256'] for e in files}
    if raw['component_sha256'] != {'actor': components['actor.pt'], 'base_identity': components['pi05_identity.json']}:
        raise ValueError('Policy component SHA mismatch')
    # Final metadata bytes must be exactly the metadata we validated, not a later rewrite.
    if yaml.safe_load(policy.read_text()) != raw or json.loads((run / 'summary.json').read_text()) != summary:
        raise ValueError('Final metadata changed')
    for e in files:
        if signature(Path(e['source'])) != e['stat']:
            raise ValueError('Source changed before READY')
    content = sha(canonical([{k: e[k] for k in ('relative', 'size', 'sha256')} for e in files[:3]]))
    model = {k: expected[k] for k in ('job_id', 'request_sha256', 'task_id', 'max_steps', 'training_seed', 'algorithm', 'training_steps', 'priority')}
    model.update(policy_id=pid, artifact_content_sha256=content, file_indices=list(range(len(files))),
                 training_config_relative='training_records/effective_config.yaml')
    return {'schema_version': 1, 'kind': 'r21_final_model', 'source_id': contract['source_id'],
            'models': [model], 'files': files, 'terminal': {'job': job, 'summary': summary}}


def ready_manifest(root, pid):
    folder = safe(root / 'models' / token(pid))
    receipt = json.loads(safe(folder / 'READY.json').read_text())
    raw = safe(folder / 'manifest.json').read_bytes()
    if receipt['manifest_sha256'] != sha(raw):
        raise ValueError('READY manifest SHA mismatch')
    manifest = json.loads(raw)
    if manifest['models'][0]['policy_id'] != pid:
        raise ValueError('READY policy identity mismatch')
    return receipt, manifest


def scan(contract):
    root = safe(contract['catalog_root'])
    root.mkdir(parents=True, exist_ok=True)
    with safe(root / 'scan.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = json.loads(safe(contract['queue_path']).read_text())
        jobs = {j['id']: j for j in queue['jobs']}
        outcome = {'ready': [], 'waiting': [], 'errors': {}}
        for expected in contract['jobs']:
            try:
                job = jobs[expected['job_id']]
                if job['status'] != 'COMPLETED':
                    outcome['waiting'].append(expected['job_id'])
                    continue
                pid = token(Path(job['result']['policy_overlay']).parent.name)
                marker = safe(root / 'models' / pid / 'READY.json')
                if marker.exists():
                    receipt, manifest = ready_manifest(root, pid)
                    stored = manifest['models'][0]
                    if (job['id'] != expected['job_id'] or job['request_sha256'] != expected['request_sha256']
                            or stored['job_id'] != expected['job_id'] or stored['request_sha256'] != expected['request_sha256']
                            or safe(job['run_directory']) != safe(expected['run_directory'])
                            or job['result'] != manifest['terminal']['job']['result']
                            or json.loads(safe(Path(expected['run_directory']) / 'summary.json').read_text()) != manifest['terminal']['summary']):
                        raise ValueError('Existing READY request conflict')
                    for entry in manifest['files']:
                        if signature(Path(entry['source'])) != entry['stat']:
                            raise ValueError('Existing READY source changed')
                    outcome['ready'].append(receipt)
                    continue
                manifest = model_from_job(contract, expected, job)
                folder = root / 'models' / pid
                immutable(folder / 'manifest.json', manifest)
                model = manifest['models'][0]
                receipt = {k: model[k] for k in ('policy_id', 'artifact_content_sha256', 'job_id', 'priority')}
                receipt['manifest_sha256'] = sha(canonical(manifest))
                immutable(folder / 'READY.json', receipt)
                outcome['ready'].append(receipt)
            except (OSError, ValueError, KeyError, TypeError) as error:
                outcome['errors'][expected['job_id']] = str(error)
        return outcome


def serve(contract, args, output):
    if time.time() >= contract['expires_at']:
        raise ValueError('Read capability expired')
    root = safe(contract['catalog_root'])
    root.mkdir(parents=True, exist_ok=True)
    with safe(root / 'readers.lock').open('a+') as lock:
        fcntl.flock(lock, (fcntl.LOCK_EX | fcntl.LOCK_NB) if args == ['done'] else fcntl.LOCK_SH)
        if args == ['done']:
            if time.time() < contract['discovery_end']:
                raise ValueError('Cannot revoke catalog capability before discovery window closes')
            authorization = contract['revoke']
            auth = safe(authorization['authorized_keys'])
            with auth.open('r+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                lines = stream.readlines()
                matching = [line for line in lines if line.rstrip('\n') == authorization['line']]
                if len(matching) > 1:
                    raise ValueError('Duplicate scoped grant')
                stream.seek(0)
                stream.writelines(line for line in lines if line.rstrip('\n') != authorization['line'])
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            output.write(canonical({'revoked': True, 'removed': len(matching)}))
            return
        _serve(contract, args, output)


def _serve(contract, args, output):
    root = safe(contract['catalog_root'])
    allowed = {j['job_id']: j for j in contract['jobs']}
    if args == ['catalog']:
        models = []
        for marker in sorted((root / 'models').glob('*/READY.json')):
            receipt, manifest = ready_manifest(root, marker.parent.name)
            model = manifest['models'][0]
            if model['job_id'] not in allowed or model['request_sha256'] != allowed[model['job_id']]['request_sha256']:
                raise ValueError('Catalog model outside allowlist')
            models.append(receipt)
        output.write(canonical({'schema_version': 1, 'source_id': contract['source_id'], 'models': models}))
        return
    if len(args) not in (2, 4) or args[0] not in ('manifest', 'get', 'prefix'):
        raise ValueError('Unsupported read-only operation')
    _, manifest = ready_manifest(root, args[1])
    model = manifest['models'][0]
    if model['job_id'] not in allowed or model['request_sha256'] != allowed[model['job_id']]['request_sha256']:
        raise ValueError('Model outside allowlist')
    if args[0] == 'manifest' and len(args) == 2:
        output.write(canonical(manifest))
        return
    if len(args) != 4 or not all(x.isdigit() for x in args[2:]):
        raise ValueError('Invalid file index/offset')
    index, offset = map(int, args[2:])
    if index >= len(manifest['files']):
        raise ValueError('File index outside manifest')
    entry = manifest['files'][index]
    expected = allowed[model['job_id']]
    relative = entry['relative']
    permitted = ({n: safe(expected['policy_registry_root']) / model['policy_id'] / n for n in POLICY_FILES}
                 | {'training_records/' + n: safe(expected['run_directory']) / n for n in TRAIN_FILES})
    path = safe(entry['source'])
    if relative not in permitted or path != permitted[relative] or offset > entry['size'] or signature(path) != entry['stat']:
        raise ValueError('Frozen file identity mismatch')
    with path.open('rb') as stream:
        if args[0] == 'prefix':
            h = hashlib.sha256()
            remaining = offset
            while remaining:
                block = stream.read(min(1024**2, remaining))
                if not block:
                    raise ValueError('Truncated source')
                h.update(block)
                remaining -= len(block)
            data = canonical({'bytes': offset, 'sha256': h.hexdigest()})
        else:
            stream.seek(offset)
            remaining = entry['size'] - offset
            while remaining:
                block = stream.read(min(1024**2, remaining))
                if not block:
                    raise ValueError('Truncated source')
                output.write(block)
                remaining -= len(block)
            data = b''
    if signature(path) != entry['stat']:
        raise ValueError('Frozen source changed while reading')
    output.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--request', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raw = safe(args.contract).read_bytes()
    if sha(raw) != args.contract_sha256:
        raise ValueError('Contract SHA mismatch')
    contract = json.loads(raw)
    if args.scan:
        print(json.dumps(scan(contract)))
    else:
        serve(contract, args.request if args.request is not None else shlex.split(os.environ.get('SSH_ORIGINAL_COMMAND', '')), sys.stdout.buffer)


if __name__ == '__main__':
    main()
