"""Persistent, CPU-only scanner for an existing pinned R21 source tool.

Run under nohup; no systemd and no changes to the installed getter/contract.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cluster.transfer.windows import parse_windows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('Refusing symlink state')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def next_scan(now, last, final, night_closings=()):
    if last is None:
        return min(now, final)
    if last >= final:
        return None
    return min(last + 1800, final, *(end-300 for end in night_closings if end-300 > last))


def run(args, clock=time.time, sleep=time.sleep, invoke=subprocess.run):
    source, contract, root = args.source.absolute(), args.contract.absolute(), args.root.absolute()
    for path in (source, contract, root):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Refusing symlink path')
    if digest(source) != args.source_sha256 or digest(contract) != args.contract_sha256:
        raise ValueError('Pinned source/contract SHA mismatch')
    cfg = json.loads(contract.read_text())
    final = cfg['discovery_end'] - 300
    schedule_path = getattr(args,'schedule',None)
    closings = []
    schedule_sha = None
    if schedule_path:
        schedule_path = schedule_path.absolute()
        if any(p.is_symlink() for p in (schedule_path,*schedule_path.parents)):
            raise ValueError('Refusing symlink schedule')
        schedule_sha = digest(schedule_path)
        if schedule_sha != args.schedule_sha256: raise ValueError('Schedule SHA mismatch')
        schedule = json.loads(schedule_path.read_text())
        if set(schedule) != {'windows'}: raise ValueError('Source schedule only defines finite windows')
        windows = parse_windows(schedule)
        closings = [end for _,end in windows]
        if windows[-1][1] > min(cfg['discovery_end'],cfg['expires_at']):
            raise ValueError('Schedule exceeds pinned discovery end/capability')
        final = windows[-1][1]-300
    if clock() > final:
        raise ValueError('Final scan time already passed; use explicit one-shot source --scan')
    root.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': args.source_sha256, 'contract_sha256': args.contract_sha256,
                'source': str(source), 'contract': str(contract)}
    if schedule_path: identity.update(schedule=str(schedule_path),schedule_sha256=schedule_sha)
    state_path = root / 'scanner-state.json'
    with (root / 'scanner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {'identity': identity, 'last_scan': None}
        if state['identity'] != identity:
            raise ValueError('Scanner state belongs to another frozen source')
        state.update(pid=os.getpid(), status='RUNNING', final_scan=final)
        while True:
            due = next_scan(clock(), state['last_scan'], final,closings)
            state['next_scan'] = due
            write(state_path, state)
            if due is None:
                state['status'] = 'FINISHED_SCANNING_NOT_ALL_TRAINING_COMPLETED'
                write(state_path, state)
                return state
            if clock() < due:
                sleep(min(30, due - clock()))
                continue
            try:
                if digest(source) != args.source_sha256 or digest(contract) != args.contract_sha256:
                    raise ValueError('Pinned source/contract changed')
                if schedule_path and digest(schedule_path) != schedule_sha:
                    raise ValueError('Pinned schedule changed')
                result = invoke([sys.executable, str(source), '--contract', str(contract),
                                 '--contract-sha256', args.contract_sha256, '--scan'],
                                capture_output=True, text=True, timeout=3600)
                report = json.loads(result.stdout) if result.returncode == 0 else None
                state.update(last_scan=clock(), returncode=result.returncode, report=report,
                             error=result.stderr[-4000:] if result.returncode else None)
            except Exception as error:
                state.update(last_scan=clock(), error=str(error), report=None)
            # Append-only diagnostics survive parent shell exit and individual scan errors.
            with (root / 'scanner-events.jsonl').open('a') as stream:
                stream.write(json.dumps(state) + '\n')
                stream.flush()
                os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'contract', 'root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--schedule',type=Path)
    parser.add_argument('--schedule-sha256')
    print(json.dumps(run(parser.parse_args())))


if __name__ == '__main__':
    main()
