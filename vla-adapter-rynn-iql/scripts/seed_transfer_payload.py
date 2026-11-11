"""Seed an indexed receiver payload from explicitly selected complete local exports.

No network, dataset import, source mutation, partial cleanup, or upload protocol changes.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile


def checked(path, *, file=False):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'Symlink rejected: {parent}')
    if file and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'Not a regular file: {path}')
    return path


def relative(value):
    if not isinstance(value, str):
        raise ValueError('Relative path must be a string')
    p = PurePosixPath(value)
    if not value or not p.parts or p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return value


def identity(entry):
    size, sha = entry['size'], entry['sha256']
    if type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Invalid size/SHA256')
    return size, sha


def digest(path):
    checked(path, file=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def matches(path, key):
    checked(path, file=True)
    return path.stat().st_size == key[0] and digest(path) == key[1]


def copy_new(source, destination, key):
    checked(source, file=True)
    checked(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked(destination.parent)
    fd, name = tempfile.mkstemp(prefix='.r17-copy-', dir=destination.parent)
    staging = Path(name)
    try:
        with os.fdopen(fd, 'wb') as out, source.open('rb') as inp:
            shutil.copyfileobj(inp, out, length=8 * 1024**2)
            out.flush()
            os.fsync(out.fileno())
        if not matches(staging, key):
            raise ValueError(f'Copied content changed: {source}')
        if (source.stat().st_dev, source.stat().st_ino) == (staging.stat().st_dev, staging.stat().st_ino):
            raise ValueError('Copy must not share source inode')
        checked(destination)
        os.link(staging, destination)  # Atomic no-replace; link only our independent copy, never source.
        if not matches(destination, key):
            raise ValueError(f'Published file verification failed: {destination}')
    finally:
        staging.unlink(missing_ok=True)  # Only this invocation's freshly created temporary file.


def seed(receiver_root, source_roots, *, apply=False, sender_paused=False):
    if apply and not sender_paused:
        raise ValueError('Pause sender and verify no pending put before --apply --sender-paused')
    root = checked(receiver_root)
    raw = checked(root / 'manifest.json', file=True).read_bytes()
    settings = json.loads(checked(root / 'settings.json', file=True).read_text())
    if hashlib.sha256(raw).hexdigest() != settings['manifest_sha256']:
        raise ValueError('Receiver manifest does not match pinned settings SHA')
    manifest = json.loads(raw)
    entries = manifest['files']
    if not entries or any(e['index'] != i for i, e in enumerate(entries)):
        raise ValueError('Receiver indices must be sequential')
    paths = [relative(e['relative']) for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate receiver path')
    targets = {identity(e) for e in entries}
    checked(root / 'payload')
    candidates = defaultdict(list)
    sources = []
    for value in source_roots:
        source = checked(value)
        if source == root or source == root / 'payload' or root in source.parents or source in root.parents:
            raise ValueError('Source and receiver directories must not overlap')
        export_path = checked(source / 'export.json', file=True)
        original = export_path.read_bytes()
        exported = json.loads(original)
        if exported.get('complete') is not True or exported.get('kind') != 'dataset_runs_export':
            raise ValueError(f'Not a complete formal export: {export_path}')
        if not isinstance(exported['files'], dict):
            raise ValueError('Expected formal export files mapping')
        source_manifest = dict(exported['files'])
        source_manifest['export.json'] = {'size': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
        sources.append({'root': str(source), 'export_sha256': hashlib.sha256(original).hexdigest()})
        for name, entry in source_manifest.items():
            name, key = relative(name), identity(entry)
            if key in targets:
                candidates[key].append(source / name)
    report = {'mode': 'apply' if apply else 'dry-run', 'receiver_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'source_manifests': sources, 'existing': [], 'reusable': [], 'copied': [], 'missing': [],
              'rejected_candidates': [], 'receiver_sha_and_import_verified': False}
    lock_path = checked(root / 'operation.lock', file=True)
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = {}
        for entry in entries:
            key = identity(entry)
            dest = checked(root / 'payload' / entry['relative'])
            item = {'index': entry['index'], 'relative': entry['relative'], 'size': key[0], 'sha256': key[1]}
            if dest.exists():
                if not matches(dest, key):
                    raise ValueError(f'Existing destination mismatch; will not overwrite: {dest}')
                report['existing'].append(item)
                continue
            if key not in verified:
                verified[key] = None
                for candidate in candidates.get(key, []):
                    try:
                        valid = matches(candidate, key)
                    except (OSError, ValueError) as error:
                        report['rejected_candidates'].append({'path': str(candidate), 'reason': str(error)})
                        continue
                    if valid:
                        verified[key] = candidate
                        break
                    report['rejected_candidates'].append({'path': str(candidate), 'reason': 'size/SHA mismatch'})
            source = verified[key]
            if source is None:
                report['missing'].append(item)
                continue
            item['source'] = str(source)
            report['reusable'].append(item)
            if apply:
                copy_new(source, dest, key)
                report['copied'].append(item)
        # A source manifest mutation invalidates this operation's receipt.
        for source in sources:
            if digest(Path(source['root']) / 'export.json') != source['export_sha256']:
                raise ValueError('Source manifest changed during operation')
        if (root / 'manifest.json').read_bytes() != raw:
            raise ValueError('Receiver manifest changed during operation')
    report['totals'] = {name: {'files': len(report[name]), 'bytes': sum(e['size'] for e in report[name])}
                        for name in ('existing', 'reusable', 'copied', 'missing')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receiver-root', type=Path, required=True)
    parser.add_argument('--source-export', type=Path, action='append', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--sender-paused', action='store_true')
    args = parser.parse_args()
    print(json.dumps(seed(args.receiver_root, args.source_export, apply=args.apply,
                          sender_paused=args.sender_paused), indent=2))


if __name__ == '__main__':
    main()
