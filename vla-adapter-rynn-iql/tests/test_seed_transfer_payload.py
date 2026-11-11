import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('seed_payload', Path(__file__).parents[1] / 'scripts/seed_transfer_payload.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def item(data):
    return {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class SeedPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='r17-seed-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src, self.dst = self.root / 'old', self.root / 'new'
        self.src.mkdir()
        self.dst.mkdir()
        (self.dst / 'payload').mkdir()
        (self.dst / 'parts').mkdir()
        (self.dst / 'operation.lock').touch()
        self.data = b'original observations'
        (self.src / 'old-name').write_bytes(self.data)
        self.export = {'kind': 'dataset_runs_export', 'complete': True, 'files': {'old-name': item(self.data)}}
        self.manifest = {'files': [{'index': 0, 'relative': 'different/new-name', **item(self.data)},
                                   {'index': 1, 'relative': 'rewards/new.bin', **item(b'new reward')}]}
        self.save()

    def save(self):
        (self.src / 'export.json').write_text(json.dumps(self.export))
        raw = json.dumps(self.manifest).encode()
        (self.dst / 'manifest.json').write_bytes(raw)
        (self.dst / 'settings.json').write_text(json.dumps({'manifest_sha256': hashlib.sha256(raw).hexdigest()}))

    def run_seed(self, apply=False):
        return MODULE.seed(self.dst, [self.src], apply=apply, sender_paused=apply)

    def test_dry_run_readonly_then_copy_by_hash_not_name(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = self.run_seed()
        self.assertEqual(report['totals']['reusable']['files'], 1)
        self.assertEqual(report['totals']['missing']['files'], 1)
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        partial = self.dst / 'parts/00000.part'
        partial.write_bytes(b'partial')
        report = self.run_seed(True)
        dest = self.dst / 'payload/different/new-name'
        self.assertEqual(report['totals']['copied']['files'], 1)
        self.assertEqual(dest.read_bytes(), self.data)
        self.assertNotEqual(dest.stat().st_ino, (self.src / 'old-name').stat().st_ino)
        self.assertEqual(partial.read_bytes(), b'partial')
        self.assertEqual(self.run_seed(True)['totals']['existing']['files'], 1)
        self.assertEqual((self.src / 'old-name').read_bytes(), self.data)

    def test_same_size_changed_content_not_reused(self):
        (self.src / 'old-name').write_bytes(b'X' * len(self.data))
        report = self.run_seed(True)
        self.assertEqual(report['totals']['copied']['files'], 0)
        self.assertEqual(len(report['rejected_candidates']), 1)

    def test_bad_existing_file_not_overwritten(self):
        dest = self.dst / 'payload/different/new-name'
        dest.parent.mkdir()
        dest.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(ValueError, 'will not overwrite'):
            self.run_seed(True)
        self.assertEqual(dest.read_bytes(), b'X' * len(self.data))

    def test_symlink_source_and_destination_rejected(self):
        original = self.src / 'old-name'
        original.unlink()
        outside = self.root / 'outside'
        outside.write_bytes(self.data)
        original.symlink_to(outside)
        report = self.run_seed(True)
        self.assertEqual(report['totals']['copied']['files'], 0)
        self.assertIn('Symlink', report['rejected_candidates'][0]['reason'])
        (self.dst / 'payload/different').symlink_to(self.src, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_seed(True)

    def test_pinned_manifest_traversal_duplicate_and_pause_guards(self):
        with self.assertRaisesRegex(ValueError, 'Pause sender'):
            MODULE.seed(self.dst, [self.src], apply=True)
        (self.dst / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'pinned'):
            self.run_seed()
        self.manifest['files'][0]['relative'] = '../outside'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.run_seed(True)
        self.manifest['files'][0]['relative'] = 'rewards/new.bin'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_seed(True)

    def test_copy_after_source_changes_fails_without_publishing(self):
        real_copy = MODULE.shutil.copyfileobj
        def corrupt(inp, out, **kwargs):
            real_copy(inp, out, **kwargs)
            out.write(b'corrupt')
        with patch.object(MODULE.shutil, 'copyfileobj', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'Copied content changed'):
                self.run_seed(True)
        self.assertFalse((self.dst / 'payload/different/new-name').exists())
        self.assertFalse(list(self.dst.rglob('.r17-copy-*')))
        self.assertEqual((self.src / 'old-name').read_bytes(), self.data)

    def test_racing_destination_never_overwritten(self):
        real_link = os.link
        def race(src, dst):
            Path(dst).write_bytes(b'competitor')
            real_link(src, dst)
        with patch.object(MODULE.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError):
                self.run_seed(True)
        self.assertEqual((self.dst / 'payload/different/new-name').read_bytes(), b'competitor')
        self.assertFalse(list(self.dst.rglob('.r17-copy-*')))

    def test_partial_unchanged_even_when_source_absent(self):
        partial = self.dst / 'parts/00000.part'
        partial.write_bytes(b'keep me')
        (self.src / 'old-name').unlink()
        report = self.run_seed(True)
        self.assertEqual(report['totals']['missing']['files'], 2)
        self.assertEqual(partial.read_bytes(), b'keep me')

    def test_active_receiver_lock_and_incomplete_source_block_copy(self):
        with (self.dst / 'operation.lock').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaises(BlockingIOError):
                self.run_seed(True)
        self.assertFalse((self.dst / 'payload/different').exists())
        self.export['complete'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'complete formal export'):
            self.run_seed(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
