import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import yaml

SPEC = importlib.util.spec_from_file_location('night_source', Path(__file__).parents[1] / 'scripts/nightly_transfer_source.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='r21-source-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.registry = self.root / 'registry'
        self.policy = self.registry / 'pi05-test'
        self.run.mkdir()
        self.policy.mkdir(parents=True)
        self.actor = b'final frozen actor'
        (self.policy / 'actor.pt').write_bytes(self.actor)
        (self.policy / 'pi05_identity.json').write_bytes(b'identity')
        self.raw = {'policy_id': 'pi05-test', 'training_step': 10000, 'algorithm': 'bc', 'family': 'pi05',
            'actor': 'actor.pt', 'base_identity': 'pi05_identity.json',
            'parent': {'id': 'base', 'revision': 'a'*64, 'contract': {'observation': {'image_flip_horizontal': False, 'image_flip_vertical': True}}},
            'component_sha256': {'actor': M.sha(self.actor), 'base_identity': M.sha(b'identity')}}
        self.raw['parent']['sha256'] = M.sha(M.canonical(self.raw['parent']))
        (self.policy / 'policy.yaml').write_text(yaml.safe_dump(self.raw))
        self.result = {'status': 'COMPLETED', 'step': 10000, 'policy_overlay': str(self.policy / 'policy.yaml')}
        for name in M.TRAIN_FILES:
            (self.run / name).write_text('{}')
        (self.run / 'summary.json').write_text(json.dumps(self.result))
        (self.run / 'effective_config.yaml').write_text('training:\n  seed: 7\n')
        self.job = {'id': 'q_test', 'status': 'COMPLETED', 'request_sha256': 'b'*64,
                    'run_directory': str(self.run), 'result': self.result}
        self.contract = {'source_id': 'sensecore', 'catalog_root': str(self.root / 'catalog'),
            'queue_path': str(self.root / 'queue.json'), 'parent_id': 'base', 'parent_revision': 'a'*64,
            'parent_contract': self.raw['parent']['contract'],
            'expires_at': time.time()+3600, 'discovery_end': 0,
            'jobs': [{'job_id': 'q_test', 'request_sha256': 'b'*64, 'run_directory': str(self.run),
                'policy_registry_root': str(self.registry), 'training_steps': 10000, 'training_seed': 7,
                'algorithm': 'bc', 'task_id': 'middle', 'max_steps': 1200, 'priority': 0}]}
        self.queue()

    def queue(self):
        Path(self.contract['queue_path']).write_text(json.dumps({'jobs': [self.job]}))

    def call(self, *args):
        out = io.BytesIO()
        M.serve(self.contract, list(args), out)
        return out.getvalue()

    def test_only_terminal_and_late_arrival_publishes_once(self):
        self.job['status'] = 'RUNNING'
        self.queue()
        self.assertEqual(M.scan(self.contract)['ready'], [])
        self.assertEqual(json.loads(self.call('catalog'))['models'], [])
        self.job['status'] = 'COMPLETED'
        self.queue()
        first = M.scan(self.contract)
        self.assertEqual(len(first['ready']), 1)
        with patch.object(M, 'file_sha', side_effect=AssertionError('No repeat full actor hash')):
            self.assertEqual(M.scan(self.contract), first)
        manifest = json.loads(self.call('manifest', 'pi05-test'))
        self.assertEqual(len(manifest['files']), 9)
        self.assertEqual(self.call('get', 'pi05-test', '0', '2'), self.actor[2:])
        self.assertEqual(json.loads(self.call('prefix', 'pi05-test', '0', '3'))['sha256'], M.sha(self.actor[:3]))

    def test_identity_mismatch_never_ready(self):
        self.job['request_sha256'] = 'c'*64
        self.queue()
        self.assertTrue(M.scan(self.contract)['errors'])
        self.assertEqual(json.loads(self.call('catalog'))['models'], [])

    def test_component_sha_wrong_never_ready(self):
        (self.policy / 'actor.pt').write_bytes(b'changed')
        self.assertTrue(M.scan(self.contract)['errors'])
        self.assertEqual(json.loads(self.call('catalog'))['models'], [])

    def test_existing_ready_checks_current_terminal_identity(self):
        M.scan(self.contract)
        self.job['request_sha256'] = 'c'*64
        self.queue()
        self.assertTrue(M.scan(self.contract)['errors'])

    def test_changed_parent_contract_rejected(self):
        self.raw['parent']['contract']['architecture'] = {'wrong': True}
        self.raw['parent']['sha256'] = M.sha(M.canonical({k:v for k,v in self.raw['parent'].items() if k != 'sha256'}))
        (self.policy / 'policy.yaml').write_text(yaml.safe_dump(self.raw))
        self.contract['parent_contract'] = {'observation': {'image_flip_horizontal': False, 'image_flip_vertical': True}}
        self.assertTrue(M.scan(self.contract)['errors'])

    def test_partial_manifest_is_not_catalog_ready_and_can_retry(self):
        actual = M.immutable
        def failure(path, value):
            if path.name == 'READY.json':
                raise OSError('simulated publication failure')
            actual(path, value)
        with patch.object(M, 'immutable', side_effect=failure):
            self.assertTrue(M.scan(self.contract)['errors'])
        self.assertEqual(json.loads(self.call('catalog'))['models'], [])
        self.assertEqual(len(M.scan(self.contract)['ready']), 1)

    def test_no_overwrite_and_unknown_policy_or_path_rejected(self):
        M.scan(self.contract)
        marker = Path(self.contract['catalog_root']) / 'models/pi05-test/READY.json'
        before = marker.read_bytes()
        with self.assertRaises(ValueError):
            M.immutable(marker, {'bad': True})
        self.assertEqual(marker.read_bytes(), before)
        for args in [('manifest', '../x'), ('get', 'pi05-test', '-1', '0'), ('get', 'pi05-test', '99', '0'), ('shell',)]:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.call(*args)

    def test_symlink_and_source_change_rejected(self):
        M.scan(self.contract)
        (self.policy / 'actor.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.call('get', 'pi05-test', '0', '0')
        (self.policy / 'actor.pt').unlink()
        (self.root / 'outside').write_bytes(self.actor)
        (self.policy / 'actor.pt').symlink_to(self.root / 'outside')
        with self.assertRaises(ValueError):
            self.call('get', 'pi05-test', '0', '0')

    def test_done_exact_grant_after_window_and_expiry(self):
        auth = self.root / 'authorized_keys'
        auth.write_text('other\nscoped grant\n')
        grant = self.root / 'grant-line.txt'
        grant.write_text('scoped grant\n')
        self.contract['revoke'] = {'authorized_keys': str(auth), 'line_file': str(grant)}
        self.contract['discovery_end'] = time.time()+30
        with self.assertRaises(ValueError):
            self.call('done')
        self.assertEqual(auth.read_text(), 'other\nscoped grant\n')
        self.contract['discovery_end'] = 0
        self.assertEqual(json.loads(self.call('done'))['removed'], 1)
        self.assertEqual(auth.read_text(), 'other\n')
        self.contract['expires_at'] = 0
        with self.assertRaises(ValueError):
            self.call('catalog')


if __name__ == '__main__':
    unittest.main(verbosity=2)
