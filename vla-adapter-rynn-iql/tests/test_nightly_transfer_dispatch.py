import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import os
import subprocess
import shlex
import time
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import nightly_transfer_dispatch as D


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='r21-dispatch-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.allowed = [{'job_id': 'q_a', 'request_sha256': 'a'*64, 'task_id': 'middle', 'max_steps': 1200,
                         'training_seed': 7, 'algorithm': 'bc', 'training_steps': 10000, 'priority': 20},
                        {'job_id': 'q_d1', 'request_sha256': 'b'*64, 'task_id': 'middle', 'max_steps': 1200,
                         'training_seed': 7, 'algorithm': 'bc', 'training_steps': 10000, 'priority': 0}]
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'jobs': self.allowed}))
        self.config = {'root': str(self.root / 'ops'), 'window_start': '2026-11-13T00:00:00+08:00',
                       'window_end': '2026-11-13T06:00:00+08:00', 'sources': {'sensecore': {
                           'allowlist': str(allow), 'allowlist_sha256': D.sha(allow.read_bytes())}}}
        self.scheduler = D.Scheduler(self.config)
        self.s = self.scheduler.start
        self.catalog = []
        self.manifests = {}

    def add(self, i):
        expected = self.allowed[i]
        pid = 'policy-' + expected['job_id']
        paths = list(D.POLICY_FILES) + ['training_records/' + n for n in D.TRAIN_FILES]
        files = [{'index': n, 'policy_id': pid, 'relative': p, 'size': 3, 'sha256': 'c'*64} for n, p in enumerate(paths)]
        model = {**expected, 'policy_id': pid, 'file_indices': list(range(9))}
        model['artifact_content_sha256'] = D.sha(D.canonical([{k:e[k] for k in ('relative','size','sha256')} for e in files[:3]]))
        manifest = {'schema_version': 1, 'kind': 'r21_final_model', 'source_id': 'sensecore', 'models': [model], 'files': files,
                    'terminal': {'job': {'id': expected['job_id'], 'request_sha256': expected['request_sha256'],
                                         'status':'COMPLETED', 'result': {'status':'COMPLETED','step':10000}},
                                 'summary': {'status':'COMPLETED','step':10000}}}
        self.manifests[pid] = manifest
        ref = {k:model[k] for k in ('policy_id','artifact_content_sha256','job_id','priority')}
        ref['manifest_sha256'] = D.sha(D.canonical(manifest))
        self.catalog.append(ref)
        return pid

    def remote(self, source, args):
        if args == ['catalog']:
            return {'schema_version':1, 'source_id':source, 'models':self.catalog}
        return self.manifests[args[1]]

    def test_timezone_window_edges_and_smoke_does_not_start(self):
        self.assertEqual(self.s, D.timestamp('2026-11-12T16:00:00+00:00'))
        self.assertFalse(self.scheduler.scan_due(self.s-1))
        self.assertTrue(self.scheduler.scan_due(self.s))
        self.assertFalse(self.scheduler.scan_due(self.scheduler.end))
        self.add(0)
        self.scheduler.discover(self.s-100, self.remote, smoke=True)
        self.assertIsNone(self.scheduler.next_model(self.s-1))
        self.assertIsNotNone(self.scheduler.next_model(self.s))
        self.assertIsNone(self.scheduler.next_model(self.scheduler.end))
        with self.assertRaises(ValueError): D.timestamp('2026-11-13T00:00:00')

    def test_late_arrival_priority_and_persistent_duplicate(self):
        first = self.add(0)
        self.scheduler.discover(self.s, self.remote)
        self.scheduler.state['models'][first]['phase'] = 'EVALUATION_QUEUED'
        self.scheduler.save()
        late = self.add(1)
        self.scheduler = D.Scheduler(self.config)
        self.scheduler.discover(self.s+1800, self.remote)
        self.assertEqual(len(self.scheduler.state['models']), 2)
        self.assertEqual(self.scheduler.next_model(self.s+1800)['policy_id'], late)
        self.assertEqual(self.scheduler.state['models'][first]['phase'], 'EVALUATION_QUEUED')

    def test_same_id_content_change_rejected_without_manifest_overwrite(self):
        pid = self.add(0)
        self.scheduler.discover(self.s, self.remote)
        old = (self.scheduler.root/'models'/pid/'manifest.json').read_bytes()
        self.manifests[pid]['files'][0]['sha256'] = 'd'*64
        self.catalog[0]['manifest_sha256'] = D.sha(D.canonical(self.manifests[pid]))
        self.scheduler.discover(self.s+1800, self.remote)
        self.assertTrue(self.scheduler.state['source_errors'])
        self.assertEqual((self.scheduler.root/'models'/pid/'manifest.json').read_bytes(), old)

    def test_unknown_job_failed_source_and_retry(self):
        self.add(0)
        self.scheduler.discover(self.s, lambda *_: (_ for _ in ()).throw(OSError('offline')))
        self.assertEqual(self.scheduler.state['models'], {})
        self.scheduler.discover(self.s+1800, self.remote)
        self.assertFalse(self.scheduler.state['source_errors'])
        self.catalog.append({'job_id':'unknown'})
        self.scheduler.discover(self.s+3600, self.remote)
        self.assertTrue(self.scheduler.state['source_errors'])

    def test_failed_download_retries_without_done_and_only_started_can_leave_window(self):
        pid = self.add(0)
        self.add(1)
        self.scheduler.discover(self.s, self.remote)
        model = self.scheduler.state['models'][pid]
        model.update(phase='ERROR', retry_at=self.s+60, transfer_started_at=self.s+1)
        self.assertEqual(self.scheduler.next_model(self.scheduler.end+1)['policy_id'], pid)
        model['phase'] = 'EVALUATION_QUEUED'
        self.assertIsNone(self.scheduler.next_model(self.scheduler.end+1))

    def test_naive_config_restart_changed_identity_rejected(self):
        changed = copy.deepcopy(self.config)
        changed['window_end'] = '2026-11-13T07:00:00+08:00'
        with self.assertRaises(D.IdentityError): D.Scheduler(changed)

    def test_end_retry_delay_budget_and_pending_alert(self):
        pid = self.add(0)
        late = self.add(1)
        self.scheduler.discover(self.s, self.remote)
        model = self.scheduler.state['models'][pid]
        model.update(phase='ERROR', transfer_started_at=self.s+1, started_at=self.s,
                     retry_at=self.scheduler.end+60)
        self.assertIsNone(self.scheduler.next_model(self.scheduler.end))
        self.assertTrue(self.scheduler.active_retry_pending(self.scheduler.end))
        self.scheduler.close_report()
        self.assertEqual(self.scheduler.state['status'], 'NEEDS_FOLLOWUP')
        self.assertEqual(next(m for m in self.scheduler.state['unprocessed'] if m['policy_id']==late)['reason'], 'WINDOW_CLOSED_NOT_STARTED')
        self.assertFalse(self.scheduler.active_retry_pending(self.s+86401))
        self.assertEqual(model['phase'], 'BLOCKED_RETRY_BUDGET')
        self.scheduler.state['last_scan'] = self.scheduler.end-1800
        self.assertTrue(self.scheduler.scan_due(self.scheduler.end-60))

    def test_existing_destination_complete_identity_not_only_components(self):
        raw = {'policy_id':'p', 'family':'pi05', 'algorithm':'bc', 'training_step':10000,
               'component_sha256':{'actor':'x'}, 'parent': {'id':'base','family':'pi05','revision':'r','contract':{'io':7}},
               'model_config':{'base_checkpoint':'old','backbone':'full'}}
        new = copy.deepcopy(raw)
        new['model_config']['base_checkpoint'] = 'new'
        new['parent']['checkpoint'] = 'new'
        D.compare_deployed_identity(raw, new)
        for key in ('policy_id','family','algorithm','training_step'):
            changed = copy.deepcopy(new)
            changed[key] = 'wrong'
            with self.assertRaises(D.IdentityError): D.compare_deployed_identity(raw, changed)
        new['parent']['contract']['io'] = 8
        with self.assertRaises(D.IdentityError): D.compare_deployed_identity(raw, new)

    def test_atomic_training_record_retry_and_no_source_hardlink(self):
        src, dst = self.root/'record', self.root/'deployed'
        src.write_bytes(b'training record')
        entry = {'size':src.stat().st_size,'sha256':D.file_sha(src)}
        D.copy_new(src,dst,entry)
        self.assertNotEqual(src.stat().st_ino,dst.stat().st_ino)
        D.copy_new(src,dst,entry)
        dst.write_bytes(b'different')
        with self.assertRaises(D.IdentityError): D.copy_new(src,dst,entry)
        self.assertEqual(src.read_bytes(),b'training record')
        self.assertFalse(list(self.root.glob('.r21-record-*')))

    def test_duplicate_start_or_smoke_cannot_touch_state_before_lock(self):
        state = self.scheduler.state_path
        state.write_bytes(b'authoritative active writer bytes, not a constructor input')
        before = state.read_bytes()
        with (self.scheduler.root/'scheduler.lock').open('a+') as lock:
            D.fcntl.flock(lock,D.fcntl.LOCK_EX|D.fcntl.LOCK_NB)
            for smoke in (False,True):
                with self.assertRaises(BlockingIOError): D.run(self.config,smoke=smoke)
                self.assertEqual(state.read_bytes(),before)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='r21-eval-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = {'policy_id':'policy-one', 'task_id':'middle','max_steps':1200}
        self.preview = {'schedule_sha256': D.SCHEDULE, 'policy_content_sha256':'e'*64,
                        'config': {'max_steps':1200,'trials':100,'task_id':'middle','policy_id':'policy-one'}}
        self.detail = {'id':'eval1','status':'QUEUED','policy_snapshot': {'policy_id':'policy-one','content_sha256':'e'*64},
                       'task_snapshot':{'task_id':'middle'}, 'schedule_sha256':D.SCHEDULE,'config':self.preview['config']}
        self.exists = False
        self.posts = 0
        self.lose_response = False

    def call(self, path, data=None):
        if path.endswith('/preview'): return self.preview
        if path == '/api/evaluations/queue' and data is not None:
            self.posts += 1
            self.exists = True
            if self.lose_response: raise TimeoutError('response lost')
            return {'id':'eval1'}
        if path == '/api/evaluations/queue':
            return {'jobs':[{'parameters':{'policy_id':'policy-one','evaluation_id':'eval1'}}] if self.exists else []}
        if '?' in path: return []  # Recover from queue even when history lags.
        return self.detail

    def test_post_response_lost_reconciles_without_second_post(self):
        self.lose_response = True
        with self.assertRaises(TimeoutError): D.evaluation({}, self.model, self.root, self.call)
        self.assertTrue(D.evaluation({}, self.model, self.root, self.call)['reconciled'])
        self.assertEqual(self.posts, 1)

    def test_unconfirmed_intent_never_blindly_reposts(self):
        self.lose_response = True
        with self.assertRaises(TimeoutError): D.evaluation({}, self.model, self.root, self.call)
        self.exists = False
        with self.assertRaisesRegex(RuntimeError, 'AMBIGUOUS'): D.evaluation({}, self.model, self.root, self.call)
        self.assertEqual(self.posts, 1)

    def test_existing_different_content_or_failed_eval_rejected(self):
        self.exists = True
        self.detail['policy_snapshot']['content_sha256'] = 'f'*64
        with self.assertRaises(D.IdentityError): D.evaluation({}, self.model, self.root, self.call)
        self.detail['policy_snapshot']['content_sha256'] = 'e'*64
        self.detail['status'] = 'FAILED'
        with self.assertRaises(D.IdentityError): D.evaluation({}, self.model, self.root, self.call)
        self.assertEqual(self.posts, 0)

    def test_new_submission_and_schedule_mismatch(self):
        self.assertFalse(D.evaluation({}, self.model, self.root, self.call)['reconciled'])
        self.assertEqual(self.posts, 1)
        self.preview['schedule_sha256'] = '0'*64
        with self.assertRaises(D.IdentityError): D.evaluation({}, self.model, self.root, self.call)


@unittest.skipUnless(os.environ.get('R21_EVIDENCE_ROOT'), 'Optional original operator receipts')
class OriginalReceiptTests(unittest.TestCase):
    def test_actual_catalog_allowlist_and_platform_response_shapes(self):
        evidence = Path(os.environ['R21_EVIDENCE_ROOT'])
        smoke = json.loads((evidence/'C/runs/R21-source-network-smoke-20261112.json').read_text())
        allowed = json.loads((evidence/'A/handoff/R21-source-allowlist.json').read_text())
        D.validate_manifest(smoke['manifest'], smoke['catalog']['models'][0],
                            {j['job_id']:j for j in allowed['jobs']}, 'sensecore')
        queued = json.loads((evidence/'C/runs/P6-queued-20261112.json').read_text())
        running = json.loads((evidence/'C/runs/P6-running-20261112.json').read_text())
        # Execute the actual public projection on the original saved effective_config;
        # no live API calls or changed request budget.
        backend = Path(__file__).parents[2]/'liberox-vla-adapter-terminal/backend/app/services/offline_job_service.py'
        tree = ast.parse(backend.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_public_evaluation')
        method.decorator_list = []
        ns = {'copy':copy,'Path':Path,'Any':object}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(backend),'exec'),ns)
        for original in queued['requests']:
            pid = original['policy_id']
            record = next(j for j in running['jobs'] if j['job']['parameters']['policy_id']==pid)
            payload = D.yaml.safe_load(record['config_text'])
            payload.update(id=record['job']['id'], status=record['job']['status'])
            detail = ns['_public_evaluation'](payload,Path(record['job']['output_path'])/'evaluation.json',detail=True)
            calls = []
            def call(path,data=None):
                calls.append((path,data))
                if path.endswith('/preview'): return original['preview']
                if '?' in path: return []
                if path.endswith('/queue'):
                    self.assertIsNone(data)
                    return queued['queue_after']
                return detail
            with tempfile.TemporaryDirectory(prefix='r21-api-original-') as tmp:
                result = D.evaluation({}, {'policy_id':pid,'task_id':original['request']['task_id'],
                    'max_steps':original['request']['max_steps']},Path(tmp),call)
                self.assertTrue(result['reconciled'])
                self.assertEqual(result['evaluation_id'],record['job']['id'])

    def test_original_transfer_resume_through_new_source_wire(self):
        evidence = Path(os.environ['R21_EVIDENCE_ROOT'])
        netpath = evidence/'C/runs/R21-reference/NSCC-R20-20261112__network_reference.py'
        from test_nightly_transfer_source import SourceTests, M
        fixture = SourceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        M.scan(fixture.contract)
        catalog = json.loads(fixture.call('catalog'))
        manifest = json.loads(fixture.call('manifest','pi05-test'))
        root = fixture.root/'consumer'
        allow = fixture.root/'allowed.json'
        allow.write_text(json.dumps({'jobs':fixture.contract['jobs']}))
        contract = fixture.root/'source-contract.json'
        contract.write_bytes(D.canonical(fixture.contract))
        start = D.datetime.fromtimestamp(time.time()-60,D.datetime.now().astimezone().tzinfo).isoformat()
        end = D.datetime.fromtimestamp(time.time()+3600,D.datetime.now().astimezone().tzinfo).isoformat()
        config = {'root':str(root),'window_start':start,'window_end':end,'sources':{'sensecore':{
            'allowlist':str(allow),'allowlist_sha256':D.file_sha(allow),'ssh_config':'unused','ssh_alias':'source'}},
            'network_reference':str(netpath),'network_reference_sha256':'700720c8d7285ddf73f1aee1542e6ebec38bf1aed2fec2e4cae15748cca392cd'}
        scheduler = D.Scheduler(config)
        scheduler.discover(time.time(),lambda source,args:catalog if args==['catalog'] else manifest)
        model = scheduler.state['models']['pi05-test']
        staged = root/'models/pi05-test/files'
        staged.mkdir()
        (staged/'actor.pt.part').write_bytes(fixture.actor[:4])
        original_run, original_popen = subprocess.run, subprocess.Popen
        commands = []
        def rewrite(args):
            if args[0]!='ssh': return args
            command = shlex.split(args[-1]);commands.append(command)
            return [sys.executable,str(Path(__file__).parents[1]/'scripts/nightly_transfer_source.py'),
                '--contract',str(contract),'--contract-sha256',D.file_sha(contract),'--request',*command]
        def run(args,**kw): return original_run(rewrite(args),**kw)
        def popen(args,**kw): return original_popen(rewrite(args),**kw)
        with patch.object(subprocess,'run',run), patch.object(subprocess,'Popen',popen), \
             patch.object(D,'publish'), patch.object(D,'evaluation',return_value={'evaluation_id':'offline-test'}):
            D.process_model(scheduler,model,{})
        self.assertEqual(model['phase'],'EVALUATION_QUEUED',model.get('error'))
        self.assertIn(['prefix','pi05-test','0','4'],commands)
        self.assertIn(['get','pi05-test','0','4'],commands)
        self.assertEqual((staged/'actor.pt').read_bytes(),fixture.actor)
        self.assertFalse((staged/'actor.pt.part').exists())
        self.assertTrue((root/'models/pi05-test/VERIFIED.json').exists())
        self.assertTrue(scheduler.start <= model['transfer_started_at'] < scheduler.end)


if __name__ == '__main__':
    unittest.main(verbosity=2)
