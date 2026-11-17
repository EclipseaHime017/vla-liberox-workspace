import io
import json
from pathlib import Path
import sys
import unittest
import time

sys.path.insert(0,str(Path(__file__).parents[2]))
import cluster.transfer.pbs as P
import cluster.transfer.dispatch as D
import test_nightly_transfer_source as fixtures


class PBSTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SourceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.raw['dataset_sha256'] = 'd'*64
        (f.policy/'policy.yaml').write_text(P.yaml.safe_dump(f.raw))
        (f.run/'metrics.jsonl').write_text('{"step":10000,"loss":0.1}\n')
        request = f.root/'request.json'
        request.write_text('{"experiments":["long_s11"]}')
        e = {'job_id':'26185784.pbs101:long_s11','pbs_job_id':'26185784.pbs101','run':str(f.run),
             'task_id':'long','evaluation_max_steps':1600,'training_seed':7,'training_steps':10000,
             'algorithm':'bc','priority':20,'dataset_sha256':'d'*64,'provenance_sha256':P.S.file_sha(f.run/'provenance.json')}
        for key in ('effective_config','server_config','training_config'):
            e[key+'_path'] = str(f.run/'effective_config.yaml')
            e[key+'_sha256'] = P.S.file_sha(f.run/'effective_config.yaml')
        self.allow = {'kind':'r21_nscc_pbs_static_allowlist','source_id':'nscc','entries':[e],
            'request_path':str(request),'request_canonical_sha256':P.S.sha(P.S.canonical(json.loads(request.read_text()))),
            'pbs_job_id':'26185784.pbs101','parent':f.raw['parent']}
        self.allowed = f.root/'allow.json'
        self.allowed.write_text(json.dumps(self.allow))
        self.evidence = f.root/'qstat.json'
        self.evidence.write_text(json.dumps({'Jobs':{'26185784.pbs101':{'job_state':'F','Exit_status':0}}}))
        self.contract = {'allowlist':str(self.allowed),'allowlist_sha256':P.S.file_sha(self.allowed),
            'pbs_evidence':str(self.evidence),'pbs_evidence_sha256':P.S.file_sha(self.evidence),
            'catalog_root':str(f.root/'pbs_catalog'),'policy_registry_root':str(f.registry),
            'expires_at':time.time()+3600,'discovery_end':0}

    def test_pbs_final_native_evidence_and_client_wire_without_queue(self):
        outcome = P.publish(self.contract)
        self.assertFalse(outcome['errors'],outcome)
        buf = io.BytesIO()
        P.serve(self.contract,['manifest','pi05-test'],buf)
        manifest = json.loads(buf.getvalue())
        self.assertNotIn('job',manifest['terminal'])
        self.assertEqual(manifest['terminal']['kind'],'pbs')
        allowed = {e['job_id']:e for e in D.allowlisted_jobs(self.allow)}
        D.validate_manifest(manifest,outcome['ready'][0],allowed,'nscc')
        buf = io.BytesIO()
        P.serve(self.contract,['get','pi05-test','0','3'],buf)
        self.assertEqual(buf.getvalue(),self.f.actor[3:])
        buf = io.BytesIO()
        P.serve(self.contract,['done'],buf)
        self.assertFalse(json.loads(buf.getvalue())['revoked'])
        self.assertEqual(P.publish(self.contract),outcome)

    def test_failed_pbs_or_wrong_config_and_nonfinite_never_ready(self):
        self.evidence.write_text(json.dumps({'Jobs':{'26185784.pbs101':{'job_state':'F','Exit_status':1}}}))
        self.contract['pbs_evidence_sha256'] = P.S.file_sha(self.evidence)
        with self.assertRaisesRegex(ValueError,'successfully'): P.publish(self.contract)
        self.evidence.write_text(json.dumps({'Jobs':{'26185784.pbs101':{'job_state':'F','Exit_status':0}}}))
        self.contract['pbs_evidence_sha256'] = P.S.file_sha(self.evidence)
        (self.f.run/'metrics.jsonl').write_text('{"step":10000,"loss":NaN}\n')
        self.assertTrue(P.publish(self.contract)['errors'])
        (self.f.run/'effective_config.yaml').write_text('training: {seed: 99}')
        self.assertTrue(P.publish(self.contract)['errors'])
        self.assertFalse(list(Path(self.contract['catalog_root']).glob('models/*/READY.json')))

    def test_pbs_terminal_client_rejects_different_run(self):
        outcome = P.publish(self.contract)
        _,manifest = P.S.ready_manifest(Path(self.contract['catalog_root']),'pi05-test')
        manifest['terminal']['run'] = '/another/run'
        ref = dict(outcome['ready'][0],manifest_sha256=P.S.sha(P.S.canonical(manifest)))
        with self.assertRaises(D.IdentityError):
            D.validate_manifest(manifest,ref,{e['job_id']:e for e in D.allowlisted_jobs(self.allow)},'nscc')


if __name__ == '__main__': unittest.main(verbosity=2)
