import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import nightly_transfer_dispatch as D
import test_nightly_transfer_dispatch as fixtures


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.DispatchTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.pid = self.f.add(0)
        manifest = self.f.manifests[self.pid]
        folder = self.f.scheduler.root/'models'/self.pid/'files'
        folder.mkdir(parents=True)
        for e in manifest['files']:
            path = folder/e['relative'];path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'policy_id':self.pid}) if e['relative']=='policy.yaml' else '{}')
            e.update(size=path.stat().st_size,sha256=D.file_sha(path))
        model = manifest['models'][0]
        model['artifact_content_sha256']=D.sha(D.canonical([{k:e[k] for k in ('relative','size','sha256')} for e in manifest['files'][:3]]))
        ref = self.f.catalog[0]
        ref.update(artifact_content_sha256=model['artifact_content_sha256'],manifest_sha256=D.sha(D.canonical(manifest)))
        self.f.scheduler.discover(self.f.s,self.f.remote)
        self.model = self.f.scheduler.state['models'][self.pid]
        self.model.update(phase='REJECTED',error=D.PUBLICATION_DIRECTORY_ERROR,transfer_started_at=self.f.s+1)
        self.f.config['policy_registry']=str(self.f.root/'registry')
        self.f.scheduler.state['config_sha256']=D.sha(D.canonical(self.f.config))
        self.f.scheduler.save()
        D.immutable(folder.parent/'VERIFIED.json',{k:self.model[k] for k in ('manifest_sha256','artifact_content_sha256')})
        self.config_path=self.f.root/'config.json'
        self.config_path.write_text(json.dumps(self.f.config))
        self.folder=folder

    def recover(self):
        return D.recover_publication(self.config_path,D.file_sha(self.config_path),[self.pid])

    def test_exact_recovery_retains_all_files_and_is_idempotent(self):
        original={str(p):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        outcome=self.recover()
        self.assertFalse(outcome['download_or_evaluation_started'])
        state=json.loads(self.f.scheduler.state_path.read_text())
        self.assertEqual(state['models'][self.pid]['phase'],'VERIFIED')
        self.assertEqual(state['models'][self.pid]['transfer_started_at'],self.f.s+1)
        self.assertTrue(self.recover()['models'][0]['already_recovered'])
        self.assertEqual(original,{str(p):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()})

    def test_active_lock_and_different_error_do_not_change_state(self):
        before=self.f.scheduler.state_path.read_bytes()
        with (self.f.scheduler.root/'scheduler.lock').open('a+') as lock:
            D.fcntl.flock(lock,D.fcntl.LOCK_EX|D.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.recover()
        self.assertEqual(self.f.scheduler.state_path.read_bytes(),before)
        self.model['error']='Parent hash mismatch'
        self.f.scheduler.save()
        before=self.f.scheduler.state_path.read_bytes()
        with self.assertRaisesRegex(D.IdentityError,'Different rejection'):self.recover()
        self.assertEqual(self.f.scheduler.state_path.read_bytes(),before)

    def test_corrupt_bytes_or_post_intent_refused(self):
        original=(self.folder/'actor.pt').read_bytes()
        (self.folder/'actor.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(D.IdentityError,'nine original'):self.recover()
        (self.folder/'actor.pt').write_bytes(original)
        (self.folder.parent/'evaluation-intent.json').write_text('{}')
        with self.assertRaisesRegex(D.IdentityError,'already attempted'):self.recover()
        self.assertEqual(json.loads(self.f.scheduler.state_path.read_text())['models'][self.pid]['phase'],'REJECTED')

    def test_stage_failure_only_removes_own_links(self):
        original={p.name:(p.read_bytes(),p.stat().st_mode,p.stat().st_nlink) for p in self.folder.iterdir() if p.is_file()}
        def failed(source,dest,base):
            self.assertEqual(source.parent.name,self.pid)
            self.assertEqual((source.parent/'actor.pt').stat().st_ino,(self.folder/'actor.pt').stat().st_ino)
            raise ValueError('Still strict exporter failure')
        with self.assertRaisesRegex(ValueError,'Still strict'):
            D.export_named_input(self.folder,self.folder.parent,self.pid,self.f.root/'dest','base',failed)
        self.assertEqual(original,{p.name:(p.read_bytes(),p.stat().st_mode,p.stat().st_nlink) for p in self.folder.iterdir() if p.is_file()})
        self.assertFalse(list(self.folder.parent.glob('.publication-input-*')))


@unittest.skipUnless(os.environ.get('R21_EXPORT_ROOT'),'Pinned original exporter workspace required')
class RealExporterTests(unittest.TestCase):
    def test_original_strict_exporter_full_publish_without_extra_stage_copy(self):
        workspace=Path(os.environ['R21_EXPORT_ROOT'])
        src=workspace/'vla-adapter-rynn-iql/src'
        sys.path[:0]=[str(src),str(workspace/'vla-adapter-rynn-iql/tests')]
        from test_policy_export import make_policy
        from vla_rynn_iql import model_artifacts as E
        self.assertEqual(D.file_sha(Path(E.__file__)),'71a25312d7b6f5eb950b7211106dceba9272645f3b2dbbe7548d17a83d3df50a')
        with tempfile.TemporaryDirectory(prefix='r22-real-export-') as temp:
            root=Path(temp)
            source,raw=make_policy(root,'pi05-libero-base',algorithm='bc')
            raw.update(training_step=10000)
            source.write_text(D.yaml.safe_dump(raw))
            source.parent.rename(root/'files')
            files=root/'files'
            (files/'training_records').mkdir()
            for name in D.TRAIN_FILES:(files/'training_records'/name).write_text('{}')
            (files/'training_records/effective_config.yaml').write_text('training: {seed: 7}')
            base=root/'base';base.mkdir()
            (base/'pi05_identity.json').write_bytes((files/'pi05_identity.json').read_bytes())
            config={'parent_id':raw['parent']['id'],'parent_revision':raw['parent']['revision'],
                'parent_contract':raw['parent']['contract'],'base_checkpoint':str(base),'export_module_root':str(src),
                'export_module_sha256':D.file_sha(Path(E.__file__)),'policy_registry':str(root/'registry')}
            model={'policy_id':raw['policy_id'],'algorithm':'bc','training_steps':10000,'training_seed':7,'artifact_content_sha256':'a'*64}
            manifest={'files':[{'relative':str(p.relative_to(files)),'size':p.stat().st_size,'sha256':D.file_sha(p)} for p in files.rglob('*') if p.is_file()]}
            before={str(p): (D.file_sha(p),p.stat().st_mode,p.stat().st_nlink) for p in files.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError,'Policy directory must match'):
                E.export_policy(files/'policy.yaml',root/'registry'/model['policy_id'],str(base))
            D.publish(config,model,root,manifest)
            dest=root/'registry'/model['policy_id']
            E.validate_policy(dest/'policy.yaml',D.yaml.safe_load((dest/'policy.yaml').read_text()))
            self.assertNotEqual((files/'actor.pt').stat().st_ino,(dest/'actor.pt').stat().st_ino)
            self.assertFalse(list(root.glob('.publication-input-*')))
            self.assertEqual(before,{str(p): (D.file_sha(p),p.stat().st_mode,p.stat().st_nlink) for p in files.rglob('*') if p.is_file()})
            D.publish(config,model,root,manifest) # existing correct publication remains idempotent
            changed=copy.deepcopy(model);changed['policy_id']='wrong'
            with self.assertRaises(D.IdentityError):D.publish(config,changed,root,manifest)


if __name__=='__main__':unittest.main(verbosity=2)
