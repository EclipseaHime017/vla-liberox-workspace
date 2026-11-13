import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import csv
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import torch

from conftest import _episode
from test_gamma_dataset import frozen
from test_p7_dataset import p7_source
from backend.app.services.p8_reward_dataset import derive_p8_dataset, join_p8_groups
from backend.app.services.p7_reward_dataset import derive_p7_dataset
from backend.app.services.training_dataset_service import TrainingDatasetService, _stable_hash
from backend.app.services.critic_diagnostic_service import CriticDiagnosticService
from backend.app.core.exceptions import ConflictError
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import _split, iter_unique_replay_chunks, replay_end_step, replay_chunks
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.rewards import load_pinned_reward_index
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql import critic_diagnostics
from test_replay import _stats


def enrich(path):
    d=json.loads(path.read_text());d['project_id']='libero_x_vla'
    vp=path.parent/'annotations/original-version/version.json';v=json.loads(vp.read_text())
    pp=Path(v['prepared_manifest_path']);p=json.loads(pp.read_text());eps={e['run_id']:e for e in p['episodes']}
    for m in d['members']:
        r=json.loads(Path(m['artifacts']['manifest']['path']).read_text())
        m.update(source_type='manual' if r['kind']=='branch' else 'inference',
                 outcome='success' if r['success'] else 'failure',parent_run_id=r['parent_session_id'],
                 resume_step=r.get('resume_step') or 0,end_step=eps[m['run_id']]['recorded_action_count'])
    d['dataset_sha256']=_stable_hash(TrainingDatasetService._immutable_payload(d));path.write_text(json.dumps(d))
    p['source_dataset_sha256']=d['dataset_sha256']
    pp.write_text(json.dumps(p));v.update(dataset_sha256=d['dataset_sha256'],prepared_manifest_sha256=sha256_file(pp),work_dir=str(pp.parent))
    vp.write_text(json.dumps(v))
    return d


@pytest.fixture
def p8_inputs(configured,tmp_path):
    configured.raw['reward']['max_frames']=16
    oldpath,_,_=p7_source.__wrapped__(configured,tmp_path/'old')
    old=enrich(oldpath)
    cfg=copy.deepcopy(configured)
    cfg.raw['paths'].update(dataset_sources=[str(tmp_path/'newruns')],work_dir=str(tmp_path/'newwork'),
                            annotation_cache=str(tmp_path/'newcache'),output_dir=str(tmp_path/'newoutput'))
    ids=[]
    for split,n in [('train',33),('validation',7)]:
        ids.extend([f'rollout-{i}' for i in range(1000) if _split(f'rollout-{i}',7,.2)==split][:n])
    contract=[]
    for i,rid in enumerate(ids):
        _episode(tmp_path/'newruns',rid,action_count=22,success=i<34,success_from=13 if i<34 else None)
        p=next((tmp_path/'newruns').rglob(rid+'/run.json'))
        r=json.loads(p.read_text());r.update(root_session_id=rid,seed=100+i%4,init_state_index=i//4)
        p.write_text(json.dumps(r))
        contract.append({'run_id':rid,'sha256':sha256_file(p),'seed':100+i%4,'init_state_index':i//4,'success':i<34})
    newpath,_,_=frozen(cfg,tmp_path/'new','rynnvalue');new=enrich(newpath)
    combined=copy.deepcopy(old);combined.update(id='all95',include_post_success=False,
        members=old['members']+new['members'],evaluation_versions=[],evaluation_version_ids={})
    combined['dataset_sha256']=_stable_hash(TrainingDatasetService._immutable_payload(combined))
    path=tmp_path/'combined/all95/dataset.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(combined))
    cp=tmp_path/'contract.json';cp.write_text(json.dumps({'new40':contract}))
    return dict(dataset_manifest=path,old_dataset_manifest=oldpath,old_version_id='original-version',
        new_dataset_manifest=newpath,new_version_id='original-version',rollout_contract=cp,
        rollout_contract_sha256=sha256_file(cp),name='P8')


def test_two_explicit_versions_real_reduction_and_no_source_write(p8_inputs,monkeypatch,tmp_path):
    protected={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr('vla_rynn_iql.rewards.annotate_manifest',lambda *a,**k:pytest.fail('No VLM or fallback'))
    outputs=[]
    for gamma in (.99,.9):
        result=derive_p8_dataset(**p8_inputs,gamma=gamma);outputs.append(result)
        assert (result['members'],result['train_members'],result['validation_members'])==(95,77,18)
        cfg=load_train_config(Path(result['version']['config_path']))
        index=load_pinned_reward_index(cfg)
        assert cfg.raw['data']['include_post_success'] is False
        assert cfg.raw['data']['bootstrap_failed_end'] is (gamma==.99)
        assert cfg.raw['reward']['zero_success_potential'] is (gamma==.99)
        assert index['reward_config']['gamma']==gamma
        reader=object.__new__(TrainingDatasetService);reader.root=Path(result['dataset_manifest']).parent.parent
        version=reader.validate_version(result['dataset_id'],result['version_id'])
        assert CriticDiagnosticService._copy_rewards(version,tmp_path/f'diagnostic-{gamma}').is_file()
        prepared=json.loads(Path(version['prepared_manifest_path']).read_text())
        unique=list(iter_unique_replay_chunks(prepared['episodes'],split='train',include_post_success=False))
        assert result['unique_train_chunks']==len(unique)
        assert all(replay_chunks(ep)[i]['end']<=replay_end_step(ep,False) for ep,i in unique)
        assert sum(bool(e['success']) for e in prepared['episodes'])==84
        groups=json.loads(Path(version['chunk_groups_path']).read_text())
        assert groups['prepared_manifest_sha256']==version['prepared_manifest_sha256']
        assert groups['reward_manifest_sha256']==version['reward_manifest_sha256']
        newids={r['run_id'] for r in json.loads(p8_inputs['rollout_contract'].read_text())['new40']}
        for row in groups['rows']:
            if row['run_id'] in newids:
                assert row['groups'] in [['new_success_rollout'],['new_failed_rollout']]
            else: assert not any(g.startswith('new_') for g in row['groups'])
        if gamma==.99:
            assert result['critic_request']['train_steps']==10000
            assert result['critic_request']['batch_size']==64 and result['critic_request']['seed']==7
        else: assert result['critic_request'] is None
    assert all(p.read_bytes()==b for p,b in protected.items())
    assert outputs[0]['dataset_id']!=outputs[1]['dataset_id']
    # Original P7 must still reject a 95-member source, never silently relax its guard.
    with pytest.raises(ValueError,match='55 members'):
        derive_p7_dataset(Path(outputs[0]['dataset_manifest']),outputs[0]['version_id'],'not P7')


def test_real_critic_start_and_cpu_fit_group_join(p8_inputs,configured,tmp_path,monkeypatch):
    result=derive_p8_dataset(**p8_inputs)
    reader=object.__new__(TrainingDatasetService);reader.root=Path(result['dataset_manifest']).parent.parent
    reader.lock=threading.RLock();reader.repository=SimpleNamespace(upsert=lambda *a:None)
    settings=configured.section('model')
    policy={'policy_id':settings['base_id'],'label':'base','family':settings['family'],
        'base_checkpoint':settings['base_checkpoint'],'stats_key':settings['stats_key'],
        'content_sha256':'a'*64,'base_revision':'a'*64,'model_config':settings,'model_contract':model_contract(settings),
        **dict.fromkeys(('manifest','action_head','proprio_projector','training_step','compatibility_sha256'))}
    def freeze(catalog,task,destination):
        destination.mkdir();bddl=destination/'task.bddl';init=destination/'states.init'
        bddl.write_text('task');init.write_bytes(b'init')
        return SimpleNamespace(paths=lambda _:(bddl,init),manifest={'task_revision_sha256':'f'*64})
    monkeypatch.setattr('backend.app.services.task_registry.freeze_task',freeze)
    jobs=SimpleNamespace(datasets=reader,lock=threading.RLock(),jobs_root=tmp_path/'jobs',
        manager=SimpleNamespace(catalog=None),ui_config=SimpleNamespace(train_environment='vla-liberox'),
        _load_base_config=lambda *a:copy.deepcopy(configured.raw),
        preview_evaluation=lambda *a:{'policy_snapshot':policy,'task_snapshot':{},'config':{'open_loop_steps':10,'control_hz':20}},
        _new_job=lambda **kw:kw)
    queued=CriticDiagnosticService(jobs).start({**result['critic_request'],'policy_id':settings['base_id'],
                                             'max_steps':1200,'open_loop_steps':10})
    spec=yaml.safe_load(queued['config_path'].read_text())
    assert spec['critic']=={'train_steps':10000,'batch_size':64,'seed':7}
    cfg=load_train_config(Path(spec['training_config']))
    assert cfg.raw['reward']['gamma']==.99 and cfg.raw['reward']['zero_success_potential']
    assert cfg.raw['data']['bootstrap_failed_end'] and not cfg.raw['data']['include_post_success']
    monkeypatch.setattr(critic_diagnostics,'replay_stats',lambda *_:(_stats(7),_stats(8)))
    monkeypatch.setattr(critic_diagnostics,'_plots',lambda *_:None)
    monkeypatch.setattr('vla_rynn_iql.models.model_backend',lambda *_:pytest.fail('No actor'))
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        summary=critic_diagnostics.run_critic_diagnostic(cfg,cfg,tmp_path/'fit',train_steps=2,batch_size=64,seed=7,device='cpu')
    finally:torch.set_num_threads(threads)
    assert not summary['actor_updated'] and not summary['actor_loaded_for_training']
    rows=list(csv.DictReader((tmp_path/'fit/qv.csv').open()))
    sidecar=json.loads(Path(result['version']['chunk_groups_path']).read_text());groups=sidecar['rows']
    def key(r):return r['split'],r['run_id'],int(r['start']),int(r['end'])
    actual=[r for r in rows if r['split'] in ('train','validation')]
    assert len(actual)==len(groups) and {key(r) for r in actual}=={key(r) for r in groups}
    mapping={key(r):r['groups'] for r in groups}
    joined=join_p8_groups(rows,sidecar)
    assert len(joined)==len(actual)
    assert sum(r['split']=='train' for r in joined)==result['unique_train_chunks']
    with pytest.raises(ValueError,match='duplicate'):join_p8_groups(rows+[actual[0]],sidecar)
    with pytest.raises(ValueError,match='Missing'):join_p8_groups([r for r in rows if key(r)!=key(actual[0])],sidecar)
    with pytest.raises(ValueError,match='Unexpected'):join_p8_groups(rows+[{**actual[0],'run_id':'unknown'}],sidecar)
    for r in actual:
        if 'new_failed_rollout' in mapping[key(r)] and int(r['end'])==22:assert float(r['bootstrap_mask'])==1
        if 'new_success_rollout' in mapping[key(r)] and int(r['end'])==18:assert float(r['bootstrap_mask'])==0


@pytest.mark.parametrize('bad',['split','contract','four_frames','corrupt','not_ready','different_description'])
def test_reject_wrong_sources_without_ready(p8_inputs,bad):
    args=p8_inputs
    if bad=='split':
        p=args['dataset_manifest'];d=json.loads(p.read_text());d['members'][0]['split']='validation';p.write_text(json.dumps(d))
    elif bad=='contract':
        p=args['rollout_contract'];d=json.loads(p.read_text());d['new40'][0]['seed']=0;p.write_text(json.dumps(d));args['rollout_contract_sha256']=sha256_file(p)
    else:
        vp=args['new_dataset_manifest'].parent/'annotations/original-version/version.json';v=json.loads(vp.read_text())
        if bad=='not_ready':
            p=args['new_dataset_manifest'];d=json.loads(p.read_text());d['evaluation_versions'][0]['status']='RUNNING';p.write_text(json.dumps(d))
        elif bad=='corrupt':
            rm=json.loads(Path(v['reward_manifest_path']).read_text());Path(rm['episodes'][0]['reward_path']).write_bytes(b'bad')
        else:
            p=Path(v['config_path']);d=yaml.safe_load(p.read_text());d['reward']['max_frames' if bad=='four_frames' else 'camera_description']=4 if bad=='four_frames' else 'other'
            p.write_text(yaml.safe_dump(d));v['config_sha256']=sha256_file(p);vp.write_text(json.dumps(v))
    with pytest.raises((ValueError,ConflictError)):
        derive_p8_dataset(**args)
    assert len(list(args['dataset_manifest'].parent.parent.glob('*/dataset.json')))==1


def test_isolated_cli(p8_inputs,tmp_path):
    root=Path(__file__).resolve().parents[2];isolated=tmp_path/'tool'
    for rel in ['liberox-vla-adapter-terminal/scripts/derive_p8_dataset.py',
                'liberox-vla-adapter-terminal/backend/app/services/p8_reward_dataset.py']:
        dest=isolated/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dest)
    command=[sys.executable,str(isolated/'liberox-vla-adapter-terminal/scripts/derive_p8_dataset.py'),'--runtime-root',str(root)]
    for key,value in p8_inputs.items():command.extend(['--'+key.replace('_','-'),str(value)])
    result=subprocess.run(command,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['members']==95
