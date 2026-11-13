"""P8: explicit sealed 55+40 caches, new immutable 95-member Rynn versions."""
from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

import yaml

from .training_dataset_service import TrainingDatasetService


def _read(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Expected regular input: {path}")
    return json.loads(path.read_text())


def _dataset(path, count, train):
    value = _read(path)
    members = {m['run_id']: m for m in value['members']}
    if (Path(path).name != 'dataset.json' or value['id'] != Path(path).parent.name
            or TrainingDatasetService._immutable_error(value)
            or len(members) != count or len(value['members']) != count
            or Counter(m['split'] for m in members.values()) != {'train': train, 'validation': count-train}):
        raise ValueError(f"Expected unchanged frozen {count}-member dataset and splits")
    return value, members


def _sealed(path, identifier):
    from vla_rynn_iql.config_sources import read_source
    from vla_rynn_iql.io import sha256_file, stable_hash
    from vla_rynn_iql.rewards import official_inference_config, reward_derivation_config
    reader = object.__new__(TrainingDatasetService)
    reader.root = Path(path).parent.parent
    version = reader.validate_version(Path(path).parent.name, identifier)
    if version.get('legacy') or version['evaluator'] != 'rynnvalue':
        raise ValueError('Only explicit sealed RynnValue versions are supported')
    raw = read_source(Path(version['config_path']))
    prepared = _read(version['prepared_manifest_path'])
    rewards = _read(version['reward_manifest_path'])
    apath = Path(version.get('annotation_manifest_path') or Path(version['work_dir'])/'annotations/annotation_manifest.json')
    annotation = _read(apath)
    if (annotation.get('complete') is not True or annotation.get('schema_version') != 6
            or annotation.get('kind') != 'rynnvalue_annotation'
            or annotation['dataset_sha256'] != prepared['dataset_sha256']
            or stable_hash(annotation) != rewards.get('annotation_manifest_sha256')
            or annotation['annotation_config'] != official_inference_config(raw['reward'])
            or annotation['annotation_config'].get('max_frames') != 16
            or raw['reward'].get('source') != 'rynnvalue'
            or reward_derivation_config(raw['reward']) != {
                'rynnvalue': True, 'gamma': .9, 'shaping_weight': .1, 'accumulate_primitive_steps': False}):
        raise ValueError('P8 requires sealed original gamma .9 / kappa .1 / macro / 16-frame Rynn inputs')
    protected = [Path(path), Path(path).parent/'annotations'/identifier/'version.json', apath,
                 *(Path(version[k]) for k in ('config_path','prepared_manifest_path','reward_manifest_path'))]
    return version, raw, prepared, annotation, {str(p): sha256_file(p) for p in protected}


def _chunk_groups(prepared, old_members, new_members):
    from vla_rynn_iql.data import iter_unique_replay_chunks, replay_chunks
    rows = []
    parents = {rid for rid,m in old_members.items() if m['source_type']=='inference'}
    for ep,i in iter_unique_replay_chunks(prepared['episodes'],include_post_success=False):
        chunk = replay_chunks(ep)[i]
        groups = []
        if ep['run_id'] in new_members:
            groups = ['new_success_rollout' if new_members[ep['run_id']]['outcome']=='success' else 'new_failed_rollout']
        elif chunk['action_source']=='human' and ep['terminal_step'] is not None:
            groups = ['human_pre_success']
        elif ep['run_id'] in parents:
            groups = ['failed_parent']
        rows.append({'split':ep['split'],'run_id':ep['run_id'],'start':int(chunk['start']),
                     'end':int(chunk['end']),'action_source':chunk['action_source'],'groups':groups})
    return rows


def join_p8_groups(qv_rows, sidecar):
    """Join raw diagnostic data rows without relabelling newly collected rollouts.

    The caller must first verify the version's sidecar SHA and prepared/reward
    identity against the diagnostic's frozen inputs. Independent probe rows are
    not training data and are deliberately excluded.
    """
    if sidecar.get('kind')!='p8_unique_replay_groups' or sidecar.get('deduplicated') is not True:
        raise ValueError('Invalid P8 unique chunk group sidecar')
    def key(row): return row['split'],row['run_id'],int(row['start']),int(row['end'])
    expected={key(r):r for r in sidecar['rows']}
    if len(expected)!=len(sidecar['rows']): raise ValueError('Duplicate P8 membership chunk')
    result,seen=[],set()
    for row in qv_rows:
        if row['split']=='rollout': continue
        k=key(row)
        if k not in expected or k in seen or row['action_source']!=expected[k]['action_source']:
            raise ValueError('Unexpected, duplicate or mismatched P8 Q/V chunk')
        seen.add(k);result.append({**row,'groups':list(expected[k]['groups'])})
    if seen!=set(expected): raise ValueError('Missing P8 Q/V chunks')
    return result


def derive_p8_dataset(dataset_manifest: Path, old_dataset_manifest: Path, old_version_id: str,
                      new_dataset_manifest: Path, new_version_id: str, rollout_contract: Path,
                      rollout_contract_sha256: str, name: str, gamma: float = .99) -> dict:
    """No VLM or global fallback; missing/mismatched input fails before publication.

    The reviewed rollout contract is P8-submission's `new40` list (run_id, seed,
    init_state_index, success, run manifest sha256). It binds the exact D1 grid.
    Original P7's fixed-55 API is deliberately unchanged.
    """
    from vla_rynn_iql.config import validate_train_config
    from vla_rynn_iql.data import prepare_dataset, iter_unique_replay_chunks
    from vla_rynn_iql.io import atomic_json, sha256_file
    from vla_rynn_iql.rewards import _reusable_manifest_entry, materialize_reward_manifest

    if gamma not in (.9, .99) or not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError('P8 only supports gamma .9/.99 and a nonempty name up to 100 characters')
    w5 = gamma == .99
    target, members = _dataset(dataset_manifest, 95, 77)
    old, old_members = _dataset(old_dataset_manifest, 55, 44)
    new, new_members = _dataset(new_dataset_manifest, 40, 33)
    if (set(old_members) & set(new_members) or {**old_members, **new_members} != members
            or target['task_id'] != old['task_id'] or target['task_id'] != new['task_id']
            or target.get('include_post_success', True)
            or any(target[k] != old[k] or target[k] != new[k]
                   for k in ('split_seed','validation_fraction','success_consecutive_steps'))):
        raise ValueError('95 must be the exact frozen disjoint 55+40 union, unchanged splits/artifacts, no post-success')
    if (Counter((m['source_type'],m['outcome']) for m in old_members.values()) != {
            ('manual','success'):50, ('inference','failure'):5}
            or Counter((m['source_type'],m['outcome']) for m in new_members.values()) != {
                ('inference','success'):34, ('inference','failure'):6}):
        raise ValueError('Expected original 50 manual/5 failed parents and new 34 success/6 failed rollouts')
    if sha256_file(rollout_contract) != rollout_contract_sha256:
        raise ValueError('Reviewed new40 contract SHA mismatch')
    contract = _read(rollout_contract)['new40']
    if (len(contract) != 40 or {r['run_id'] for r in contract} != set(new_members)
            or Counter((r['init_state_index'],r['seed']) for r in contract) != {
                (i,s):1 for i in range(10) for s in range(100,104)}):
        raise ValueError('Only the exact D1 seed100..103 x init0..9 grid is allowed')
    for r in contract:
        m = new_members[r['run_id']]
        if r['sha256'] != m['artifacts']['manifest']['sha256'] or r['success'] != (m['outcome']=='success'):
            raise ValueError('D1 contract recording identity/outcome mismatch')
    for m in members.values():
        for a in m['artifacts'].values():
            p = Path(a['path'])
            if any(q.is_symlink() for q in (p,*p.parents)) or p.stat().st_size != a['size'] or sha256_file(p) != a['sha256']:
                raise ValueError('Frozen recording artifact changed')
    inputs = [_sealed(old_dataset_manifest, old_version_id), _sealed(new_dataset_manifest, new_version_id)]
    if inputs[0][3]['annotation_config'] != inputs[1][3]['annotation_config']:
        raise ValueError('Old/new model revision, dtype, descriptions or frame count differ')
    source_episodes, source_annotations, protected = {}, {}, {str(dataset_manifest):sha256_file(dataset_manifest),
                                                             str(rollout_contract):rollout_contract_sha256}
    for (version, raw, prepared, annotation, hashes), source_dataset in zip(inputs,(old,new)):
        expected = {m['run_id']:m for m in source_dataset['members']}
        eps = {e['run_id']:e for e in prepared['episodes']}
        anns = {e['run_id']:e for e in annotation['episodes']}
        if (len(eps)!=len(prepared['episodes']) or len(anns)!=len(annotation['episodes'])
                or set(eps)!=set(expected) or set(anns)!=set(expected)
                or prepared['source_dataset_sha256']!=source_dataset['dataset_sha256']):
            raise ValueError('Sealed prepared/annotation membership differs from frozen source')
        for rid, ep in eps.items():
            if ep['split'] != expected[rid]['split'] or _reusable_manifest_entry(ep,raw['reward'],annotation) is None:
                raise ValueError(f'Pinned 16-frame output identity/boundaries mismatch: {rid}')
            source_episodes[rid] = ep
            source_annotations[rid] = anns[rid]
            protected[anns[rid]['annotation_path']] = anns[rid]['annotation_sha256']
        protected.update(hashes)

    stamp = datetime.now(timezone.utc).isoformat()
    did, vid = 'ds_'+uuid.uuid4().hex, 'ann_'+uuid.uuid4().hex
    directory = Path(dataset_manifest).absolute().parent.parent/did
    work = directory/'annotations'/vid/'work'
    work.mkdir(parents=True,exist_ok=False)
    raw = copy.deepcopy(inputs[0][1])
    raw['paths'].update(work_dir=str(work),output_dir=str(directory/'derivation'),annotation_cache=str(work/'official_outputs'))
    raw['data'].update(selection_manifest=str(Path(dataset_manifest).absolute()),include_post_success=False,
                       bootstrap_failed_end=w5,stage_annotations_manifest=None)
    raw['reward'].update(gamma=gamma,zero_success_potential=w5,manifest_path=None,manifest_sha256=None,version_id=None)
    cfgpath = work.parent/'effective_config.yaml'
    config = validate_train_config(raw,cfgpath)
    prepared_path = prepare_dataset(config).manifest
    prepared = _read(prepared_path)
    for ep in prepared['episodes']:
        saved = source_episodes[ep['run_id']]
        for key in ('source_manifest_sha256','trajectory_sha256','observations_sha256','prompt',
                    'reward_boundaries','evaluation_chunks','split','recorded_action_count','terminal_step'):
            if ep[key] != saved[key]:
                raise ValueError(f'95 regenerated {key} differs from sealed source: {ep["run_id"]}')
    prepared['source_dataset_id'] = did
    atomic_json(prepared_path,prepared)
    annotation = copy.deepcopy(inputs[0][3])
    annotation.update(dataset_sha256=prepared['dataset_sha256'],episodes=[])
    cache = work/'official_outputs';cache.mkdir(exist_ok=True)
    for ep in prepared['episodes']:
        entry = copy.deepcopy(source_annotations[ep['run_id']])
        dest = cache/(ep['run_id']+'.npz')
        shutil.copyfile(entry['annotation_path'],dest)
        if sha256_file(dest) != entry['annotation_sha256']: raise ValueError('Official output copy SHA mismatch')
        entry['annotation_path'] = str(dest)
        annotation['episodes'].append(entry)
    atomic_json(work/'annotations/annotation_manifest.json',annotation)
    reward_path = materialize_reward_manifest(config)
    reward = _read(reward_path)
    groups_path = work/'p8_chunk_groups.json'
    atomic_json(groups_path, {'kind':'p8_unique_replay_groups','schema_version':1,
        'dataset_id':did,'version_id':vid,'source_dataset_sha256':target['dataset_sha256'],
        'prepared_manifest_sha256':sha256_file(prepared_path),'reward_manifest_sha256':sha256_file(reward_path),
        'include_post_success':False,'deduplicated':True,
        'rows':_chunk_groups(prepared,old_members,new_members)})
    raw['data']['selection_manifest'] = str(directory/'dataset.json')
    raw['reward'].update(manifest_path=str(reward_path),manifest_sha256=sha256_file(reward_path),version_id=vid)
    cfgpath.write_text(yaml.safe_dump(raw,sort_keys=False))
    parameters = {'source':'rynnvalue','gamma':gamma,'shaping_weight':.1,'accumulate_primitive_steps':False,
                  'max_frames':16,'zero_success_potential':w5,'bootstrap_failed_end':w5,'include_post_success':False}
    published = {'id':vid,'dataset_id':did,'dataset_sha256':target['dataset_sha256'],'evaluator':'rynnvalue',
        'status':'READY','complete':True,'created_at':stamp,'completed_at':stamp,'error':None,'parameters':parameters,
        'run_ids':list(members),'work_dir':str(work),'config_path':str(cfgpath),'config_sha256':sha256_file(cfgpath),
        'prepared_manifest_path':str(prepared_path),'prepared_manifest_sha256':sha256_file(prepared_path),
        'annotation_manifest_path':str(work/'annotations/annotation_manifest.json'),
        'reward_manifest_path':str(reward_path),'reward_manifest_sha256':sha256_file(reward_path),
        'reward_config':reward['reward_config'],'derivation_implementation_sha256':reward['derivation_implementation_sha256'],
        'chunk_groups_path':str(groups_path),'chunk_groups_sha256':sha256_file(groups_path),
        'derivation':{'kind':'p8_explicit_55_plus_40','source_sha256':protected,
                      'old_version_id':old_version_id,'new_version_id':new_version_id,'critic_budget':{'train_steps':10000,'batch_size':64,'seed':7}}}
    atomic_json(work.parent/'version.json',published)
    child = copy.deepcopy(target)
    child.update(id=did,name=name.strip(),parent_dataset_id=target['id'],created_at=stamp,updated_at=stamp,
        evaluation_versions=[{k:published.get(k) for k in ('id','evaluator','status','parameters','created_at','completed_at','error')}],
        evaluation_version_ids={'rynnvalue':vid},annotation_history=[],pending_annotation_id=None,
        pending_annotation_config=None,annotation_id=vid,reward_version_id=vid,robometer_version_id=None,
        annotation_status='READY',annotation_config=parameters)
    if TrainingDatasetService._immutable_error(child) or any(sha256_file(Path(p))!=h for p,h in protected.items()):
        raise ValueError('Protected input changed during derivation')
    # No source/active writes. dataset.json is the sole final publication point.
    with (directory/'dataset.json').open('x') as out: json.dump(child,out,indent=2)
    reader = object.__new__(TrainingDatasetService);reader.root=directory.parent
    reader.validate_version(did,vid)
    request = {'dataset_id':did,'task_id':target['task_id'],'reward_source':'rynnvalue',
               'train_steps':10000,'batch_size':64,'seed':7,'init_state_index':0}
    return {'dataset_id':did,'version_id':vid,'dataset_manifest':str(directory/'dataset.json'),
            'version':published,'critic_request':request if w5 else None,
            'members':95,'train_members':77,'validation_members':18,
            'unique_train_chunks':sum(1 for _ in iter_unique_replay_chunks(prepared['episodes'],split='train',include_post_success=False)),
            'groups':{'original_manual':50,'original_failed_parent':5,'new_success_rollout':34,'new_failed_rollout':6}}
