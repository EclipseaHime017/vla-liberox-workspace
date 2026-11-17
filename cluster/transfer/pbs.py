"""R21 static PBS terminal publication, separate from the A queue contract.

The operator supplies original qstat JSON plus its SHA after actual termination.
No PBS submission, fabricated queue/result, credential changes or source writes.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time

import yaml
import cluster.transfer.source as S


def jobs(allow):
    return [{**e, 'request_sha256': allow['request_canonical_sha256'],
             'max_steps': e['evaluation_max_steps'], 'run_directory': e['run']}
            for e in allow['entries']]


def load(contract):
    raw = S.safe(contract['allowlist']).read_bytes()
    if S.sha(raw) != contract['allowlist_sha256']:
        raise ValueError('PBS allowlist SHA mismatch')
    allow = json.loads(raw)
    if allow['kind'] != 'r21_nscc_pbs_static_allowlist' or allow['source_id'] != 'nscc':
        raise ValueError('Not a frozen PBS allowlist')
    return allow


def terminal(contract, allow):
    raw = S.safe(contract['pbs_evidence']).read_bytes()
    if S.sha(raw) != contract['pbs_evidence_sha256']:
        raise ValueError('Original PBS terminal evidence SHA mismatch')
    record = json.loads(raw)['Jobs'][allow['pbs_job_id']]
    if record.get('job_state') != 'F' or record.get('Exit_status') != 0:
        raise ValueError('PBS did not finish successfully')
    return record


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite training metrics')
    if isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)


def build(contract, allow, expected, pbs):
    run = S.safe(expected['run_directory'])
    for key in ('effective_config', 'server_config', 'training_config'):
        if S.file_sha(S.safe(expected[key+'_path'])) != expected[key+'_sha256']:
            raise ValueError('Frozen configuration changed: ' + key)
    if S.file_sha(run/'provenance.json') != expected['provenance_sha256']:
        raise ValueError('Frozen provenance changed')
    summary = json.loads(S.safe(run/'summary.json').read_text())
    if summary.get('status') != 'COMPLETED' or summary.get('step') != expected['training_steps']:
        raise ValueError('Run is not completed at the final step')
    policy = S.safe(summary['policy_overlay'])
    if policy.name != 'policy.yaml' or policy.parent.parent != S.safe(contract['policy_registry_root']):
        raise ValueError('Final policy outside frozen registry')
    raw = yaml.safe_load(policy.read_text())
    pid = S.token(raw['policy_id'])
    if (policy.parent.name != pid or raw['training_step'] != expected['training_steps']
            or raw['algorithm'] != expected['algorithm'] or raw['family'] != 'pi05'
            or raw['parent'] != allow['parent'] or raw['actor'] != 'actor.pt'
            or raw['base_identity'] != 'pi05_identity.json' or raw['dataset_sha256'] != expected['dataset_sha256']):
        raise ValueError('Final PBS policy/parent/dataset identity mismatch')
    if raw['parent']['sha256'] != S.sha(S.canonical({k:v for k,v in raw['parent'].items() if k != 'sha256'})):
        raise ValueError('Parent hash mismatch')
    training = yaml.safe_load(S.safe(run/'effective_config.yaml').read_text())
    if training['training']['seed'] != expected['training_seed']:
        raise ValueError('Training seed mismatch')
    steps = []
    with S.safe(run/'metrics.jsonl').open() as stream:
        for line in stream:
            metric = json.loads(line)
            finite(metric)
            if 'step' in metric: steps.append(metric['step'])
    if not steps or max(steps) != expected['training_steps']:
        raise ValueError('Metrics have no final training step')
    files = []
    for path, relative in [(policy.parent/n,n) for n in S.POLICY_FILES] + [(run/n,'training_records/'+n) for n in S.TRAIN_FILES]:
        before = S.signature(path)
        digest = S.file_sha(path)
        if S.signature(path) != before: raise ValueError('Source changed during hash')
        files.append({'index':len(files),'policy_id':pid,'source':str(path),'relative':relative,
                      'size':before['size'],'sha256':digest,'stat':before})
    hashes = {e['relative']:e['sha256'] for e in files}
    if raw['component_sha256'] != {'actor':hashes['actor.pt'],'base_identity':hashes['pi05_identity.json']}:
        raise ValueError('Frozen component SHA mismatch')
    if yaml.safe_load(policy.read_text()) != raw or json.loads((run/'summary.json').read_text()) != summary:
        raise ValueError('Final metadata changed')
    for e in files:
        if S.signature(Path(e['source'])) != e['stat']: raise ValueError('Source changed before READY')
    model = {k:expected[k] for k in ('job_id','request_sha256','task_id','max_steps','training_seed','algorithm','training_steps','priority')}
    model.update(policy_id=pid,file_indices=list(range(9)),training_config_relative='training_records/effective_config.yaml',
                 artifact_content_sha256=S.sha(S.canonical([{k:e[k] for k in ('relative','size','sha256')} for e in files[:3]])))
    return {'schema_version':1,'kind':'r21_final_model','source_id':'nscc','models':[model],'files':files,
            'terminal':{'kind':'pbs','pbs_job_id':allow['pbs_job_id'],'pbs':pbs,
                        'pbs_evidence_sha256':contract['pbs_evidence_sha256'],
                        'request_sha256':allow['request_canonical_sha256'],'run':str(run),'summary':summary}}


def publish(contract):
    allow = load(contract)
    pbs = terminal(contract,allow)
    request = json.loads(S.safe(allow['request_path']).read_text())
    if S.sha(S.canonical(request)) != allow['request_canonical_sha256']:
        raise ValueError('Original PBS request changed')
    root = S.safe(contract['catalog_root'])
    root.mkdir(parents=True,exist_ok=True)
    outcome = {'ready':[],'errors':{}}
    with S.safe(root/'scan.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for expected in jobs(allow):
            try:
                manifest = build(contract,allow,expected,pbs)
                model = manifest['models'][0]
                folder = root/'models'/model['policy_id']
                S.immutable(folder/'manifest.json',manifest)
                receipt = {k:model[k] for k in ('policy_id','artifact_content_sha256','job_id','priority')}
                receipt['manifest_sha256'] = S.sha(S.canonical(manifest))
                S.immutable(folder/'READY.json',receipt)
                outcome['ready'].append(receipt)
            except (OSError,ValueError,KeyError,TypeError) as error:
                outcome['errors'][expected['job_id']] = str(error)
    return outcome


def serve(contract,args,output):
    if time.time() >= contract['expires_at']: raise ValueError('Read capability expired')
    allow = load(contract)
    mapped = {**contract,'source_id':'nscc','jobs':[{**j,'policy_registry_root':contract['policy_registry_root']} for j in jobs(allow)]}
    root = S.safe(contract['catalog_root'])
    root.mkdir(parents=True,exist_ok=True)
    with S.safe(root/'readers.lock').open('a+') as lock:
        fcntl.flock(lock,(fcntl.LOCK_EX|fcntl.LOCK_NB) if args==['done'] else fcntl.LOCK_SH)
        if args==['done']:
            if time.time() < contract['discovery_end']: raise ValueError('Discovery window not closed')
            output.write(S.canonical({'acknowledged':True,'revoked':False,'reason':'password authentication; no key modified'}))
        else:
            S._serve(mapped,args,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--request',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raw = S.safe(args.contract).read_bytes()
    if S.sha(raw) != args.contract_sha256: raise ValueError('Contract SHA mismatch')
    contract = json.loads(raw)
    if args.publish: print(json.dumps(publish(contract)))
    elif args.request: serve(contract,args.request,sys.stdout.buffer)
    else: parser.error('Choose --publish or --request')


if __name__ == '__main__': main()
