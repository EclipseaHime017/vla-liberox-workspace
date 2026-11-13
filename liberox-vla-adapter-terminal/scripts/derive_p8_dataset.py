#!/usr/bin/env python3
"""Seal P8 95-member Rynn inputs from two explicit 16-frame versions, without VLM."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-manifest','old-dataset-manifest','new-dataset-manifest','rollout-contract'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('old-version-id','new-version-id','rollout-contract-sha256','name'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--gamma',type=float,choices=(.9,.99),default=.99)
    p.add_argument('--runtime-root',type=Path,default=ROOT)
    args = vars(p.parse_args());runtime=args.pop('runtime_root').absolute()
    sys.path[:0]=[str(runtime/'liberox-vla-adapter-terminal'),str(runtime/'liberox-vla-adapter-terminal/scripts'),
                  str(runtime/'vla-adapter-rynn-iql/src')]
    spec=importlib.util.spec_from_file_location('backend.app.services.p8_reward_dataset',
        ROOT/'liberox-vla-adapter-terminal/backend/app/services/p8_reward_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    print(json.dumps(module.derive_p8_dataset(**args),indent=2,ensure_ascii=False))
