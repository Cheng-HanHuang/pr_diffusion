#!/usr/bin/env python3
"""Seal unmodified v5 draft after QA and before held XP-N outcome unblinding."""
import argparse,datetime,hashlib,json,platform
from pathlib import Path
import numpy,scipy
from check_th1_imagenet_v5_contract import check

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser()
    for name in ('draft','output','code_commit','prefix4','prefix4_sha256','clip4','clip4_sha256'):
        a.add_argument('--'+name.replace('_','-'),required=True)
    a=a.parse_args()
    if Path(a.output).exists():raise FileExistsError('refuse overwriting registered v5')
    check(a.draft)
    data=json.loads(Path(a.draft).read_text())
    for name in ('prefix4','clip4'):
        path=getattr(a,name); expected=getattr(a,name+'_sha256')
        assert digest(path)==expected==data['input_sha256'][name]
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):
        raise ValueError('full frozen code commit SHA required')
    data['version']='TH1_PREDICTIONS_v5_XP_N'
    data['status']='FROZEN_BEFORE_XP_N_HELD_RUNS_UNBLINDING'
    data['code_commit']=a.code_commit
    data['freeze_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    data['draft_sha256']=digest(a.draft)
    data['python']=platform.python_version()
    data['numpy']=numpy.__version__;data['scipy']=scipy.__version__
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print('XP_N_V5_FROZEN',str(target),'SHA256',digest(target),'UTC',data['freeze_utc'])
    print('REGISTRATION REQUIRED BEFORE XP_N RUNS 4--63 UNSEAL')
if __name__=='__main__':main()
