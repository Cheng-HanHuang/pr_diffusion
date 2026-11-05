#!/usr/bin/env python3
"""A48 fail-closed append-only GPU event journal. CPU infrastructure only.

Never repair or truncate a malformed journal automatically. Caller must hold
the project-wide flock while deciding GPU admission and writing START. Every
worker START must be durable *before* process spawn; END after wait. The journal
is a durable charge record, not a GPU reservation or science gate.
"""
from __future__ import annotations
import argparse,fcntl,json,os,sys,time,uuid
from pathlib import Path
from scripts.track_r.xp_a47_xpk_gpu_ledger_cpu import charge,FAMILIES,PHASES

def gate(ok,msg):
    if not ok:raise RuntimeError("STOP_A48_"+msg)
def _open_existing(path):
    path=Path(path)
    gate(path.is_file() and not path.is_symlink(),"MISSING_JOURNAL")
    gate(path.stat().st_nlink==1,"JOURNAL_HARDLINK")
    return path
def read_events(path):
    path=_open_existing(path)
    data=path.read_bytes()
    gate(not data or data.endswith(b"\n"),"TRUNCATED_JOURNAL")
    events=[]
    for line in data.splitlines():
        gate(bool(line.strip()),"BLANK_JOURNAL_LINE")
        events.append(json.loads(line))
    # validate event order and duplicates before accepting any mutation.
    charge(events,time.time()+1)
    return events
def append_event(path,event):
    path=_open_existing(path)
    payload=(json.dumps(event,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()
    fd=os.open(str(path),os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
    try:
        with os.fdopen(fd,"ab",buffering=0) as f:
            f.write(payload)
            os.fsync(f.fileno())
    except BaseException:
        raise
    parent_fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
def mutate(path,kind,worker_id,gpu,*,family=None,phase=None,now=None):
    gate(kind in ("START","END"),"EVENT_KIND")
    gate(type(worker_id)==str and worker_id and len(worker_id)<=150,"WORKER_ID")
    gate(type(gpu)==int and gpu in range(4),"GPU_INDEX")
    if kind=="START":
        gate(family in FAMILIES and phase in PHASES,"START_PHASE")
    t=time.time() if now is None else now
    # Exclusive journal lock is NOT the project-wide GPU admission lock.
    # Controller must separately hold the global Track R lock.
    lock=Path(str(path)+".lock")
    gate(lock.is_file() and not lock.is_symlink(),"MISSING_LOCK")
    # LOCK_EX must use a descriptor opened for writing. Some local filesystems
    # tolerate a read-only descriptor, but PAC/shared filesystems may return
    # EBADF. The sidecar lock contains no payload; r+b changes no journal bytes.
    with lock.open("r+b") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        old=read_events(path)
        starts={e["id"] for e in old if e["kind"]=="START"}
        ends={e["id"] for e in old if e["kind"]=="END"}
        if kind=="START":
            gate(worker_id not in starts,"WORKER_ID_REUSE")
            event=dict(id=worker_id,kind=kind,gpu=gpu,wall_unix=t,
                       family=family,phase=phase)
        else:
            gate(worker_id in starts and worker_id not in ends,"ORPHAN_OR_DUPLICATE_END")
            start=next(e for e in old if e["id"]==worker_id and e["kind"]=="START")
            gate(start["gpu"]==gpu,"END_GPU_MISMATCH")
            event=dict(id=worker_id,kind=kind,gpu=gpu,wall_unix=t)
        charge(old+[event],max(time.time()+1,float(t)+1))
        append_event(path,event)
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    return event
def initialize(path):
    path=Path(path)
    gate(path.parent.is_dir() and not path.is_symlink() and not path.exists(),"NEW_JOURNAL_ONLY")
    with path.open("x") as f:
        f.flush();os.fsync(f.fileno())
    lock=Path(str(path)+".lock")
    with lock.open("x") as f:
        f.flush();os.fsync(f.fileno())
    return path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--journal",required=True)
    p.add_argument("--action",choices=("init","start","end","audit"),required=True)
    p.add_argument("--id")
    p.add_argument("--gpu",type=int)
    p.add_argument("--family",choices=FAMILIES)
    p.add_argument("--phase",choices=PHASES)
    a=p.parse_args()
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    if a.action=="init":initialize(a.journal);result=dict(status="A48_JOURNAL_INITIALIZED")
    elif a.action=="audit":
        result=dict(status="A48_JOURNAL_AUDIT",ledger=charge(read_events(a.journal),time.time()))
    else:
        gate(a.id is not None and a.gpu is not None,"MISSING_EVENT_ARGUMENTS")
        result=dict(status="A48_EVENT_DURABLE",event=mutate(a.journal,a.action.upper(),a.id,a.gpu,
                             family=a.family,phase=a.phase))
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("ACTION_NEEDED_A48",repr(exc),file=sys.stderr)
        sys.exit(8)
