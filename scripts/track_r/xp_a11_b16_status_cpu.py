#!/usr/bin/env python3
"""Read-only <=20-line B16 gate->production 12-hour campaign digest."""
from __future__ import annotations
import argparse,datetime,json,sys,time
from pathlib import Path
from scripts.track_r.xp_a11_b16_prod_prepare_cpu import BASE,GATE_NAME,safe
def read(p):
    return json.loads(safe(p).read_text())
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    a=p.parse_args()
    r=a.run
    if (r.parent!=BASE or not r.name.startswith("xp_a11_b16_prod_")
        or not r.is_dir() or r.is_symlink()):
        raise RuntimeError("STOP_STATUS_RUN")
    manifest=read(r/"XP_A11_B16_PROD_TASKS.json")
    group=(manifest["b22_groups"]+manifest["b24_groups"]+
           manifest["b32_b22_groups"]+manifest["b32_b24_groups"])
    statepath=r/"B16_PRODUCTION_STATE.json"
    gatepath=BASE/GATE_NAME/"B16_GATE_STATE.json"
    gate=read(gatepath) if gatepath.is_file() else {}
    state=read(statepath) if statepath.is_file() else {}
    done=state.get("done",[])
    completed=sum(group[i]["new_roots"] for i in done)
    b16_done=[i for i in done if i<1060]
    b32_done=[i for i in done if i>=1060]
    b16_count=sum(group[i]["new_roots"] for i in b16_done)
    b32_count=sum(group[i]["new_roots"] for i in b32_done)
    used=float(gate.get("spent_gpu_seconds",0.)) + float(state.get("spent_gpu_seconds",0.))
    deadline=state.get("deadline_epoch") or gate.get("deadline_epoch")
    print("B16 12H CAMPAIGN STATUS")
    print("Gate:",gate.get("status","NO_STATE"),"historical images",
          len(gate.get("done",[])),"/100")
    print("Handoff:",state.get("status","NOT_STARTED"))
    print("B16 CRITICAL:",len(b16_done),"/1060 tasks;",b16_count,"/2160 roots")
    print("B32 BACKGROUND:",len(b32_done),"/1380 tasks;",b32_count,"/2880 roots")
    print("Combined completed new roots:",completed,"/5040")
    print("Active:",len(state.get("active",[])),"physical GPU slots")
    print("B22 smoke:",state.get("b22_smoke_pass",False),
          "DEV80 smoke:",state.get("b24_smoke_pass",False))
    print("Embargo: NO new GT labels opened (generation-only)")
    print("Recorded GPU-h (completed workers):",round(used/3600,3),"/48.0")
    if deadline:
        print("Original campaign deadline UTC:",
              datetime.datetime.fromtimestamp(float(deadline),
                          datetime.timezone.utc).isoformat())
        print("Remaining wall-h:",round(max(0,float(deadline)-time.time())/3600,2))
    else:print("Deadline: waiting for replay controller")
    if state.get("status","").startswith("STOP") or gate.get("status","").startswith("STOP"):
        print("ACTION NEEDED: REPLAY_OR_PRODUCTION_STOP")
    else:print("OK")
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("ACTION NEEDED: B16_STATUS "+str(e))
        sys.exit(8)
