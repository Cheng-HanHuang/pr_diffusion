#!/usr/bin/env python3
"""Read-only XP digest; bounded to <=20 lines, ends OK or ACTION NEEDED."""
from __future__ import annotations
import argparse,json
from pathlib import Path
BAD=("confirmation","confirm305")
BASE=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")
def read(p):
    if any(t in str(p).lower() for t in BAD):raise RuntimeError("FORBIDDEN_PATH")
    return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=str(BASE))
    a=p.parse_args()
    root=Path(a.root)
    if any(t in str(root).lower() for t in BAD):raise RuntimeError("FORBIDDEN_PATH")
    lines=[];action=[]
    # Only named status JSONs. Never walk experimental data trees.
    directories=sorted((d for d in root.iterdir()
        if d.is_dir() and (d.name.startswith("xp_0_") or d.name.startswith("xp_t_")
                          or d.name.startswith("xp_controller_") or d.name.startswith("xp_t_sitcom_") or d.name.startswith("xp_sitcom_replay_") or d.name.startswith("xp_fast_replay_")
                          or d.name.startswith("r0n2_"))),
        key=lambda d:d.name,reverse=True)[:64]
    seen_inventory=False
    seen_a8=False
    seen_import=False
    seen_ladder=False
    seen_fast_replay=False
    for d in directories:
        if any(t in str(d).lower() for t in BAD):raise RuntimeError("FORBIDDEN_PATH")
        if d.name.startswith("xp_0_imagenet_"):
            receipt=d/"XP_IMAGENET100_IMPORT_RECEIPT.json"
            if receipt.is_file() and not seen_import:
                obj=read(receipt)
                lines.append("ImageNet "+obj.get("status","UNKNOWN")+" n="+str(obj.get("image_count")))
                seen_import=True
        elif d.name.startswith("xp_0_"):
            fp=d/"XP_0_INVENTORY.json"
            if fp.is_file() and not seen_inventory:
                seen_inventory=True
                v=read(fp);lines.append("XP-0 "+v.get("status","UNKNOWN")+
                    " FFHQ="+str(v.get("ffhq_n"))+" IN="+str(v.get("imagenet_n")))
                if v.get("status")!="XP_0_PASS":action.append("XP-0 missing ImageNet")
        elif d.name.startswith("r0n2_a43_"):
            fp=d/"R0N2_A43_COMPLETE.json"
            if fp.is_file() and not seen_ladder:
                seen_ladder=True
                v=read(fp);lines.append("A4.3 "+v.get("status","UNKNOWN")+
                    " strata="+str(v.get("n_strata")))
        elif d.name.startswith("r0n2_"):
            fp=d/"B2_A8_RECEIPT.json"
            if fp.is_file() and not seen_a8:
                seen_a8=True
                v=read(fp);lines.append("R0N2 "+v.get("status","UNKNOWN")+
                    " n="+str(v.get("n_candidates")))
        elif d.name.startswith("xp_fast_replay_"):
            recovered=d/"XP_FAST_REPLAY_RECOVERY_STATUS.json"
            fp=d/"XP_FAST_REPLAY_STATUS.json"
            if recovered.is_file() and not seen_fast_replay:
                seen_fast_replay=True
                v=read(recovered)
                status=v.get("status","UNKNOWN")
                if (status=="XP_FAST_REPLAY_RECOVERY_PASS" and
                    v.get("r07_label_matches")==8 and
                    v.get("n_roots")==8 and
                    float(v.get("max_abs_delta_db",1e6))<=0.1 and
                    v.get("gpu_rerun_performed") is False and
                    (d/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").is_file()):
                    lines.append("FAST-REPLAY PASS n=8 max_delta_db="+str(v["max_abs_delta_db"])+
                                 " timing="+str(v.get("timing",{}).get("timing_class","UNKNOWN")))
                else:
                    lines.append("FAST-REPLAY RECOVERY_UNVERIFIED")
                    action.append("FAST replay recovery receipt failed audit")
            elif fp.is_file() and not seen_fast_replay:
                seen_fast_replay=True
                v=read(fp)
                state=v.get("state","UNKNOWN")
                lines.append("FAST-REPLAY "+state+" gpu="+str(v.get("gpu"))+
                             " max_delta_db="+str(v.get("max_abs_psnr_delta_db","—"))+
                             " timing="+str(v.get("timing_class","pending")))
                if state in ("FAILED_PREFLIGHT","FAILED_GPU","REPLAY_GATE_FAILED"):
                    action.append("FAST replay "+state+" log="+str(d/"XP_FAST_SUPERVISOR.log"))
            elif not seen_fast_replay and (d/"XP_FAST_REPLAY_PREP_RECEIPT.json").is_file():
                lines.append("FAST-REPLAY PREPARED (launch/worker pending)")
                seen_fast_replay=True
        elif d.name.startswith("xp_t_daps4k_"):
            receipt=d/"XP_T_DAPS4K_PREP.json"
            status=d/"XP_T_DAPS4K_STATUS.json"
            if status.is_file():
                v=read(status)
                lines.append("XP-T DAPS4K "+v.get("state","UNKNOWN")+
                    " completed="+str(v.get("completed_roots",0)))
            elif receipt.is_file():
                v=read(receipt)
                lines.append("XP-T DAPS4K "+v.get("status","UNKNOWN")+
                    " roots="+str(v.get("n_roots",0))+
                    " gpu_performed="+str(v.get("gpu_work_performed")))
        elif d.name.startswith("xp_t_sitcom_"):
            prep=d/"XP_T_SITCOM_PREP.json"
            live=d/"XP_T_SITCOM_STATUS.json"
            done=d/"XP_T_SITCOM_COMPLETE.json"
            if done.is_file():
                v=read(done)
                ff=v.get("per_dataset",{}).get("ffhq",{})
                im=v.get("per_dataset",{}).get("imagenet",{})
                lines.append("XP-T SITCOM "+v.get("status","UNKNOWN")+
                    " roots="+str(v.get("n_roots"))+
                    " ffhq_s="+str(round(float(ff.get("sampling_wall_s_median",0)),3))+
                    " image_s="+str(round(float(im.get("sampling_wall_s_median",0)),3))+
                    " timing_clean="+str(v.get("clean_timing_eligible")))
            elif live.is_file():
                v=read(live)
                lines.append("XP-T SITCOM "+v.get("state","UNKNOWN")+
                    " roots="+str(v.get("completed_roots",0))+"/18"+
                    " active="+str(v.get("running_roots",0))+
                    " gpu_h="+str(round(float(v.get("gpu_hours",0)),4)))
                if v.get("state","").startswith(("FAILED","STOP")):
                    action.append("XP-T SITCOM "+str(v.get("reason")))
            elif prep.is_file():
                v=read(prep)
                lines.append("XP-T SITCOM "+str(v.get("status","UNKNOWN"))+
                    " ffhq="+str(v.get("ffhq_roots"))+
                    " imagenet="+str(v.get("imagenet_roots")))
        elif d.name.startswith("xp_sitcom_replay_"):
            receipt=d/"XP_SITCOM_REPLAY_PREP.json"
            status=d/"XP_SITCOM_REPLAY_STATUS.json"
            offline=d/"XP_SITCOM_REPLAY_OFFLINE_SCORE.json"
            if status.is_file():
                v=read(status)
                lines.append("SITCOM-REPLAY "+v.get("state","UNKNOWN")+
                    " roots="+str(v.get("completed_roots","?"))+"/8"+
                    " gpu="+str(v.get("gpu"))+
                    " gpu_h="+str(round(float(v.get("gpu_hours",0)),4)))
                if v.get("state","").startswith(("FAILED","STOP")) or (
                    v.get("state")=="REPLAY_GATE_FAILED"):
                    action.append("SITCOM replay stopped: "+str(v.get("reason")))
                if offline.is_file():
                    v=read(offline)
                    lines.append("SITCOM-EQUIVALENCE "+str(v.get("status","UNKNOWN"))+
                        " delta_db="+str(round(float(v.get("max_abs_delta_db",0)),5))+
                        " exact="+str(v.get("n_exact_historical","?"))+"/8")
            elif receipt.is_file():
                v=read(receipt)
                lines.append("SITCOM-REPLAY "+v.get("status","UNKNOWN")+
                    " n="+str(v.get("n_roots")))
        elif d.name.startswith("xp_controller_"):
            fp=d/"XP_CONTROLLER_STATUS.json"
            if fp.is_file():
                v=read(fp)
                lines.append("XP-CONTROLLER "+v.get("state","UNKNOWN")+" head="+str(v.get("head",""))[:12])
                for lane in ("CRITICAL","BACKGROUND"):
                    q=v.get("lanes",{}).get(lane,{})
                    lines.append(lane+" done/running/failed/queued "+
                                 "/".join(str(q.get(k,0)) for k in ("done","running","failed","queued")))
                lines.append("GPU-H "+str(round(float(v.get("gpu_hours",0.)),3))+
                    "/400 projected="+str(v.get("projected_gpu_hours","UNKNOWN")))
                if v.get("stop_flags"):action.extend(v["stop_flags"])
        elif d.name.startswith("xp_t_timing_audit_"):
            receipt=d/"XP_T_FAST_TIMING_REAUDIT.json"
            if receipt.is_file():
                v=read(receipt)
                detail=", ".join(
                    str(z.get("dataset"))+"="+str(z.get("qualified_timing_class"))
                    for z in v.get("results",[]))
                lines.append("XP-T TIMING AUDIT "+v.get("status","UNKNOWN")+" "+detail)
                if v.get("status")!="XP_T_FAST_TIMING_REAUDIT_PASS":
                    action.append("XP-T timing re-audit failed")
        elif d.name.startswith("xp_t_fast_"):
            complete=d/"XP_T_FAST_COMPLETE.json"
            status=d/"XP_T_FAST_STATUS.json"
            if complete.is_file():
                v=read(complete)
                lines.append("XP-T FAST "+v.get("status","UNKNOWN")+
                    " roots="+str(v.get("new_roots"))+
                    " median_s="+str(round(float(v.get("sampling_wall_s_median",0)),3))+
                    " timing="+str(v.get("timing",{}).get("timing_class","UNKNOWN")))
            elif status.is_file():
                v=read(status)
                state=v.get("state","UNKNOWN")
                lines.append("XP-T FAST "+state+" gpu="+str(v.get("gpu")))
                if state.startswith("FAILED"):
                    action.append("XP-T FAST "+state+" log="+str(d/"XP_T_FAST_SUPERVISOR.log"))
            elif (d/"XP_T_FAST_PREP.json").is_file():
                lines.append("XP-T FAST PREPARED")
        else:
            fp=d/"XP_T_RECEIPT.json"
            if fp.is_file():
                v=read(fp);lines.append("XP-T "+v.get("status","UNKNOWN"))
    if not seen_a8:
        action.append("A8 receipt missing")
    if not seen_inventory:
        action.append("XP-0 inventory missing")
    if not lines:lines.append("NO_XP_RECEIPTS")
    for s in lines[:16]:print(s)
    print("ACTION NEEDED: "+"; ".join(action[:2]) if action else "OK")
if __name__=="__main__":
    try:main()
    except Exception as e:print("ACTION NEEDED: "+str(e))
