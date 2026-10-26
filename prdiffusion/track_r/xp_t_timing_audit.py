"""Offline audit of XP-T GPU timing snapshots.

This does NOT change generation outcomes or original timing claims. A post-exit
nvidia-smi accounting tail may receive independently checked, qualified
post-hoc eligibility without rewriting the original UNVERIFIED classification.
"""
from __future__ import annotations
import json

def audit_timing_events(events, state, completion):
    gpu=state.get("gpu")
    worker=state.get("worker_pid")
    timing=state.get("timing",{})
    if (state.get("state")!="PASS" or state.get("worker_rc")!=0 or
        worker is None or not isinstance(worker,int) or
        completion.get("gpu_index")!=gpu or
        completion.get("timing")!=timing or
        completion.get("new_roots")!=9 or
        completion.get("n_images")!=3 or
        completion.get("status") not in
           ("XP_T_FAST_FFHQ_BATCH1_PASS","XP_T_FAST_IMAGENET_BATCH1_PASS")):
        raise RuntimeError("STOP_XPT_AUDIT_ORIGINAL_RECEIPT")

    samples=[]
    monotonic=[]
    for e in events:
        if e.get("action") not in ("PRE_GPU","HEARTBEAT","POST_GPU"):
            continue
        if e.get("stage")!="XP-T" or e.get("selected_gpu")!=gpu:
            raise RuntimeError("STOP_XPT_AUDIT_EVENT_IDENTITY")
        cards=[c for c in e.get("gpu_observations",[]) if c.get("index")==gpu]
        if len(cards)!=1:raise RuntimeError("STOP_XPT_AUDIT_MISSING_GPU")
        x=cards[0]
        if (x.get("visibility")!="known" or
            x.get("foreign_pids") or
            any(p!=worker for p in x.get("process_pids",[]))):
            raise RuntimeError("STOP_XPT_AUDIT_FOREIGN_OR_UNKNOWN")
        if not (0<=x.get("used_mib",-1)<=x.get("total_mib",0) and
                0<=x.get("utilization_pct",-1)<=100):
            raise RuntimeError("STOP_XPT_AUDIT_NUMERIC_FIELDS")
        if e["action"]=="HEARTBEAT" and e.get("tracked_roots")!=[worker]:
            raise RuntimeError("STOP_XPT_AUDIT_WORKER_TRACKING")
        monotonic.append(e.get("monotonic_sec"))
        samples.append({"action":e["action"],
                        "used_mib":x["used_mib"],
                        "utilization_pct":x["utilization_pct"],
                        "worker_visible":worker in x.get("process_pids",[]),
                        "all_pids":x.get("process_pids",[])})
    if (len(samples)!=timing.get("n_samples") or
        len(samples)<6 or any(not isinstance(x,(float,int)) for x in monotonic) or
        any(a>=b for a,b in zip(monotonic,monotonic[1:])) or
        samples[0]["action"]!="PRE_GPU" or
        samples[-1]["action"]!="POST_GPU" or
        samples[-1]["used_mib"]>2048 or
        samples[-1]["all_pids"] or
        samples[-1]["utilization_pct"]!=0):
        raise RuntimeError("STOP_XPT_AUDIT_SNAPSHOT_ORDER")

    # Unlike unattributed memory, which must clear the conservative
    # 2048 MiB threshold, owned active computation is positively identified
    # by its tracked PID and utilization. The actual FFHQ worker only used
    # 1603 MiB, so reusing the ghost-memory threshold here would incorrectly
    # erase all four legitimate FFHQ sampling heartbeats.
    active=[i for i,x in enumerate(samples)
        if x["action"]=="HEARTBEAT" and
           x["worker_visible"] and x["used_mib"]>512 and
           x["utilization_pct"]>0]
    if len(active)<2:raise RuntimeError("STOP_XPT_AUDIT_TOO_FEW_ACTIVE")
    # Any high memory with the worker absent must be in a post-exit heartbeat,
    # after the last observed real sampling work, with zero GPU utilization.
    trailing=[i for i,x in enumerate(samples) if
              x["action"]=="HEARTBEAT" and x["used_mib"]>2048 and
              not x["worker_visible"]]
    if trailing:
        last=trailing[-1]
        if (len(trailing)!=1 or last<=max(active) or
            last!=len(samples)-2 or
            samples[last]["utilization_pct"]!=0 or
            samples[last]["used_mib"]>max(samples[i]["used_mib"] for i in active)):
            raise RuntimeError("STOP_XPT_AUDIT_NOT_ONLY_POST_EXIT_MEMORY")
    # If an observed heartbeat is actively using GPU while our worker is not
    # visible, it is unsafe even when PID telemetry is incomplete.
    if any(x["utilization_pct"]>0 and not x["worker_visible"]
           for x in samples if x["action"]=="HEARTBEAT"):
        raise RuntimeError("STOP_XPT_AUDIT_UNATTRIBUTED_COMPUTE")

    expected=(["unattributed_gpu_memory"] if trailing else [])
    if (timing.get("timing_class")!=("UNVERIFIED" if trailing else "CLEAN") or
        timing.get("timing_valid") is not (not trailing) or
        sorted(timing.get("reasons",[]))!=expected or
        timing.get("foreign_pids")!=[]):
        raise RuntimeError("STOP_XPT_AUDIT_CLASSIFIER_DISAGREES")
    if completion.get("dataset") not in ("ffhq","imagenet"):
        raise RuntimeError("STOP_XPT_AUDIT_DATASET")
    return {
        "status":"XP_T_TIMING_POSTHOC_AUDIT_PASS",
        "dataset":completion["dataset"],
        "original_timing_class":timing["timing_class"],
        "qualified_timing_class":"CLEAN_POSTHOC" if trailing else "CLEAN_ORIGINAL",
        "eligible_from_observed_snapshots":True,
        "qualification":"POST_EXIT_NVML_MEMORY_ACCOUNTING_TAIL" if trailing else
                        "NO_UNATTRIBUTED_HIGH_MEMORY",
        "n_samples":len(samples),
        "n_active_worker_visible":len(active),
        "n_unattributed_post_exit_samples":len(trailing),
        "n_foreign_pid_samples":0,
        "n_unknown_visibility":0,
        "sampled_peak_used_mib":max(x["used_mib"] for x in samples),
        "sampling_wall_s_median":completion["sampling_wall_s_median"],
        "sampling_cuda_event_s_median":completion["sampling_cuda_event_s_median"],
        "gpu_worker_wall_s":completion["gpu_worker_wall_s"],
        "gpu_hours":completion["gpu_hours"],
        "caveat":"Sampled telemetry cannot exclude activity entirely between samples; "
                 "post-hoc classification is not a replacement for prospective "
                 "independent clean-timing replicate.",
    }
