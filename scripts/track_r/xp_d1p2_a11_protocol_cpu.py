#!/usr/bin/env python3
"""Execute the six immutable A11 D1-P2 held-out protocol commands on CPU.

Loads the reference script BYTES from planner c598176, verifies its SHA,
then executes it unchanged with Python -I and all six frozen CLI variants.
No GPU, reconstruction, GT reread or new fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_d1p2_a11_export_cpu import (
 ROOT,BASE,WT,git,sha,need)
PROTOCOL_REF="c598176"
CODE_PATH="docs/track_r/protocol/gtfree_protocol.py"
SPEC_PATH="docs/track_r/protocol/GTFREE_PROTOCOL_v1.md"
CODE_SHA="d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658"
SPEC_SHA="7e9a5fd8b817a97351d07ae371b947688c8430cd5b9268fc99b954a493223919"
COMMON=["--tau","0.0500829","--m","2","--seed","20261029"]
RESULTS=("D1P2_m2_n16.json","D1P2_fresh_m2_n12.json",
         "D1P2_frontier.json","D1P2_self.json",
         "D1P2_g2.json","D1P2_e0.json")
def run_protocol(args):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    need(git("rev-parse","HEAD").decode().strip()==args.expected_head
         and not git("status","--porcelain"),"HEAD")
    inp=Path(args.input_dir)
    rec=json.loads((inp/"XP_D1P2_A11_INPUT_RECEIPT.json").read_text())
    S=inp/"XP_D1P2_16RUN_STRINGS.csv"
    C=inp/"XP_D1P2_16RUN_CLIP_DECISIONS.csv"
    need(rec["status"]=="XP_D1P2_A11_1600_FROZEN_PROTOCOL_INPUTS_PASS"
         and sha(S)==rec["strings_sha256"]
         and sha(C)==rec["clip_sha256"],"INPUTS_SHA")
    code=git("show",PROTOCOL_REF+":"+CODE_PATH)
    spec=git("show",PROTOCOL_REF+":"+SPEC_PATH)
    need(hashlib.sha256(code).hexdigest()==CODE_SHA and
         hashlib.sha256(spec).hexdigest()==SPEC_SHA,
         "FROZEN_PROTOCOL_SHA")
    out=Path(args.out)
    need(out.parent==BASE and out.name.startswith("xp_d1p2_protocol_results_")
         and not out.exists() and not out.is_symlink(),"OUTPUT")
    out.mkdir()
    script=out/"gtfree_protocol_v1_original.py"
    doc=out/"GTFREE_PROTOCOL_v1_original.md"
    with script.open("xb") as f:f.write(code)
    with doc.open("xb") as f:f.write(spec)
    need(sha(script)==CODE_SHA and sha(doc)==SPEC_SHA,
         "CODE_ROUNDTRIP_SHA")
    def cmd(stem,more,clip=True):
        c=[str(Path(sys.executable).resolve()),"-I",str(script),
            "--strings",str(S)]
        if clip:c+=["--clip",str(C)]
        c+=more+["--out",str(out/stem)]
        print("A11_D1P2_PROTOCOL_COMMAND",stem,"ARGS",more,flush=True)
        subprocess.run(c,check=True,cwd=str(WT),
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
    cmd(RESULTS[0],["--tau","0.0500829","--m","2","--nmax","16","--R","4000","--seed","20261029"])
    cmd(RESULTS[1],["--tau","0.0500829","--m","2","--nmax","12","--R","4000","--seed","20261029","--run-range","4:16"])
    cmd(RESULTS[2],["--tau","0.0500829","--m","2","--nmax","16","--R","1000","--seed","20261029","--frontier"])
    cmd(RESULTS[3],["--tau-rule","self_mad3","--m","2","--nmax","16","--R","4000","--seed","20261029"])
    cmd(RESULTS[4],["--clip-good-col","g2_chosen_good","--clip-margin-col","g2_margin",
                    "--tau","0.0500829","--m","2","--nmax","16","--R","4000","--seed","20261029"])
    cmd(RESULTS[5],["--tau","0.0500829","--m","2","--nmax","16","--R","4000","--seed","20261029"],clip=False)
    d={p:json.loads((out/p).read_text()) for p in RESULTS}
    need(all(v["protocol_version"]=="v1" for v in d.values()),"VERSION")
    a=d[RESULTS[0]]
    need(a["n_images"]==100 and a["N_MAX"]==16 and
         a["R"]==4000 and a["M"]==2 and
         a["orientation_step"] is True and
         d[RESULTS[1]]["N_MAX"]==12 and
         d[RESULTS[5]]["orientation_step"] is False and
         len(d[RESULTS[2]]["frontier"])==12,"FROZEN_GRID")
    h1=a["gain_vs_fixed_equal_runs"]>=-.01
    h3=(None if a["flag_c_expected_count"]<2 else
         a["flag_c_precision_unsolved"]>=.5)
    h4=(a["success_unflagged"]>=
         a["success_given_not_flag_c"]+.05)
    summary=dict(status="XP_D1P2_A11_FIRST_HELDOUT_PROTOCOL_V1_PASS",
         expected_head=args.expected_head,
         protocol_commit=PROTOCOL_REF,
         protocol_code_sha256=CODE_SHA,protocol_spec_sha256=SPEC_SHA,
         input_strings_sha256=rec["strings_sha256"],
         input_clip_sha256=rec["clip_sha256"],
         n_images=100,n_candidates=1600,
         outputs_sha256={p:sha(out/p) for p in RESULTS},
         primary={k:a[k] for k in (
             "success_all","runs_mean","gain_vs_fixed_equal_runs",
             "gap_vs_fixed_nmax","flag_c_expected_count",
             "flag_c_precision_unsolved","success_unflagged",
             "success_given_not_flag_c","n_unsolved_panel")},
         H1_d1p2=bool(h1),H3_d1p2=h3,H4_d1p2=bool(h4),
         H2_pending_B16_F8=True,
         no_GPU=True,no_GT_reread=True,
         original_frozen_protocol_code_unmodified=True)
    receipt=out/"XP_D1P2_A11_PROTOCOL_RESULT_RECEIPT.json"
    with receipt.open("x") as f:
        json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("XP_D1P2_A11_FROZEN_SIX_COMMANDS_PASS",flush=True)
    print("D1P2_A11_HELDOUT_H1",h1,"H3",h3,"H4",h4,flush=True)
    print("D1P2_A11_HELDOUT_PRIMARY",json.dumps(summary["primary"],sort_keys=True),flush=True)
    print("D1P2_A11_PROTOCOL_RESULT_DIR",out,flush=True)
    print("D1P2_A11_PRIMARY_SHA",sha(out/RESULTS[0]),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--input-dir",required=True)
    p.add_argument("--out",required=True)
    try:run_protocol(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_D1P2_A11_PROTOCOL "+str(e),flush=True)
        sys.exit(8)
