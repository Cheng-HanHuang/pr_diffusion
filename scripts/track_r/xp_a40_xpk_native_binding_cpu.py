#!/usr/bin/env python3
"""A40 A24 XP-K actual native parameter-binding CPU-only proof.

Frozen native code must consume the one changed scalar in the intended
measurement step. SITCOM Euler update must remain unchanged; FAST's epsilon
must reach measurement-ball projection; DAPS Hydra resolved configs must
differ at ONLY the historical pixel-Langevin tau and native MCMC code must
use tau. This emits source/hash evidence, NEVER sampling authorization.
A failure means STOP/ASK, not a looser gate or alternative config.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a32_a23_config_diff_cpu import normalize,check_exact_one_change

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SITCOM=ROOT/"external/SITCOM_ODE"
FAST=ROOT/"external/FAST_DIPS_A15"
DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
PIN_SITCOM="275ab67efbd8146bffca20155171ba6be1169c09"
PIN_FAST="a9758b38906b7c7950fa04491f37f6634948db26"
PIN_DAPS="e7a77d094167084faed19b599b96673b7bb11447"
PATCH_DAPS_SAMPLER_BLOB="ae1fc6908bb1e7f0c0349d4f82cdf20364389db8"
A39_LAST=BASE/"xp_a39_xpk_cpu_20261105T070342Z"
F8_PREP=BASE/"xp_f8_legacy_preflight_dev80seedfix_20261029T064359Z"
F8_MANIFEST_SHA="0929a94537c30faa893346a000c6cca5b51d5f45b48fbc2dfb0f6e64fa2537b9"
TARGET_DAPS=math.sqrt(2.)*.05

def gate(condition,reason):
    if not condition:raise RuntimeError("STOP_A40_NATIVE_BINDING_"+reason)
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    with Path(path).open("rb") as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
        return h.hexdigest()
def read(path):
    p=Path(path)
    gate(p.is_file() and not p.is_symlink(),"SOURCE_FILE_"+str(p))
    return json.loads(p.read_text())
def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],
                                   text=True,timeout=60).strip()
def checked_native_source(repo,head,path,allow_patched=False):
    gate(repo.is_dir() and not repo.is_symlink(),"MISSING_NATIVE_"+str(repo))
    gate(git(repo,"rev-parse","HEAD")==head,"NATIVE_HEAD_"+str(repo))
    source=repo/path
    gate(source.is_file() and not source.is_symlink(),"MISSING_SOURCE_"+path)
    work=source.read_bytes()
    expected=subprocess.check_output(["git","-C",str(repo),"show",f"HEAD:{path}"],
                                     timeout=60)
    if not allow_patched:
        gate(work==expected,"NATIVE_SOURCE_DRIFT_"+path)
    else:
        gate(git(repo,"hash-object","--",path)==PATCH_DAPS_SAMPLER_BLOB,
             "PATCHED_DAPS_BLOB_DRIFT")
    ast.parse(work.decode("utf-8")) if path.endswith(".py") else None
    return work.decode("utf-8"),{
       "path":str(source),"actual_sha256":sha_bytes(work),
       "committed_sha256":sha_bytes(expected),
       "content_matches_git_head":work==expected,
       "source_head":head,
       "original_A10_patch_exception":bool(allow_patched)}

def verify_sitcom():
    src,proof=checked_native_source(SITCOM,PIN_SITCOM,"sampler.py")
    tree=ast.parse(src)
    lgvd=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="LangevinDynamics")
    daps=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DAPS")
    methods={n.name:ast.get_source_segment(src,n) for n in lgvd.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    gate("self.tau = tau" in methods["__init__"]
       and "self.tau ** 2" in methods["sample"]
       and "self.lgvd = LangevinDynamics(**lgvd_config)" in ast.get_source_segment(src,daps),
       "SITCOM_LGVD_NATIVE_TAU_NOT_IN_LOSS")
    # The independent Euler component is not changed by the new tau.
    other=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DiffusionSampler")
    euler=next(n for n in other.body if isinstance(n,ast.FunctionDef) and n.name=="_euler")
    euler_src=ast.get_source_segment(src,euler)
    gate("self.tau" not in euler_src and "LangevinDynamics" not in euler_src,
         "SITCOM_EULER_MUST_BE_INDEPENDENT")
    adapter=(WT/"scripts/track_r/xp_sitcom_replay_gpu.py").read_text()
    gate('"tau":cfg["lgvd_tau"]' in adapter
         and 'lgvd_config={' in adapter,
         "SITCOM_NATIVE_ADAPTER_BINDING")
    a=read(A39_LAST/"SITCOM_BASELINE.json")
    b=read(A39_LAST/"SITCOM_PROPOSED.json")
    c=check_exact_one_change(a,b,"SITCOM","lgvd_tau")
    proof.update(adapter_sha256=sha(WT/"scripts/track_r/xp_sitcom_replay_gpu.py"),
                 source_tau_loss="operator.error(x,measurement)/(2*tau^2)",
                 frozen_euler_source_sha256=sha_bytes(euler_src.encode()),
                 one_leaf_receipt=c,
                 original_parameter=0.01,registered_target=0.05)
    return proof

def verify_fast():
    src,proof=checked_native_source(FAST,PIN_FAST,"main.py")
    tree=ast.parse(src)
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="FAST_DIPS")
    methods={n.name:ast.get_source_segment(src,n) for n in klass.body
             if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    gate("self.epsilon = float(epsilon)" in methods["__init__"]
         and "(self.epsilon / nrm).clamp_max(1.0)" in methods["sample"]
         and "nrm <= self.epsilon" in methods["sample"],
         "FAST_NATIVE_PROJECTED_L2_BALL_BINDING")
    adapter=(WT/"scripts/track_r/xp_fast_gpu.py").read_text()
    gate('epsilon=conf["epsilon"]' in adapter
         and "upstream.FAST_DIPS(" in adapter,
         "FAST_NATIVE_ADAPTER_BINDING")
    a=read(A39_LAST/"FAST_DIPS_BASELINE.json")
    b=read(A39_LAST/"FAST_DIPS_PROPOSED.json")
    c=check_exact_one_change(a,b,"FAST_DIPS","epsilon")
    proof.update(adapter_sha256=sha(WT/"scripts/track_r/xp_fast_gpu.py"),
                 source_epsilon_rule="sample projects measurement residual into L2 radius epsilon",
                 one_leaf_receipt=c,
                 original_parameter=0.05,registered_target=33.26)
    return proof

def verify_daps():
    # Native historical 400x5 DAPS is the original PATCHED source that
    # passed F8, NOT the pristine DAPS-4K 400x10 source.
    src,proof=checked_native_source(DAPS,PIN_DAPS,"sampler.py",True)
    f8=read(F8_PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")
    gate(sha(F8_PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")==F8_MANIFEST_SHA
         and len(f8["rows"])==8,"PREVIOUS_F8_MANIFEST")
    from scripts.track_r.xp_f8_legacy_replay_gpu import native_command,env
    from scripts.track_r.xp_f8_legacy_preflight_cpu import MODEL,MODEL_SHA
    gate(sha(MODEL)==MODEL_SHA,"DAPS_CHECKPOINT_SHA")
    row=f8["rows"][0]
    gate(row["source"]=="S1_b22" and row["family"]=="DAPS"
         and sha(row["measurement_path"])==row["measurement_sha256"],
         "DAPS_ORIGINAL_F8_B22_ROW")
    import yaml
    dest=BASE/"xp_a40_native_hydra_probe_000"
    baseline=native_command(row,dest)
    after=baseline+["task.pixel.mcmc_sampler_config.tau="+repr(TARGET_DAPS)]
    def resolved(cmd):
        e=env(row)
        e["CUDA_VISIBLE_DEVICES"]=""
        e["PYTHONDONTWRITEBYTECODE"]="1"
        e["PYTHONNOUSERSITE"]="1"
        x=subprocess.run(cmd+["--cfg","job","--resolve"],
                         cwd=DAPS,env=e,capture_output=True,
                         text=True,timeout=300)
        gate(x.returncode==0,"DAPS_HYDRA_CONFIG_"+x.stderr[-500:])
        return yaml.safe_load(x.stdout)
    original=resolved(baseline)
    adjusted=resolved(after)
    key=("task","pixel","mcmc_sampler_config","tau")
    def at(obj):
        for x in key:obj=obj[x]
        return obj
    gate(math.isclose(float(at(original)),.01,rel_tol=0,abs_tol=1e-12)
         and math.isclose(float(at(adjusted)),TARGET_DAPS,rel_tol=0,abs_tol=1e-10),
         "DAPS_HYDRA_NATIVE_TAU_VALUE")
    left,right=normalize(original),normalize(adjusted)
    changes=[k for k in left if k not in right or left[k]!=right[k]]
    gate(set(left)==set(right) and changes==[key],
         "DAPS_HYDRA_NOT_EXACTLY_ONE_NATIVE_LEAF")
    # The pinned source makes the measurement Langevin penalty depend on
    # tau; do not trust Hydra resolution alone.
    # The historical patched DAPS checkout may keep the MCMC primitive
    # under a different module path than pristine DAPS-4K. Discover only
    # tracked source files and require a concrete native tau-loss witness.
    candidates=[]
    for candidate in sorted(DAPS.rglob("*.py")):
        if (".git" in candidate.parts or "__pycache__" in candidate.parts
                or candidate.is_symlink()):
            continue
        relative=candidate.relative_to(DAPS).as_posix()
        if "site-packages" in relative or "external/" in relative:
            continue
        try:
            data=candidate.read_text()
        except (UnicodeDecodeError,OSError):
            continue
        if not ("tau" in data and ("mcmc" in data.lower() or
             "Langevin" in data or "langevin" in data)):
            continue
        if ("tau ** 2" not in data and "tau**2" not in data and
            "self.tau" not in data and "/ tau" not in data):
            continue
        if relative=="sampler.py":
            # Historical B20/B21 default-off patched sampler is already
            # strictly checked via the frozen blob above.
            candidates.append(dict(path=relative,source_sha256=sha(candidate),
                                   exception="accepted_prior_F8_patch"))
            continue
        verified,sr=checked_native_source(DAPS,PIN_DAPS,relative)
        candidates.append(sr)
    gate(candidates,"DAPS_MCMC_TAU_NATIVE_SOURCE_NOT_FOUND")
    proof.update(mcmc_native_source_witnesses=candidates,
                 head_native_patch_sha=PATCH_DAPS_SAMPLER_BLOB,
        Hydra_original_tau=float(at(original)),Hydra_calibrated_tau=float(at(adjusted)),
        exact_changed_leaf=".".join(key),
        full_original_yaml_sha256=sha_bytes(json.dumps(original,sort_keys=True,default=str).encode()),
        full_calibrated_yaml_sha256=sha_bytes(json.dumps(adjusted,sort_keys=True,default=str).encode()),
        previous_F8_manifest_sha256=F8_MANIFEST_SHA,
        original_parameter=0.01,registered_target=TARGET_DAPS)
    return proof

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"WRONG_DIRTY_CODE")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a40_xpk_binding_")
         and not out.exists() and not out.is_symlink(),"NEW_OUTPUT")
    gate(A39_LAST.is_dir() and
         (A39_LAST/"A39_XPK_CPU_GATE_INVENTORY.json").is_file()
         and read(A39_LAST/"A39_XPK_CPU_GATE_INVENTORY.json")["status"]==
         "A39_A24_CPU_SOURCE_SCREENS_COMPLETE_NOT_NATIVE_GATE",
         "A39_PRIOR_CHECKPOINT")
    results={};errors={}
    # Inspect all families even if one fails, but no GPU when any fails.
    for family,func in (("SITCOM",verify_sitcom),
                        ("FAST_DIPS",verify_fast),
                        ("DAPS_FFHQ",verify_daps)):
        try:
            results[family]=func()
            print("A40_NATIVE_SOURCE_BINDING_PASS",family,flush=True)
        except Exception as exc:
            errors[family]=repr(exc)
            print("ACTION_NEEDED_A40_NATIVE_SOURCE_BINDING",
                  family,repr(exc),flush=True)
    out.mkdir(mode=0o700)
    status="A40_ALL3_NATIVE_SOURCE_BINDINGS_PASS" if not errors else "A40_NATIVE_BINDINGS_PARTIAL_STOP"
    record=dict(status=status,head=a.expected_head,
       a39_cpu_input=str(A39_LAST),
       gate_i_native_source_binding=results,
       failed_or_not_verified=errors,
       A23_5b_gate_i_committed_actual_adapter_diff=False,
       A23_5b_gate_ii_two_B22_root_replay_passed=False,
       A23_5b_gate_iii_f8_previous_replay_documented=True,
       calibrated_XP_K_generation_authorized=False,
       no_gpu_jobs_launched=True,
       no_GT_image_loaded=True,
       main_paper_cut_closed=True)
    with (out/"A40_NATIVE_BINDING_CPU_RECEIPT.json").open("x") as f:
        json.dump(record,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("A40_CPU_SOURCE_RECEIPT",out,flush=True)
    gate(not errors,"SCIENCE_GATE_FAILURE_STOP_ASK")
    print("A40_SOURCE_BINDING_EVIDENCE_PASS_BUT_ADAPTED_WORKER_DIFF_AND_IDENTITY_REPLAY_PENDING",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception:
        traceback.print_exc()
        sys.exit(8)
