"""CPU-only synthetic R0.1 regression tests; no checkpoint and no PAC data."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from prdiffusion.track_r.physics import (
    ORBIT_PATTERNS, apply_orbit, magnitude, rho_signed, rho_clipped,
    to_rgb8, from_rgb8, psnr_rgb8, centered_fft_witness,
    orbit_invariance_relative,
)
from prdiffusion.track_r.prior_score import PriorScorer, prior_seed
from prdiffusion.track_r.manifest import (
    REQUIRED, assert_no_forbidden, validate_row, uid_for, write_manifest, read_manifest,
)
from prdiffusion.track_r.analysis import (
    label_candidate, auc_binary, bootstrap_cluster_rate,
    seed_bootstrap, select_policy, evaluate_policy, tally_labels,
)


def test_fft_witness_and_orbit_invariance():
    w=centered_fft_witness()
    assert w["expected_impulse"]==pytest.approx(1/384)
    x=torch.rand((2,3,256,256),generator=torch.Generator().manual_seed(123))*2-1
    errs=orbit_invariance_relative(x)
    assert set(errs)==set(ORBIT_PATTERNS)
    assert max(errs.values()) <= 1e-4
    assert torch.equal(apply_orbit(x,"111"),torch.rot90(x,2,(-2,-1)))
    for p in ORBIT_PATTERNS:
        assert torch.equal(apply_orbit(apply_orbit(x,p),p),x)


def test_signed_clipped_and_quantization():
    x=torch.full((1,3,256,256),-0.5)
    y=magnitude(x)
    assert float(rho_signed(x,y)[0]) < 1e-6
    y[0,0,0,0]=-1.0
    assert float(rho_signed(x,y)[0]) > float(rho_clipped(x,y)[0])
    x=torch.linspace(-1,1,3*256*256).reshape(1,3,256,256)
    q=to_rgb8(x)
    assert torch.equal(to_rgb8(from_rgb8(q)),q)
    assert q.dtype==torch.uint8
    a=torch.zeros((1,3,256,256),dtype=torch.uint8)
    b=a.clone()
    b[0,0,0,0]=1
    expected=-10*math.log10(1/(255**2*3*256*256))
    assert float(psnr_rgb8(a,b)[0]) == pytest.approx(expected,abs=1e-4)
    assert float(psnr_rgb8(a,a)[0]) == pytest.approx(120.0,abs=1e-4)


def test_paired_noise_equivariant_and_batch_independence():
    x=torch.rand((1,3,256,256),generator=torch.Generator().manual_seed(88))*2-1
    model=lambda xt,t: (xt * 0.15 + 0.02).repeat(1,2,1,1)  # learned sigma: first 3
    setup=dict(eps_model=model,timesteps=(50,300),draws=2,device="cpu")
    a=PriorScorer(**setup,batch_size=8).score_orbit(x,"S3_a12|DAPS|68539|rep2")
    b=PriorScorer(**setup,batch_size=64).score_orbit(x,"S3_a12|DAPS|68539|rep2")
    assert a["s_star"]=="000"
    assert a["unet_forwards"]==32
    assert a["unet_batch_calls"]==4
    assert a["max_batch_observed"]==8
    assert b["unet_batch_calls"]==1
    assert b["max_batch_observed"]==32
    assert a["noise_device"]=="cpu"
    for p in ORBIT_PATTERNS:
        assert a["S"][p]==pytest.approx(a["S"]["000"],abs=1e-6)
        assert a["S"][p]==pytest.approx(b["S"][p],abs=1e-7)
    assert prior_seed("a",50,0)==prior_seed("a",50,0)
    assert prior_seed("a",50,0)!=prior_seed("a",50,1)
    assert a==PriorScorer(**setup,batch_size=8).score_orbit(x,"S3_a12|DAPS|68539|rep2")


def test_orientation_biased_mock_recovers_planted_pattern():
    x=torch.full((1,3,256,256),-0.75)
    x[:,0,5:45,10:60]=0.9
    x[:,1,80:120,20:60]=0.7
    x[:,2,140:190,170:220]=0.8
    planted="101"
    template=apply_orbit(x,planted)
    a=0.5
    # Non-equivariant mock: its fixed spatial/colour template intentionally
    # prefers the planted orientation, independent of the measurement.
    model=lambda xt,t: (xt-math.sqrt(a)*template.expand_as(xt))/math.sqrt(1-a)
    out=PriorScorer(model,[0.99,a],timesteps=(1,),draws=2,batch_size=3).score_orbit(x,"toy")
    assert out["s_star"]==planted
    assert out["S"][planted] < 1e-10
    assert out["S"]["000"] > out["S"][planted]


def sample_row():
    return dict(uid=uid_for("S3_a12","DAPS","68539","rep2"),
        source="S3_a12",family="DAPS",image_id="68539",
        run_index=2,run_label="rep2",
        candidate_path="/tmp/exposed_candidate.png",candidate_format="png",
        gt_path="/tmp/exposed_gt.pt",gt_format="pt",
        measurement_path="/tmp/exposed_y.pt",measurement_sha256=None,
        recorded_psnr_db=6.36,recorded_psnr_convention="RGB8",
        native_selector_name=None,native_selector_value=None,
        recorded_cost_seconds=2.0,recorded_unet_forwards=None,notes="")


def test_manifest_contract_and_protection(tmp_path):
    row=sample_row()
    assert set(REQUIRED)==set(row)
    validate_row(row)
    p=tmp_path/"manifest.csv"
    write_manifest([row],p)
    saved=read_manifest(p)
    assert saved[0]["uid"]==row["uid"]
    with pytest.raises(FileExistsError):
        write_manifest([row],p)
    with pytest.raises(RuntimeError):
        assert_no_forbidden("/tmp/confirmation/payload.pt")
    with pytest.raises(RuntimeError):
        assert_no_forbidden("/tmp/confirm305.pt")
    row["candidate_path"]="/tmp/confirmation/payload.pt"
    with pytest.raises(RuntimeError):
        validate_row(row)


def test_labels_auc_and_bootstrap_reproducible():
    orbit={"000":6.0,"111":31.0}
    assert label_candidate(6.,orbit,.09,.1)=="ORBIT_ERROR"
    assert label_candidate(28.,orbit,.09,.1)=="GOOD"
    assert label_candidate(6.,{"000":6.,"111":10.},.09,.1)=="NONORBIT_CONSISTENT"
    assert label_candidate(6.,{"000":6.,"111":10.},.11,.1)=="NONORBIT_INCONSISTENT"
    assert auc_binary([0,1,1,0],[0,1,2,0])==pytest.approx(1.0)
    assert auc_binary([0,1],[1,1])==pytest.approx(0.5)
    assert auc_binary([0,0],[1,2]) is None
    rows=[dict(source="S1_b22",family="DAPS",image_id=str(i//2),failure=i<4)
          for i in range(8)]
    a=bootstrap_cluster_rate(rows,lambda x:x["failure"],"test-case",replicates=100)
    b=bootstrap_cluster_rate(rows,lambda x:x["failure"],"test-case",replicates=100)
    assert a==b
    assert a["rate"]==0.5 and a["n_images"]==4
    assert np.array_equal(seed_bootstrap("a").integers(100,size=10),
                          seed_bootstrap("a").integers(100,size=10))


def test_e0_e3_empty_feasible_fallback_and_oracles():
    def row(i,raw,rho,s,p,rr,repair):
        return dict(run_index=i,psnr_raw_db=raw,rho_signed_raw=rho,
                    rho_signed_repaired=rr,S_repaired=s,
                    p_fake_repaired=p,psnr_repaired_db=repair,
                    psnr_orbit_db={"000":raw,"111":max(raw,repair)})
    a=row(0,30,.1,2.,.8,.11,29.)
    b=row(1,10,.2,1.,.2,.09,27.)
    c=row(2,15,.3,0.5,.4,.07,25.)
    rows=[c,a,b]  # deliberately unsorted; selection must use run_index.
    assert select_policy(rows,"E0",.095)==(a,False)
    assert select_policy(rows,"E1",.095)==(a,True)
    assert select_policy(rows,"E2",.095)==(c,True)
    assert select_policy(rows,"E3",.095)==(b,True)
    assert select_policy(rows,"E2",.01)==(c,True)  # empty -> smallest repaired residual
    assert evaluate_policy(rows,"E0",.095)==30.
    assert evaluate_policy(rows,"E1",.095)==29.
    assert evaluate_policy(rows,"O_raw",.095)==30.
    assert evaluate_policy(rows,"O_orbit",.095)==30.
    assert tally_labels([{"label":"GOOD"},{"label":"ORBIT_ERROR"},{"label":"GOOD"}])=={"GOOD":2,"ORBIT_ERROR":1}
