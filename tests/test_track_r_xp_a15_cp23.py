"""CPU regression and prereg checks for A15 CP2/CP3."""
import ast
import inspect
import unittest
from pathlib import Path
from scripts.track_r import xp_a15_cp2_d4_cpu as cp2
from scripts.track_r import xp_a15_cp2_audit_cpu as audit
from scripts.track_r import xp_a15_cp3_score_cpu as cp3
from scripts.track_r import xp_a15_cp3_protocol_cpu as protocol

ROOT=Path(cp2.__file__).resolve().parents[2]
class A15Tests(unittest.TestCase):
 def test_python_syntax(self):
  for name in ("xp_a15_cp2_d4_cpu.py","xp_a15_cp2_audit_cpu.py",
               "xp_a15_cp3_score_cpu.py","xp_a15_cp3_protocol_cpu.py",
               "xp_a17_cp3_a15_descriptive_cpu.py",
               "xp_a17_cp3_clean_timing_cpu.py",
               "xp_a17_cp3_prepare_return_cpu.py"):
   p=ROOT/"scripts/track_r"/name
   ast.parse(p.read_text(),filename=str(p))
 def test_frozen_sources(self):
  self.assertEqual(cp2.PREREG_HEAD,
   "988aaed5fb61099ca43cbe950a181a6c38ab74c7")
  self.assertEqual(cp2.V4[2],
   "b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3")
  self.assertEqual(cp2.PROTO[2],
   "d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658")
  s=inspect.getsource(cp2.provenance)
  self.assertLess(s.index("artifacts=frozen(wt)"),s.index("g1a=jread("))
  self.assertIn("validate_complete",s)
  # GPU G1a execution head differs intentionally from frozen manifest
  # producer head; never authenticate native root against producer SHA.
  self.assertIn('g1a["source_head"]',s)
  self.assertIn('g1b["source_head"]',s)
  self.assertIn('A12_G1A_FROZEN_PRODUCER',s)
  self.assertNotIn('16,16,mf_a["head"]',s)
  self.assertNotIn('32,32,mf_b["head"]',s)
  self.assertIn('"CP2_PRODUCER_VS_EXECUTION_HEADS"',s)
 def test_u_statistic(self):
  one=dict(image_id="test",labels_by_run="G;O_g;I;I",
           rho_signed_by_run=".1;.2;.3;.4")
  clip=[dict(image_id="test",run=i,g1_chosen_good=int(i<2))
        for i in range(4)]
  for k in (1,2,3,4):
   v=cp2.exact_observed([one],clip,4,k)
   self.assertLessEqual(v["E0"],v["O_raw"]+1e-12)
   self.assertLessEqual(v["E1_CLIP"],v["O_orbit"]+1e-12)
  self.assertAlmostEqual(cp2.exact_observed([one],clip,4,4)["O_orbit"],1.)
 def test_cp3_cohorts(self):
  self.assertEqual(cp3.ff_key("S1_b22","60044"),"S1_b22_60044")
  self.assertEqual(cp3.ff_key(None,"S2_b24_dev80_60044"),
                   "S2_b24_dev80_60044")
  self.assertIn("shared_tau=float(np.quantile",inspect.getsource(cp3.score))
 def test_a15_cp3_release_requires_clean_timing_and_no_matched_NFE(self):
  from scripts.track_r import xp_a17_cp3_prepare_return_cpu as release
  source=inspect.getsource(release)
  self.assertIn('len(lines)<=20',source)
  self.assertIn('A15_CP3_DESCRIPTIVE_EXTENSIONS_AUDITED_NOT_CHECKPOINT',source)
  self.assertIn('A17_CP3_CLEAN_PAIRED_TIMING_PASS',source)
  self.assertIn('DAPS4K_nominal_NFE_per_run',source)
  self.assertIn('XP_I_matched_NFE',source)
  self.assertIn('WITHDRAWN_UNRESOLVED',source)
  self.assertIn('theory_inputs',source)
  self.assertIn("def accepted_timing",source)
  self.assertIn("A17_RELEASE_ONE_VALID_CLEAN_TIMING",source)
  self.assertIn("failed_timing_attempts=failed_timing",source)
  self.assertEqual(release.files_for("G2_0.02").name,
                   "xp_a15_cp3_G2_0p02_20261030T210259Z")
  self.assertEqual(release.files_for("G2_0.10").name,
                   "xp_a15_cp3_G2_0p10_20261030T210259Z")
  self.assertIn('checked[stage]=(len(strings),len(clips))',source)
  self.assertIn('digest=build_digest(desc,summary,timing)',source)
  self.assertIn('oracle=a86(summary)',source)
  self.assertIn('comp=compute_labeled_table(desc,summary,timing)',source)
  self.assertIn('LINEAR_EXTRAPOLATION_NOT_DIRECT_POLICY_TIMING_EXCLUDES_CLIP',source)
  self.assertIn('XP_A17_CP3_COMPUTE_LABELED_POLICY_TABLE.json',source)
  self.assertIn('safe_timing=json.loads(json.dumps(timing))',source)
  self.assertIn('item["attempt_run"]=oldpath.parent.name',source)
  self.assertEqual(release.files_for("G4").name,
                   "xp_a15_cp3_G4_20261030T210259Z")
  pub=(ROOT/"scripts/track_r/xp_a17_cp3_publish.sh").read_text()
  self.assertIn('sha256sum -c SHA256SUMS',pub)
  self.assertIn('core.whitespace=cr-at-eol diff --cached --check',pub)
  self.assertNotIn('sed -i',pub)
  self.assertIn('STOP_CP3_WRONG_ORIGIN_URL',pub)
  self.assertIn('git -C "$PUBLISH" push -u origin "$BRANCH"',pub)
  self.assertIn('A17_CP3_REMOTE_CHECKPOINT_PUBLISHED_AND_VERIFIED',pub)
 def test_a15_clean_timing_uses_real_DAPS4K_and_same_y(self):
  from scripts.track_r import xp_a17_cp3_clean_timing_cpu as t
  code=inspect.getsource(t)
  self.assertIn("DAPS4K",code)
  self.assertEqual(t.PY_PC,t.PY_DAPS)
  self.assertTrue(str(t.PY_PC).endswith("/conda-envs/daps/bin/python"))
  self.assertIn("import torchdiffeq, yaml, torch, torchvision",code)
  self.assertIn("A17_TIMING_NATIVE_G3_DEP_IMPORT",code)
  self.assertIn("previous_A17_charged_gpu_hours=prior_charged",code)
  self.assertIn("A17_TIMING_SUCCESS_EXISTS_DO_NOT_DUPLICATE",code)
  self.assertIn("G6_prior_DAPS400x5_does_NOT_measure_DAPS4K_400x10",code)
  self.assertIn('and sha(pr["measurement_path"])==sha(drw["measurement_path"])',code)
  self.assertIn('canonical="S1_b22_"+did',code)
  self.assertIn('image_id=canonical',code)
  self.assertIn('canonical_ffhq_id=canonical',code)
  self.assertIn('r["image_id"]==job["image_id"]',code)
  self.assertIn("choose_gpu",code)
  self.assertIn("if rc!=0 or not entry[\"clean\"]",code)
  self.assertIn("STAGE_CAP_GPU_H=3.",code)
  self.assertIn('"G4","--phase","PROD"',code)
 def test_a15_cp3_native_receipts_ci_and_fixed_e1(self):
  from scripts.track_r import xp_a17_cp3_a15_descriptive_cpu as a17
  code=inspect.getsource(a17)
  self.assertEqual(a17.G4_RUN_PER_NFE,4000)
  self.assertEqual(a17.BOOT_SEED,20261031)
  self.assertEqual(a17.BOOT_REPS,20000)
  self.assertIn('cfg==DAPS4K',code)
  self.assertIn('["num_steps"]==10',code)
  self.assertIn('["num_steps"]==100',code)
  self.assertIn("rng.integers(0,180,size=180)",code)
  self.assertIn('frozen_Eprime8_protocol=protocol_point',code)
  self.assertIn('"WITHDRAWN_UNRESOLVED"',code)
  self.assertIn("no_matched_compute_claim=True",code)
  self.assertIn('"PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN"',code)
 def test_joint_pc_daps_readout_keeps_unmatched_nfe_explicit(self):
  from scripts.track_r import xp_a16_cp3_joint_readout_cpu as joint
  src=inspect.getsource(joint)
  self.assertIn('same_signed_measurements_verified=True',src)
  self.assertIn('frozen_M2_protocol_applied_to_one_run=False',src)
  self.assertIn('not_exact_NFE_matched=True',src)
  self.assertIn('"exact_matched_NFE":False',src)
  self.assertIn('"CP3_RETURN_AUTHORIZED":False',src)
  self.assertIn('len(pgroup)==180',src)
  self.assertIn('PC_NATIVE_FOUR_PARTICLE_METRIC_MEAN',src)
  self.assertIn('independently_spot_recompute',src)
  self.assertIn('CP3_READOUT_METRIC_SUMMARY_KEYS_',src)
  self.assertIn('CP3_READOUT_PC_DAPS_IDENTICAL_FROZEN_SIGNED_Y_',src)
  self.assertEqual(joint.STAGE_ORDER,("G4","G5","G3"))
 def test_cp3_metric_dependency_isolation_and_native_cpu_launch(self):
  for name in ("xp_a16_cp3_piq_dependency_bootstrap.sh",
               "xp_a16_cp3_metric_weights_bootstrap.sh",
               "xp_a16_cp3_piq_only_launch.sh"):
   script=(ROOT/"scripts/track_r"/name).read_text()
   self.assertIn("CUDA_VISIBLE_DEVICES",script)
   self.assertNotIn("conda install",script)
  setup=(ROOT/"scripts/track_r/xp_a16_cp3_piq_dependency_bootstrap.sh").read_text()
  self.assertIn("bad83f8e68bad6c09df09c2bdf50d1e8ccf5e852d7b64562317c27ed5eb63159",setup)
  self.assertIn("--no-deps",setup)
  self.assertIn("--target",setup)
  self.assertIn("xp_a16_cp3_metric_weights_bootstrap.sh",setup)
  weight=(ROOT/"scripts/track_r/xp_a16_cp3_metric_weights_bootstrap.sh").read_text()
  self.assertIn("397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",weight)
  self.assertIn("lpips_weights.pt",weight)
  launch=(ROOT/"scripts/track_r/xp_a16_cp3_piq_only_launch.sh").read_text()
  self.assertIn("for stage in G4 G5 G3",launch)
  self.assertIn("XP_A16_CP3_PIQ_LATEST_LOG.txt",launch)
  self.assertNotIn("xp_a15_cp2_d4_cpu.py",launch)
 def test_cp3_pending_science_outputs_are_fail_closed(self):
  for name in ("xp_a16_cp3_nfe_ledger_cpu.py",
               "xp_a16_cp3_piq_metrics_cpu.py",
               "xp_a16_cp3_imagenet_gt_y_cpu.py"):
   text=(ROOT/"scripts/track_r"/name).read_text()
   ast.parse(text,filename=name)
   self.assertIn("CP3_RETURN_AUTHORIZED=False",text)
   self.assertIn('CUDA_VISIBLE_DEVICES',text)
  nfe=(ROOT/"scripts/track_r/xp_a16_cp3_nfe_ledger_cpu.py").read_text()
  self.assertIn("matched_nfe_certified=False",nfe)
  self.assertIn("exact_pc_nfe_available=False",nfe)
  self.assertIn("1920",nfe)
  metrics=(ROOT/"scripts/track_r/xp_a16_cp3_piq_metrics_cpu.py").read_text()
  self.assertIn("piq.LPIPS(replace_pooling=True,reduction=\"none\")",metrics)
  self.assertIn("weights_only=True",metrics)
  self.assertIn('GT_used_for_evaluation_only=True',metrics)
  gt=(ROOT/"scripts/track_r/xp_a16_cp3_imagenet_gt_y_cpu.py").read_text()
  self.assertIn('seed("XPMEAS|imagenet",iid)',gt)
  self.assertIn("err<=1e-4",gt)
 def test_cp3_audit_consumes_exact_frozen_clip_slices(self):
  from scripts.track_r import xp_a15_cp3_independent_audit_cpu as ai
  auditor=inspect.getsource(ai.rerun_protocol)
  producer=inspect.getsource(protocol.main)
  self.assertIn('stage+"_CLIP_"+cohort+".csv"',auditor)
  self.assertIn('stage+"_CLIP_"+cohort+".csv"',producer)
  self.assertNotIn('stage+"_"+cohort+"_CLIP.csv"',auditor)
  self.assertIn('CP3_AUDIT_CLIP_SLICE',auditor)
 def test_cp3_release_block(self):
  s=inspect.getsource(protocol.main)
  for x in ("CP3_RETURN_AUTHORIZED=False","NFE_matched_comparison_complete=False",
            "LPIPS_SSIM_complete=False","E0_only_primary_no_CLIP"):
   self.assertIn(x,s)
 def test_pipeline_stage_order(self):
  p=(ROOT/"scripts/track_r/xp_a15_cp23_cpu_pipeline.sh").read_text()
  self.assertLess(p.index("xp_a15_cp2_audit_cpu.py"),
                  p.index("xp_a15_cp2_publish.sh"))
  self.assertLess(p.index("xp_a15_cp2_publish.sh"),
                  p.index("xp_a15_cp3_score_cpu.py"))
 def test_shared_clip_accepts_stage_specific_candidate_counts(self):
  source=inspect.getsource(cp2.clip_new)
  self.assertIn('len(result)==len(newrows)',source)
  self.assertNotIn('len(result)==4800',source)
  self.assertIn('CP2_4800_G1_CLIP',inspect.getsource(cp2.main))
 def test_cp3_only_continuation_never_replays_cp2(self):
  pipeline=(ROOT/"scripts/track_r/xp_a15_cp3_only_pipeline.sh").read_text()
  launcher=(ROOT/"scripts/track_r/xp_a15_cp3_only_launch.sh").read_text()
  self.assertIn('XP_A15_CP2_INDEPENDENT_AUDIT_PASS',pipeline)
  self.assertIn('REMOTE_CP2',pipeline)
  self.assertIn('REMOTE',launcher)
  self.assertIn('ef1cf10',pipeline)
  self.assertIn('ef1cf10',launcher)
  self.assertIn('xp_a15_cp3_score_cpu.py',pipeline)
  self.assertIn('xp_a15_cp3_protocol_cpu.py',pipeline)
  self.assertNotIn('xp_a15_cp2_d4_cpu.py',pipeline)
  self.assertNotIn('xp_a15_cp2_publish.sh',pipeline)
  self.assertIn('CP3_PROVISIONAL_CPU_ANALYSIS_COMPLETE__DO_NOT_RETURN_OR_PUBLISH_CP3',pipeline)
if __name__=="__main__": unittest.main()
