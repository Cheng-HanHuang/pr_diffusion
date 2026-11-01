"""A20 read-only reconciliation and immutable native resume regression tests."""
from __future__ import annotations
import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r.xp_a20_xpn_recover_gpu import check_chunk,path_for,job_id
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT as PAC_ROOT
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,sha
from prdiffusion.track_r.xp_gpu_telemetry import GPU,choose_gpu

ROOT=Path(__file__).resolve().parents[1]
# Production JSON readers deliberately reject sources outside PAC_ROOT.
# Synthetic fixtures MUST use a private temp directory under that root;
# do not weaken the same safe() check used on native GPU receipts.
SAFE_TEST_TMP=PAC_ROOT/"tmp"
SRC=ROOT/"scripts/track_r/xp_a20_xpn_recover_gpu.py"

class RecoveryTests(unittest.TestCase):
 def test_python_and_shell_source_structure(self):
  ast.parse(SRC.read_text(),filename=str(SRC))
  s=SRC.read_text()
  self.assertIn('"--resume"',s)
  self.assertIn('if info["status"]=="PARTIAL":cmd.append("--resume")',s)
  self.assertIn('ix=chk["original_gpu"]',s)
  self.assertIn('git(checkout,"rev-parse","HEAD")==head',s)
  self.assertIn("original_charged_gpu_h=prior",s)
  self.assertIn("abs(charged-float(old[",s)
  self.assertIn("PRESSURE_PARTIAL_PRESERVED",s)
  self.assertIn("HARD_GPU_H_CAP_STOP_OWN",s)
  self.assertIn("os.killpg",s)
  self.assertNotIn("gt_path",s)
  self.assertNotIn("model.sample(",s)
  v5=(ROOT/"scripts/track_r/xp_a20_v5_after_recovery.sh").read_text()
  self.assertIn("A18_XPN_PILOT_0_3",v5)
  self.assertIn('if [ "$count" -eq 20 ]',v5)
  self.assertNotIn("A18_XPN_MAIN_4_15",v5)
  self.assertNotIn("A18_XPN_LATE_16_63",v5)

 def fixture(self,root,n=2):
  job=dict(phase="A18_XPN_PILOT_0_3",start=0,count=n)
  head="a"*40;rootsha="b"*64
  rows=[dict(uid=f"XP|FAST|imagenet_n|test{i}|rep0",
    seed=i+22,measurement_sha256="c"*64) for i in range(n)]
  chunk=path_for(root,job);chunk.mkdir(parents=True)
  meta=dict(schema="prdiff.track_r.r0.5.fast-chunk.v1",
     status="R0_5_CHUNK_CONFIG",head=head,root_sha256=rootsha,
     start=0,count=n,gpu=2,uids=[r["uid"] for r in rows])
  (chunk/"R0_5_CHUNK_META.json").write_text(json.dumps(meta))
  return job,head,rootsha,rows,chunk,(Path("unused"),rootsha,rows)
 def add_root(self,chunk,row,head,rootsha):
  p=root_folder(chunk,row);p.mkdir()
  (p/"reconstruction.pt").write_bytes(b"frozen native tensor")
  (p/"reconstruction.png").write_bytes(b"frozen native png")
  receipt=dict(status="R0_FAST_ROOT_PASS",uid=row["uid"],
    seed=row["seed"],source_head=head,
    frozen_root_sha256=rootsha,
    measurement_sha256=row["measurement_sha256"],gt_access=False,
    method="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5",
    fast_source_head="a9758b38906b7c7950fa04491f37f6634948db26",
    reconstruction_pt_sha256=sha(p/"reconstruction.pt"),
    reconstruction_png_sha256=sha(p/"reconstruction.png"))
  (p/"R0_FAST_ROOT_COMPLETE.json").write_text(json.dumps(receipt))

 def test_partial_and_complete_resume_sha(self):
  SAFE_TEST_TMP.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="xp_a20_receipt_test_", dir=SAFE_TEST_TMP) as td:
   root=Path(td)
   job,head,digest,rows,chunk,manifest=self.fixture(root)
   self.add_root(chunk,rows[0],head,digest)
   r=check_chunk(root,job,manifest,head)
   self.assertEqual((r["status"],r["reused"],r["original_gpu"]),
                    ("PARTIAL",1,2))
   with self.assertRaises(RuntimeError):
    check_chunk(root,job,manifest,head,require_complete=True)
   self.add_root(chunk,rows[1],head,digest)
   # A SIGTERM after final root but before complete receipt is resumable.
   self.assertEqual(check_chunk(root,job,manifest,head)["reused"],2)
   c=dict(status="R0_5_CHUNK_PASS",start=0,count=2,head=head,
       new=1,reused=1,meta_sha256=sha(chunk/"R0_5_CHUNK_META.json"))
   (chunk/"R0_5_CHUNK_COMPLETE.json").write_text(json.dumps(c))
   r=check_chunk(root,job,manifest,head,require_complete=True)
   self.assertEqual(r["status"],"COMPLETE")
   self.assertEqual(r["chunk_sha256"],sha(chunk/"R0_5_CHUNK_COMPLETE.json"))

 def test_orphan_unreceipted_root_fails_closed(self):
  SAFE_TEST_TMP.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="xp_a20_receipt_test_", dir=SAFE_TEST_TMP) as td:
   root=Path(td)
   job,head,digest,rows,chunk,manifest=self.fixture(root)
   root_folder(chunk,rows[0]).mkdir()
   with self.assertRaisesRegex(RuntimeError, "STOP_R05_PROTECTED_OR_MISSING_SOURCE"):
    check_chunk(root,job,manifest,head)
   self.assertTrue(root_folder(chunk,rows[0]).exists())

 def test_gpu_admission_waits_not_co_tenancy_veto(self):
  def g(used):
   return GPU(0,"mock",97887,used,97887-used,98,
      process_pids=(999,),foreign_pids=(999,),visibility="known")
  self.assertIsNone(choose_gpu([g(81951)],8192,
      safety_margin_mib=2048,fraction=.80,clean_only=False)[0])
  self.assertIsNone(choose_gpu([g(70000)],8192,
      safety_margin_mib=2048,fraction=.80,clean_only=False)[0])
  self.assertEqual(choose_gpu([g(67000)],8192,
      safety_margin_mib=2048,fraction=.80,clean_only=False)[0],0)

if __name__=="__main__":unittest.main()
