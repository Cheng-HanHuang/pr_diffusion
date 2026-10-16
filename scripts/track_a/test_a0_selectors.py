"""Synthetic metadata-only A0 unit tests. No image/model/FFT/PAC input."""
import unittest
from scripts.track_a.a0_selectors import (
    AuditStop, best, required_candidates, native, oracle, ladders, paired, summarize)
from scripts.track_a.a0_source_reconcile import pycache, source_only_names
from scripts.track_a.a0_historical_compare import align_frozen_b21_b22

SHA = "a"*64
MODEL = "b"*64

def rows(cohort, method, psnrs=None, scores=None, losses=None, representation="raw_float"):
    layout = {
        ("B22","DAPS"): [("rep",0),("rep",1)],
        ("B22","SITCOM"): [("run",i) for i in range(4)],
        ("B22","NP"): [("lf",i) for i in range(100,104)] +
                       [("s2_preproj_lam001",i) for i in range(100,104)],
        ("B24","DAPS"): [("rep",i) for i in range(4)],
        ("B24","SITCOM"): [("rep",i) for i in range(4)],
        ("B24","NP"): [("lf",i) for i in range(4)],
    }[(cohort,method)]
    psnrs = psnrs or [20.0+i for i in range(len(layout))]
    scores = scores or [float(i) for i in range(len(layout))]
    losses = losses or [float(i) for i in range(len(layout))]
    return [{"cohort":cohort,"image_id":"60044","method":method,"group":g,
             "index":i,"candidate_id":f"{g}-{i}","representation":representation,
             "measurement_sha256":SHA,"model_sha256":MODEL,"source_id":"frozen",
             "psnr":p,"postproj_lf_mse":s,"step160_correction_norm":s,
             "daps_signed_loss":loss}
            for (g,i),p,s,loss in zip(layout,psnrs,scores,losses)]

class SelectorsTest(unittest.TestCase):
    def test_daps_exact_strict_margin(self):
        c=rows("B22","DAPS",psnrs=[15,30],losses=[9.0,8.3])
        self.assertEqual(native("B22","DAPS",c,2)["index"],0)
        c[1]["daps_signed_loss"]=8.2999
        self.assertEqual(native("B22","DAPS",c,2)["index"],1)
        self.assertEqual(oracle(c,2)["index"],1)
        with self.assertRaises(AuditStop):native("B24","DAPS",rows("B24","DAPS"),4)
    def test_np_heterogeneous_full_eight_and_ties(self):
        c=rows("B22","NP",psnrs=[15,28,18,27,29,30,31,32],scores=[3,2,2,4,5,6,7,8])
        self.assertEqual(native("B22","NP",c,8)["candidate_id"],"lf-101")
        self.assertEqual(oracle(c,8)["candidate_id"],"s2_preproj_lam001-103")
        self.assertEqual(len([r for r in ladders("B22","NP",c) if r["status"]=="PASS"]),26)
    def test_sitcom_step160_and_absence(self):
        c=rows("B22","SITCOM",scores=[2,1,1,3])
        self.assertEqual(native("B22","SITCOM",c,4)["index"],1)
        with self.assertRaises(AuditStop):native("B22","SITCOM",c,2)
        with self.assertRaises(AuditStop):native("B24","SITCOM",rows("B24","SITCOM"),4)
    def test_argmin_monotone_negative_and_earliest_tie(self):
        c=rows("B22","SITCOM",scores=[-4.0,-2.0,-1.0,9.0])
        self.assertEqual(best(c,"step160_correction_norm",False)["index"],0)
        self.assertEqual(native("B22","SITCOM",c,4)["index"],0)
        c=rows("B22","SITCOM",scores=[4.0,2.0,2.0,1.0])
        self.assertEqual(best(c,"step160_correction_norm",False)["index"],3)
        c=rows("B22","SITCOM",scores=[2.0,-5.0,-5.0,1.0])
        self.assertEqual(best(c,"step160_correction_norm",False)["index"],1)
        c=rows("B22","NP",scores=[20,19,18,17,16,15,14,13])
        self.assertEqual(native("B22","NP",c,8)["candidate_id"],"s2_preproj_lam001-103")
        c=rows("B22","NP",scores=[-2,-2,1,2,3,4,5,6])
        self.assertEqual(native("B22","NP",c,8)["candidate_id"],"lf-100")

    def test_argmax_psnr_independent_of_score(self):
        c=rows("B22","SITCOM",psnrs=[13,99,98,97],
               scores=[0.1,0.9,0.2,0.8])
        self.assertEqual(oracle(c,4)["index"],1)
        self.assertEqual(native("B22","SITCOM",c,4)["index"],0)
    def test_missing_selector_fails_closed(self):
        c=rows("B24","NP")
        c[1].pop("postproj_lf_mse")
        with self.assertRaises(AuditStop):native("B24","NP",c,4)
    def test_source_metric_seed_guards(self):
        c=rows("B24","DAPS")
        c[1]["representation"]="other"
        with self.assertRaises(AuditStop):required_candidates("B24","DAPS",c)
        c=rows("B22","NP")
        c[4]["group"]="lf"
        with self.assertRaises(AuditStop):required_candidates("B22","NP",c)
    def test_oracle_tie_first(self):
        c=rows("B22","DAPS",psnrs=[25.0,25.0])
        self.assertIs(oracle(c,2),c[0])
    def test_exact_mcnemar(self):
        a=[]; b=[]
        for i,(p,q) in enumerate(((26,20),(26,20),(26,20),(20,26),(26,26))):
            c=rows("B22","DAPS")[0]
            c["image_id"]=str(i);c["psnr"]=p
            x={"status":"PASS","cohort":"B22","image_id":str(i),"representation":"raw_float",
               "measurement_sha256":SHA,"K":2,"selected_psnr":p}
            y=dict(x,selected_psnr=q)
            a.append(x);b.append(y)
        result=paired(a,b,reps=100,seed=1)
        self.assertEqual((result["rescues"],result["harms"]),(3,1))
        self.assertEqual(result["mcnemar_exact_p"],0.625)
        b[0]["representation"]="8bit"
        with self.assertRaises(AuditStop):paired(a,b,reps=100)
    def test_b21_b22_alignment_independent_row_orders(self):
        # B21 arm order and B22 filename-sorted manifest row order differ.
        ids=[f"{60000+i:05d}" for i in range(100)]
        b21=[{"image_id":id_, "row_id":str(j),
              "measurement_path":f"/frozen/{id_}.pt"}
             for j,id_ in enumerate(reversed(ids))]
        frozen=[{"image_id":id_, "row_id":j,
                 "measurement_path":f"/frozen/{id_}.pt",
                 "measurement_file_sha256":SHA,
                 "measurement_content_sha256":MODEL}
                for j,id_ in enumerate(ids)]
        manifest={"stage":"full", "selected_row_count":100,"rows":frozen}
        joined=align_frozen_b21_b22(b21,manifest)
        self.assertEqual(len(joined),100)
        self.assertEqual(joined[0][0],0)
        self.assertEqual(joined[0][1]["image_id"],ids[0])
        self.assertEqual(joined[0][1]["row_id"],"99")
        self.assertEqual(joined[0][2]["image_id"],ids[0])
        self.assertEqual(joined[-1][1]["row_id"],"0")
        b21[0]["measurement_path"]="/incorrect/measurement.pt"
        with self.assertRaises(AuditStop):
            align_frozen_b21_b22(b21,manifest)

    def test_b21_b22_alignment_reject_duplicate_or_incomplete_panel(self):
        ids=[f"{60000+i:05d}" for i in range(100)]
        b21=[{"image_id":id_, "row_id":j,
              "measurement_path":f"/frozen/{id_}.pt"}
             for j,id_ in enumerate(ids)]
        frozen=[{"image_id":id_, "row_id":j,
                 "measurement_path":f"/frozen/{id_}.pt",
                 "measurement_file_sha256":SHA,
                 "measurement_content_sha256":MODEL}
                for j,id_ in enumerate(ids)]
        manifest={"stage":"full", "selected_row_count":100,"rows":frozen}
        b21[1]["image_id"]=b21[0]["image_id"]
        with self.assertRaises(AuditStop):align_frozen_b21_b22(b21,manifest)
        b21[1]["image_id"]=ids[1]
        manifest["rows"][0]["measurement_content_sha256"]=""
        with self.assertRaises(AuditStop):align_frozen_b21_b22(b21,manifest)
        manifest["rows"][0]["measurement_content_sha256"]=MODEL
        manifest["rows"][0]["row_id"]=1
        with self.assertRaises(AuditStop):align_frozen_b21_b22(b21,manifest)

    def test_summary_and_provenance(self):
        c=rows("B24","NP")
        results=ladders("B24","NP",c)
        selected=[r for r in results if r["status"]=="PASS" and r["policy"]=="NATIVE" and r["K"]==4]
        self.assertEqual(summarize(selected)["n"],1)
        self.assertTrue(pycache("model/__pycache__/unet.pyc"))
        self.assertEqual(source_only_names(["posterior_sample.py","model/__pycache__/unet.pyc"]),["posterior_sample.py"])

if __name__ == "__main__":
    unittest.main()
