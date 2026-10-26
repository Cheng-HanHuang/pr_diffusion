"""Synthetic frozen XP-T SITCOM shared-y, seed, provenance checks."""
from __future__ import annotations
import hashlib
import unittest
from scripts.track_r.xp_t_sitcom_prepare_cpu import freeze_rows,seed

def fixture():
    ff={str(i):("/research/y/ff_"+str(i)+".pt",f"{i:064x}")
        for i in (100,200,300)}
    names=[f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(100)]
    common=[{"basename":name,
             "measurement_path":f"/research/y/im_{name}.pt",
             "measurement_sha256":hashlib.sha256(name.encode()).hexdigest()}
             for name in names]
    inn={r["basename"]:(r["measurement_path"],r["measurement_sha256"])
         for r in common[:3]}
    return ff,inn,common

class SITCOMTimingPrepare(unittest.TestCase):
    def test_two_datasets_exact_18_roots_shared_y(self):
        ff,im,common=fixture()
        roots=freeze_rows(ff,im,common)
        self.assertEqual(len(roots),18)
        self.assertEqual({r["dataset"] for r in roots},{"ffhq","imagenet"})
        self.assertEqual([r["rep"] for r in roots],[0,1,2]*6)
        self.assertEqual(len({r["uid"] for r in roots}),18)
        self.assertEqual([r["seed"] for r in roots[:3]],
                         [seed("ffhq","100",i) for i in range(3)])
        for i in range(0,18,3):
            block=roots[i:i+3]
            self.assertEqual(len({r["measurement_sha256"] for r in block}),1)
            self.assertEqual(len({r["measurement_path"] for r in block}),1)
            self.assertEqual(len({r["seed"] for r in block}),3)
    def test_seeds_deterministic_distinct_dataset(self):
        self.assertEqual(seed("ffhq","60044",0),seed("ffhq","60044",0))
        self.assertNotEqual(seed("ffhq","60044",0),seed("imagenet","60044",0))
        self.assertNotEqual(seed("ffhq","60044",0),seed("ffhq","60044",1))
    def test_imagenet_common_y_mismatch_rejected(self):
        ff,im,common=fixture()
        wrong=dict(im)
        name=next(iter(im))
        wrong[name]=(wrong[name][0],"0"*64)
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_SITCOM_IMAGENET_COMMON_Y"):
            freeze_rows(ff,wrong,common)
    def test_imagenet_order_and_missing_rows_rejected(self):
        ff,im,common=fixture()
        bad=list(reversed(common))
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_SITCOM_IMAGENET_SORT_ORDER"):
            freeze_rows(ff,im,bad)
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_SITCOM_IMAGENET100"):
            freeze_rows(ff,im,common[:-1])

if __name__=="__main__":
    unittest.main()
