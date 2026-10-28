"""A9 FAST post-hoc 11-case census synthetic tests."""
import unittest

from scripts.track_r.xp_a9_cases_cpu import case_row,census

class TestA9Cases(unittest.TestCase):
    @staticmethod
    def row(i,cohort="B22",kind="normal"):
        labels=["I"]*64
        if kind=="normal":labels[0]="G"
        if kind=="loss":labels[2]="G"
        rho=[.05+.001*j for j in range(64)]
        if kind=="loss":rho[2]=.09
        from collections import Counter
        cnt=Counter(labels)
        return {
            "cohort":cohort,"image_id":str(i),
            "labels_by_run":";".join(labels),
            "rho_signed_by_run":";".join(map(str,rho)),
            "e0_rank_order":";".join(str(j) for j in sorted(range(64),
                key=lambda j:(rho[j],j))),
            **{"n_"+k:str(cnt[k]) for k in ("G","O_g","O_p","C","I")}
        }
    def test_single_cases(self):
        self.assertEqual(case_row(self.row(1))["category"],"OTHER")
        self.assertEqual(case_row(self.row(1,kind="never"))["category"],"NEVER_GOOD")
        loss=case_row(self.row(1,kind="loss"))
        self.assertEqual(loss["category"],"K64_SELECTION_LOSS")
        self.assertLess(loss["winner_minus_best_good_rho_over_sigma"],0)
    def test_full_census_cohorts(self):
        rows=[self.row(i) for i in range(100)]
        rows.extend(self.row(i,"DEV80") for i in range(80))
        rows[0]=self.row(0,kind="never")
        for j in range(1,7):
            rows[100+j]=self.row(j,"DEV80","never")
        for j in range(7,11):
            rows[100+j]=self.row(j,"DEV80","loss")
        result=census(rows)
        self.assertEqual(len(result),11)
        self.assertEqual(sum(r["category"]=="NEVER_GOOD" for r in result),7)
        self.assertEqual(sum(r["category"]=="K64_SELECTION_LOSS" for r in result),4)

if __name__=="__main__":
    unittest.main()
