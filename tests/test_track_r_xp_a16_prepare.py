"""Synthetic CPU-only verification of XP-A16 frozen historical extension."""
import unittest
from unittest.mock import patch
from scripts.track_r.xp_a16_prepare_cpu import build
from scripts.track_r.r0_5_fast_prepare_cpu import native_seed

def fake():
    rows=[];catalog=[];counter=0
    for cohort,n in (("S1_b22",100),("S2_b24_dev80",80)):
        for k in range(n):
            iid=f"{k:05d}";path=f"/tmp/y_{cohort}_{iid}.pt"
            digest=f"{counter:064x}";counter+=1
            for rep in range(4):
                rows.append(dict(source="S5_r0_fast",
                    measurement_source=cohort,image_id=iid,rep=rep,
                    uid=f"S5_r0_fast|FAST|{cohort}_{iid}|rep{rep}",
                    seed=native_seed(cohort,iid,rep),
                    measurement_path=path,measurement_sha256=digest))
                catalog.append(dict(source="S5_r0_fast",
                    measurement_source=cohort,image_id=iid,
                    measurement_path=path,measurement_sha256=digest))
    return rows,catalog

class ExpandK16(unittest.TestCase):
    def test_frozen_180_extends_to_2160_with_disjoint_seeds(self):
        old,cat=fake()
        with (patch("scripts.track_r.xp_a16_prepare_cpu.frozen_roots",
                    return_value=old),
              patch("scripts.track_r.xp_a16_prepare_cpu.sha",
                    side_effect=lambda x:next(z["measurement_sha256"] for z in cat
                                              if z["measurement_path"]==x))):
            new=build({},cat)
        self.assertEqual(len(new),2160)
        self.assertEqual(len({x["uid"] for x in new}),2160)
        self.assertEqual({x["rep"] for x in new},set(range(4,16)))
        self.assertEqual(sum(x["measurement_source"]=="S1_b22" for x in new),1200)
        self.assertEqual(sum(x["measurement_source"]=="S2_b24_dev80" for x in new),960)
        for x in new:
            self.assertEqual(x["seed"],native_seed(
                x["measurement_source"],x["image_id"],x["rep"]))
    def test_historical_seed_drift_stops(self):
        old,cat=fake()
        old[5]["seed"]+=17
        with (patch("scripts.track_r.xp_a16_prepare_cpu.frozen_roots",
                    return_value=old),
              patch("scripts.track_r.xp_a16_prepare_cpu.sha",return_value="")):
            with self.assertRaises(RuntimeError):
                build({},cat)
    def test_conflicting_historical_measurement_stops(self):
        old,cat=fake()
        old[0]["measurement_sha256"]="b"*64
        with patch("scripts.track_r.xp_a16_prepare_cpu.frozen_roots",
                   return_value=old):
            with self.assertRaisesRegex(RuntimeError,"Y_IDENTITY"):
                build({},cat)
    def test_179_ids_not_silently_accepted(self):
        old,cat=fake()
        old=[x for x in old if not (x["measurement_source"]=="S2_b24_dev80"
                                    and x["image_id"]=="00079")]
        with patch("scripts.track_r.xp_a16_prepare_cpu.frozen_roots",
                   return_value=old):
            with self.assertRaisesRegex(RuntimeError,"180_IDS"):
                build({},cat)

if __name__=="__main__":unittest.main()
