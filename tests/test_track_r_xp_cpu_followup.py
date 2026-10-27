"""Synthetic CPU-only invariants for A8b publication, D1 K4 and C/C2 freeze."""
import unittest
from copy import deepcopy
from collections import Counter
from scripts.track_r.xp_r0n2_a8b_publish_cpu import validate as validate_a8b
from scripts.track_r.xp_d1p1_score_cpu import labels,per_image
from scripts.track_r.xp_c_c2_prepare_cpu import c2_tasks,T,ROT

class HistoricalAndProspectiveTests(unittest.TestCase):
    def test_A8b_unchanged_original_and_tau(self):
        a=[dict(source="S1_b22",family="FAST",image_id=f"{i:05d}",
                n_runs="4",run_labels="GGCI") for i in range(760)]
        b=[dict(x,rho_signed_by_run="0.1;0.12;0.14;0.16",
                tau_signed_good_q95="0.14") for x in a]
        tau=[dict(source="S1_b22",family="FAST",
                  n_candidates="3240",tau_signed_good_q95="0.14")]
        # Deliberate tau group deficiency must fail.
        receipt=dict(status="R0N2_A8B_PASS",n_rows=760,n_candidates=3240,
                     tau_groups=12,A8_original_sha256="197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6",
                     R07_original_sha256="0c01c80acf54c48771904725bdcf0a7c5f0b966ac168b074c5cb8a5690c6ad59",
                     A8B_csv_sha256="e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901",
                     tau_csv_sha256="a86d625ec574614ad42131ae648b84beefe7ce8cb340516ce6a2ae65f20db9f5",
                     historical_only=True,gt_images_read=False,gpu_used=False,
                     new_candidate_outcomes_inspected=False)
        with self.assertRaisesRegex(RuntimeError,"COUNTS"):
            validate_a8b(a,b,tau,receipt)
        # Exercise the receipt violation with an otherwise count-valid
        # 12-group synthetic threshold table.
        grouped=[dict(source=f"S{i}",family="FAST",
                      n_candidates=str(270),
                      tau_signed_good_q95="0.14") for i in range(12)]
        grouped[0]["source"]="S1_b22"
        tampered=deepcopy(receipt);tampered["gt_images_read"]=True
        with self.assertRaisesRegex(RuntimeError,"SOURCE_RECEIPT"):
            validate_a8b(a,b,grouped,tampered)

    def test_D1_Good20_tau_exact_q95_linear_and_order(self):
        rows=[]
        for iid in (f"image_{i:03d}.png" for i in range(100)):
            for j in range(4):
                rows.append(dict(image_id=iid,rep=j,
                  psnr_raw=21.0 if j<2 else 14.0,
                  psnr_orbit_best=22. if j==2 else 16.,
                  s_gt="111" if j==2 else "000",
                  rho_signed=0.02+0.01*j))
        tau=labels(rows)
        self.assertAlmostEqual(tau,0.03)
        self.assertEqual("".join(r["label"] for r in rows[:4]),"GGOI")
        z=per_image(rows,tau)
        self.assertEqual(len(z),100)
        self.assertEqual(z[0]["run_labels"],"GGOI")
        self.assertEqual(z[1]["e0_rank_order"],"0,1,2,3")

    def test_D1_tau_from_Good20_not_best_orbit(self):
        rows=[dict(psnr_raw=10.,psnr_orbit_best=21.,
                   s_gt="111",rho_signed=0.5),
              dict(psnr_raw=21.,psnr_orbit_best=21.,
                   s_gt="000",rho_signed=0.01)]
        self.assertAlmostEqual(labels(rows),0.01)
        self.assertEqual(rows[0]["label"],"O")

    def test_C2_280_x_9_x_2_x_2_exact(self):
        images=[dict(dataset="ffhq",basename=f"id_{i:03d}",source_sha256="a"*64)
                for i in range(180)]
        images += [dict(dataset="imagenet",basename=f"im_{i:03d}",source_sha256="b"*64)
                   for i in range(100)]
        tasks=c2_tasks(images)
        self.assertEqual(len(tasks),10080)
        self.assertEqual(set(T),set(range(100,901,100)))
        self.assertEqual(set(ROT),{"100","111"})
        self.assertEqual(len({z["key"]+"|"+z["rotation"] for z in tasks}),10080)
        # Paired rotated test uses identical noise per t/draw, not reseeding.
        self.assertEqual(tasks[0]["noise_seed"],tasks[2]["noise_seed"])
        self.assertNotEqual(tasks[0]["noise_seed"],tasks[1]["noise_seed"])

if __name__=="__main__":
    unittest.main()
