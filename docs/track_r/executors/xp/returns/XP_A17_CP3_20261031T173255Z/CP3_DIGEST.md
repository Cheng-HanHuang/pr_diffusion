# CP3 — frozen protocol and A15 XP-I descriptive checkpoint
Planner prereg A15 @ 817f78a; independently scored and audited; no protocol v2.
CP2 ACCEPTED: N32 H1 FAIL -0.0179, H3 FAIL (1/3), H4 PASS +0.131.
CP2 ACCEPTED: N64 H1 FAIL -0.0344, H3 N/A (one flag), H4 PASS +0.136.
CP2 v4: race selection 18/18 vs S2 1/18; orbit oracle 0/6.
G2 sigma .02 H-G1 FAIL (9.89% tau mismatch); H-G2 PASS, 8.0/8 runs (no savings).
G2 sigma .10 H-G1 FAIL (2.12% tau mismatch); H-G2 PASS.
G4 FFHQ DAPS4K H1 PASS +3.93pp; success 0.9396, mean runs 2.803.
G5 ImageNet DAPS4K H1 PASS only +0.10pp; success 0.6389, mean runs 2.404.
XP-I A12 matched-NFE rule WITHDRAWN, UNRESOLVED (planner acknowledged design error).
XP-I E1 minus native PSNR: +1.026 dB, paired bootstrap 95% [+0.540,+1.605].
XP-I E1 minus native SSIM: +0.0254, paired 95% [+0.0132,+0.0405].
XP-I E1 minus native GOOD25: +5.42pp, paired 95% [+2.78,+8.47].
XP-I: PC bag E1 vs DAPS4K run0, fixed E1 K1/K2 and Eprime8: compute-labeled table (K2/Eprime8 clock is extrapolated).
DAPS4K: 400 annealing x 10 ODE = 4000 nominal NFE/run, 100 Langevin/anneal (native receipts).
PC-DAPS: 1920 nominal NFE/bag, logged mean lower bound 1928.4; exact pruning NFE unknown.
Clean batch1 native sampling seconds, same GPU/3 images: PC bag 30.58; DAPS4K run 65.98.
A8.6: ImageNet GT/y all 100 exact; per-metric GT best-of4/8 reference table in A8_6.json.
Strings/CLIP decisions for all five panels mirrored to theory_inputs; path/GT tensors excluded.
No matched-compute claim, no frozen protocol changes. CP4 by Nov 15; tail I2 before K.
