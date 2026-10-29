# A11 first held-out D1-P2 protocol-v1 result

Frozen protocol: c598176. No refitting.

100 ImageNet images, 16 FAST runs per image; 1,600 G1/G2
per-candidate decisions. Primary M=2, N_MAX=16.

Primary results:
- Success: 0.514455
- Mean runs: 3.3592625
- Gain versus fixed equal-compute E1: +0.0033265975
- FLAG_C expected count: 3.0
- FLAG_C precision for unsolved images: 1/3
- Success unflagged: 0.6493644
- Success given not FLAG_C: 0.5200567

Registered hypotheses: H1 PASS, H3 FAIL, H4 PASS.

All six unmodified protocol configurations and their receipt
are included. The 16-run strings and 1,600 G1/G2 decisions
are also copied to theory_inputs for v4 fitting.
