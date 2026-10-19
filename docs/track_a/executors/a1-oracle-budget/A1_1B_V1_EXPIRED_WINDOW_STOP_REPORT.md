# A1.1b V1 GPU attempt — verified controlled STOP: expired date-gated reservation

**Date:** 2026-10-19 ~19:13 America/Detroit. **Executor:** continuing Track A executor. **Status:** `A1_1B_V1_CONTROLLED_STOP_EXPIRED_WINDOW`, 0 native trajectories; V1 receipt preserved.

User ran `scripts/track_a/a1_1b_run_gpu_smoke.sh` on pinned A1 code `f745898c4073466af8ca41d58b1979017d93fea3`. Reported `A1_1B_NATIVE_RC=4`, `A1_1B_GPU_FINAL_STATUS=3`, runner RC=1. User uploaded ZIP `A1_1B_GPU_NATIVE_RETURN_V1.zip`; independently verified exact ZIP SHA-256 **`ea631b9b581adc07b86511197574f247262ccf6dc62931c84f4574fa3b681672`** and ZIP integrity (`ZipFile.testzip()=None`).

The single archive member is **`A1_1B_GPU_NATIVE_ONEEACH_V1.log`** (109 bytes; SHA-256 `787d56cbe718eabb9c3a60fa75eddb84655dc7bb8758ecba3abd9e1874f148ac`) containing exactly:

```
A1_1B_STOP: RuntimeError: GPU start outside the conditional Oct 19 12:30-16:00 EDT window; renew reservation
```

Cause is expired **dated** run authorization in Python `run_start_in_approved_window`, fixed to 2026-10-19 12:30–16:00 EDT; user's evening attempt was later. This STOP happened after source, probe and model SHA precheck but **before** any device availability probe, before `OUT.mkdir`, before importing B24 helper, before DAPS/SITCOM/NP subprocess launch. Therefore the controller itself performed **zero GPU inference/measurement** and generated no candidates. No FFHQ, B24 confirmation305 or real-image corpus touched. From the ZIP alone we cannot prove global PAC process activity; `no GPU work` applies to this bounded Track A controller.

The V1 PAC output log/ZIP must remain immutable. We will use the *new* `a1_1b_one_each_v2` directory and `A1_1B_GPU_NATIVE_RETURN_V2.zip` on a new source commit with a **new** conditional evening reservation. The new attempt is an engineering smoke (one native DAPS, one SITCOM, one NP LF, one NP S2 on the same procedural observation), not a rerun of scientifically completed GPU data. New GPU scheduling still respects Track B priority, physical GPU3 UUID, free-memory and watchdog gates.

The original A1.1b CPU plan PASS (13/13) and A1.1a procedural measurement remain valid and are not rerun or modified. It would be incorrect to call this a DAPS/NP failure or a GPU resource failure; it is a scheduling-expiration STOP.
