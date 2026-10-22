"""Paired-noise DDPM denoising-loss orbit scorer. No GT or image records."""
from __future__ import annotations

import hashlib
import math
import torch
from .physics import ORBIT_PATTERNS, apply_orbit


def prior_seed(uid, t, j):
    payload = ("R0PRIOR|%s|%d|%d" % (uid, t, j)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63)-1)


def frozen_alpha_schedule():
    """DDPM 1000-step linear beta 0.0001 to 0.02; double product then float32."""
    return torch.cumprod(torch.linspace(0.0001, 0.02, 1000, dtype=torch.float64).sub(1).neg(), 0).float()


class PriorScorer:
    def __init__(self, eps_model, alphas_cumprod=None,
                 timesteps=(50,100,200,300,400,500,600,700),
                 draws=4, device="cpu", batch_size=8):
        self.eps_model = eps_model
        self.alphas = torch.as_tensor(
            frozen_alpha_schedule() if alphas_cumprod is None else alphas_cumprod,
            dtype=torch.float32, device="cpu")
        self.timesteps = tuple(int(t) for t in timesteps)
        self.draws = int(draws)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if (not self.timesteps or self.draws < 1 or self.batch_size < 1 or
            any(t < 0 or t >= self.alphas.numel() for t in self.timesteps) or
            not torch.isfinite(self.alphas).all() or
            bool((self.alphas <= 0).any()) or bool((self.alphas >= 1).any())):
            raise ValueError("invalid schedule, draws or batch size")

    @torch.no_grad()
    def score_orbit(self, x_c, uid):
        if (not isinstance(x_c,torch.Tensor) or x_c.shape != (1,3,256,256)
            or x_c.dtype != torch.float32 or not bool(torch.isfinite(x_c).all())):
            raise ValueError("expected canonical x_c [1,3,256,256] float32")
        if not isinstance(uid, str) or not uid:
            raise ValueError("nonempty candidate uid required")
        x = x_c.to(self.device)
        members = {p: apply_orbit(x, p) for p in ORBIT_PATTERNS}
        losses = {p: [] for p in ORBIT_PATTERNS}
        by_t = {t: {p: [] for p in ORBIT_PATTERNS} for t in self.timesteps}
        calls = 0
        pending_inputs, pending_eps, pending_meta = [], [], []
        max_batch_observed = 0

        def flush():
            nonlocal calls, max_batch_observed
            if not pending_inputs:
                return
            batch = len(pending_inputs)
            max_batch_observed = max(max_batch_observed, batch)
            xt = torch.cat(pending_inputs, dim=0)
            target = torch.cat(pending_eps, dim=0)
            tb = torch.tensor([v[0] for v in pending_meta],
                              dtype=torch.long, device=self.device)
            predicted = self.eps_model(xt, tb)
            if hasattr(predicted, "sample"):
                predicted = predicted.sample
            if isinstance(predicted, tuple):
                predicted = predicted[0]
            if (not isinstance(predicted, torch.Tensor) or
                    predicted.ndim != 4 or
                    tuple(predicted.shape[:2]) not in
                    ((batch, 3), (batch, 6)) or
                    tuple(predicted.shape[-2:]) != (256, 256)):
                raise ValueError("eps_model invalid output")
            eps = predicted[:, :3].float()
            vals = (eps-target).float().square().flatten(1).mean(1).cpu().tolist()
            for (t, j, p), v in zip(pending_meta, vals):
                if not math.isfinite(v):
                    raise RuntimeError("nonfinite prior loss")
                losses[p].append(float(v))
                by_t[t][p].append(float(v))
            calls += 1
            pending_inputs.clear()
            pending_eps.clear()
            pending_meta.clear()

        # Batch across (t, j, orbit) rather than just the eight orbit
        # members. Thus batch_size=8 and 64 are genuinely different
        # model-forward workloads for the frozen R0.3 determinism gate.
        # CPU random draws remain keyed solely by (uid, t, j), not batch.
        with torch.inference_mode():
            for t in self.timesteps:
                alpha = float(self.alphas[t])
                for j in range(self.draws):
                    rng = torch.Generator(device="cpu").manual_seed(
                        prior_seed(uid,t,j))
                    noise = torch.randn(
                        (1,3,256,256), generator=rng,
                        dtype=torch.float32, device="cpu").to(self.device)
                    for p in ORBIT_PATTERNS:
                        rotated_noise = apply_orbit(noise, p)
                        pending_inputs.append(
                            math.sqrt(alpha)*members[p] +
                            math.sqrt(1-alpha)*rotated_noise)
                        pending_eps.append(rotated_noise)
                        pending_meta.append((t,j,p))
                        if len(pending_meta) >= self.batch_size:
                            flush()
            flush()
        per_t = {p: {str(t): sum(by_t[t][p])/len(by_t[t][p])
                     for t in self.timesteps} for p in ORBIT_PATTERNS}
        scores = {p: sum(losses[p])/len(losses[p]) for p in ORBIT_PATTERNS}
        mid = {p: sum(per_t[p][str(t)] for t in self.timesteps if t in (300,400,500,600)) /
                    sum(t in (300,400,500,600) for t in self.timesteps)
               for p in ORBIT_PATTERNS} if any(t in (300,400,500,600) for t in self.timesteps) else None
        best = min(ORBIT_PATTERNS, key=lambda p: scores[p])
        if scores["000"] - scores[best] <= 1e-7:
            best = "000"
        differences = [a-b for a,b in zip(losses["000"],losses[best])]
        n = len(differences)
        margin = sum(differences)/n
        sd = math.sqrt(sum((d-margin)**2 for d in differences)/(n-1)) if n>1 else 0.0
        z = margin/(sd/math.sqrt(n)) if sd>0 else (0.0 if abs(margin)<1e-15 else float("inf"))
        return {"uid":uid, "S":scores, "S_mid":mid,
                "per_t_means":per_t, "s_star":best,
                "margin":margin, "z":z,
                "unet_forwards":len(ORBIT_PATTERNS)*len(self.timesteps)*self.draws,
                "unet_batch_calls":calls, "max_batch_observed":max_batch_observed,
                "draw_losses":losses,
                "noise_device":"cpu"}
