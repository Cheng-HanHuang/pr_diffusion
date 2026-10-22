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
        members = [apply_orbit(x, p) for p in ORBIT_PATTERNS]
        losses = {p: [] for p in ORBIT_PATTERNS}
        per_t = {p: {} for p in ORBIT_PATTERNS}
        calls = 0
        with torch.inference_mode():
            for t in self.timesteps:
                alpha = float(self.alphas[t])
                by_t = {p: [] for p in ORBIT_PATTERNS}
                for j in range(self.draws):
                    rng = torch.Generator(device="cpu").manual_seed(prior_seed(uid,t,j))
                    noise = torch.randn((1,3,256,256), generator=rng,
                                        dtype=torch.float32, device="cpu").to(self.device)
                    for start in range(0, len(ORBIT_PATTERNS), self.batch_size):
                        patterns = ORBIT_PATTERNS[start:start+self.batch_size]
                        rotated_noise = [apply_orbit(noise,p) for p in patterns]
                        z = torch.cat([math.sqrt(alpha)*members[ORBIT_PATTERNS.index(p)] +
                                       math.sqrt(1-alpha)*n
                                       for p,n in zip(patterns,rotated_noise)], 0)
                        tb = torch.full((len(patterns),), t, device=self.device,
                                        dtype=torch.long)
                        predicted = self.eps_model(z,tb)
                        if isinstance(predicted, tuple):
                            predicted = predicted[0]
                        if not isinstance(predicted,torch.Tensor) or predicted.shape[0]!=len(patterns) or predicted.ndim!=4 or predicted.shape[1]<3:
                            raise ValueError("eps_model invalid output")
                        eps = predicted[:,:3].float()
                        target = torch.cat(rotated_noise,dim=0)
                        vals = (eps-target).float().square().flatten(1).mean(1).cpu().tolist()
                        for p,v in zip(patterns,vals):
                            if not math.isfinite(v):
                                raise RuntimeError("nonfinite prior loss")
                            losses[p].append(float(v))
                            by_t[p].append(float(v))
                        calls += 1
                for p in ORBIT_PATTERNS:
                    per_t[p][str(t)] = sum(by_t[p])/len(by_t[p])
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
                "unet_batch_calls":calls, "draw_losses":losses,
                "noise_device":"cpu"}
