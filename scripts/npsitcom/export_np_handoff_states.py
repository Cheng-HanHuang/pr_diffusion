#!/usr/bin/env python3
"""Branch B: export timestep-compatible NP handoff states.

This does not call SITCOM. It creates .pt files containing x_t states generated
from NP-selected x0 candidates:

  x_t = sqrt(alpha_bar_t) x_np + sqrt(1-alpha_bar_t) eps

The resulting manifest can be consumed by a SITCOM wrapper once the local
SITCOM_ODE entrypoint supports init-state/start-timestep arguments.
"""
from __future__ import annotations

import argparse, csv, importlib.util, math, os, sys, time
from pathlib import Path
from typing import Dict, List

import torch

ROOT=Path(__file__).resolve().parents[2]
SEL=ROOT/'scripts'/'pr_external_difffpr_np_guided_lf_s2_selector.py'

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    assert spec and spec.loader; spec.loader.exec_module(mod); return mod

selector=load_module('npsitcom_selector', SEL)
base=selector.base
from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image


def write_csv(path: str, rows: List[Dict[str,object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows: raise ValueError('no rows')
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def parse_ints(s): return [int(x.strip()) for x in str(s).split(',') if x.strip()]
def parse_floats(s): return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--image_list_file', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--guided_model_path', required=True)
    ap.add_argument('--guided_diffusion_dir', default=None)
    ap.add_argument('--guided_preset', default='difffpr_ffhq_10m')
    ap.add_argument('--seeds', default='100,101,102,103')
    ap.add_argument('--noise_values', default='0.05')
    ap.add_argument('--max_images', type=int, default=5)
    ap.add_argument('--np_steps', type=int, default=1000)
    ap.add_argument('--late_start', type=int, default=300)
    ap.add_argument('--soft_candidates', type=int, default=5)
    ap.add_argument('--hard_candidates', type=int, default=1)
    ap.add_argument('--score_radius', type=float, default=0.6)
    ap.add_argument('--proj_radius', type=float, default=0.2)
    ap.add_argument('--s2_lambda', type=float, default=0.01)
    ap.add_argument('--s2_lambda_schedule', default='pre_projection_only')
    ap.add_argument('--score_huber_delta', type=float, default=0.05)
    ap.add_argument('--oversample', type=float, default=2.0)
    ap.add_argument('--handoff_timesteps', default='700,500,300,100')
    ap.add_argument('--measurement_noise_seed', type=int, default=20260423)
    ap.add_argument('--clip_noisy_magnitude', action='store_true')
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    state_dir=os.path.join(args.outdir,'states'); os.makedirs(state_dir, exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle=load_guided_diffusion_model(args.guided_model_path, device=device, preset=args.guided_preset, guided_diffusion_dir=args.guided_diffusion_dir)
    image_size=int(bundle.unet.config.sample_size)
    pad=base.oversample_pad(image_size,args.oversample)
    images=base.collect_images(args.data_root,args.image_list_file)[:args.max_images]
    seeds=parse_ints(args.seeds); noises=parse_floats(args.noise_values); handoff_ts=parse_ints(args.handoff_timesteps)
    variant=base.NPVariant(name=f'np_soft{args.soft_candidates}_hard{args.hard_candidates}', soft=args.soft_candidates, hard=args.hard_candidates, proj_start=args.late_start, use_lowfreq_score=True, use_lowfreq_projection=True)
    configs=[('lf','lf',0.0,'constant'),('s2_preproj','prev_l2',args.s2_lambda,args.s2_lambda_schedule)]
    rows=[]
    scheduler=bundle.scheduler
    for noise_std in noises:
      for image_i,name in enumerate(images):
        x_gt=load_image(base.resolve_image_path(args.data_root,name), size=image_size, device=device)
        mag_clean=base.oversampled_magnitude(x_gt,pad); mag_target=mag_clean
        if noise_std>0:
            gen=torch.Generator(device=device).manual_seed(args.measurement_noise_seed+image_i)
            mag_target=mag_clean+noise_std*torch.randn(mag_clean.shape,device=device,dtype=mag_clean.dtype,generator=gen)
            if args.clip_noisy_magnitude: mag_target=mag_target.clamp_min(0.0)
        for cfg_tag,score_mode,lam,sched in configs:
          for seed in seeds:
            x_np,stats=selector.reconstruct_with_selector_stat(mag_target,pad=pad,seed=seed,unet=bundle.unet,scheduler=scheduler,device=device,variant=variant,num_steps=args.np_steps,score_radius=args.score_radius,proj_radius=args.proj_radius,proj_radius_schedule=None,score_mode=score_mode,score_reg_lambda=lam,score_reg_lambda_schedule=sched,score_huber_delta=args.score_huber_delta,log_every=0)
            for t in handoff_ts:
              alpha=scheduler.alphas_cumprod[int(t)].to(device=device,dtype=x_np.dtype)
              gen=torch.Generator(device=device).manual_seed(900000+seed*1000+int(t)+image_i)
              eps=torch.randn(x_np.shape,device=device,dtype=x_np.dtype,generator=gen)
              x_t=torch.sqrt(alpha)*x_np + torch.sqrt(1-alpha)*eps
              fn=f'{stamp}_{name}_noise{noise_std:g}_{cfg_tag}_seed{seed}_t{t}.pt'.replace('/','_')
              path=os.path.join(state_dir,fn)
              torch.save({'x_t':x_t.cpu(),'x0_np':x_np.cpu(),'timestep':int(t),'image_basename':name,'noise_std':float(noise_std),'seed':int(seed),'config_tag':cfg_tag,'selector_stats':stats}, path)
              row=dict(state_path=path,image_basename=name,measurement_noise_std=noise_std,seed=seed,config_tag=cfg_tag,handoff_timestep=t,np_steps=args.np_steps,score_radius=args.score_radius,proj_radius=args.proj_radius,**stats)
              rows.append(row)
              print('[handoff]', name, noise_std, cfg_tag, seed, 't', t, path, flush=True)
    write_csv(os.path.join(args.outdir,'handoff_manifest.csv'), rows)
    print('wrote', os.path.join(args.outdir,'handoff_manifest.csv'))

if __name__=='__main__':
    main()
