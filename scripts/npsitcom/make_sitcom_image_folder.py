#!/usr/bin/env python3
"""Create a SITCOM-compatible image folder from an NP split file.

SITCOM_ODE's `ImageDataset` reads all images recursively under `data.root`,
sorts them, then slices by `data.start_id:data.end_id`.  This helper builds a
flat folder of symlinks/copies whose sorted order follows our split file.
"""
from __future__ import annotations

import argparse, os, shutil
from pathlib import Path
from typing import Dict, List

EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')


def read_split(path: str) -> List[str]:
    out=[]
    for line in Path(path).read_text().splitlines():
        s=line.strip()
        if s and not s.startswith('#'):
            out.append(s)
    return out


def index_images(root: str) -> Dict[str, Path]:
    idx={}
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix in EXTS:
            idx.setdefault(p.name, p)
            idx.setdefault(p.stem, p)
    return idx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--copy', action='store_true', help='copy instead of symlink')
    args=ap.parse_args()
    names=read_split(args.split_file)
    idx=index_images(args.data_root)
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    manifest=[]
    for i,name in enumerate(names):
        key=Path(name).name
        src=idx.get(key) or idx.get(Path(key).stem)
        if src is None:
            raise FileNotFoundError(f'Could not resolve split entry {name!r} under {args.data_root}')
        dst=out / f'{i:05d}_{src.name}'
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        manifest.append(f'{i},{name},{src},{dst}')
    (out/'manifest.csv').write_text('index,split_entry,source_path,sitcom_path\n'+'\n'.join(manifest)+'\n')
    print(f'wrote {len(names)} images to {out}')

if __name__ == '__main__':
    main()
