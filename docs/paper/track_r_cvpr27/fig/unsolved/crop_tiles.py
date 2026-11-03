"""Crop the three 256-px tiles out of each Nov 4 panel PNG (XP 953e2f0). Lossless; no resampling.
Boxes (left, top, right, bottom): reference (10,26,266,282), lowest residual (276,26,532,282), best orbit-oracle run (542,26,798,282).
Usage: python3 -I crop_tiles.py <dir with the 15 PNGs> <out dir>"""
import sys, os, glob
from PIL import Image
src, dst = sys.argv[1:3]
B = {"ref": (10, 26, 266, 282), "e0": (276, 26, 532, 282), "orc": (542, 26, 798, 282)}
for f in sorted(glob.glob(os.path.join(src, "*.png"))):
    im = Image.open(f).convert("RGB"); b = os.path.basename(f)[:-4]
    for k, box in B.items():
        im.crop(box).save(os.path.join(dst, f"{b}_{k}.png"))
