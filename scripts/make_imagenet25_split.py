from pathlib import Path
import random
from PIL import Image

IMG_DIR = Path("/egr/research-pac/huang248/data/imagenet/imagenet256_val")
OUT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/imagenet_available25.txt")

N = 25
SEED = 20260430

paths = sorted(IMG_DIR.glob("*.JPEG"))

if len(paths) < N:
    raise RuntimeError(f"Only found {len(paths)} images in {IMG_DIR}, cannot sample {N}")

# Keep only images that PIL can open and that are exactly 256 x 256.
valid = []

for p in paths:
    try:
        with Image.open(p) as img:
            if img.size == (256, 256):
                valid.append(p)
    except Exception:
        pass

print(f"Found {len(paths)} total images")
print(f"Found {len(valid)} valid 256x256 images")

if len(valid) < N:
    raise RuntimeError(f"Only found {len(valid)} valid images, cannot sample {N}")

random.seed(SEED)
chosen = sorted(random.sample(valid, N))

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w") as f:
    for p in chosen:
        f.write(str(p.resolve()) + "\n")

print(f"Wrote {N} image paths to:")
print(OUT)
print()
print("First few:")
for p in chosen[:5]:
    print(p)
