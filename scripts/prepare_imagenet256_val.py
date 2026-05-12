from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

SRC = Path("/egr/research-pac/huang248/data/imagenet/val_raw")
DST = Path("/egr/research-pac/huang248/data/imagenet/imagenet256_val")
DST.mkdir(parents=True, exist_ok=True)

SIZE = 256

def resize_center_crop(img: Image.Image, size: int = 256) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")

    w, h = img.size
    scale = size / min(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2

    return img.crop((left, top, left + size, top + size))

paths = sorted(SRC.glob("*.JPEG"))
print(f"Found {len(paths)} raw validation images in {SRC}")

bad = []

for p in tqdm(paths):
    out = DST / p.name
    if out.exists():
        continue

    try:
        img = Image.open(p)
        img = resize_center_crop(img, SIZE)
        img.save(out, quality=95)
    except Exception as e:
        bad.append((str(p), str(e)))
        print(f"[WARN] failed: {p} :: {e}")

out_count = len(list(DST.glob("*.JPEG")))

print(f"Saved resized images to {DST}")
print(f"Output count: {out_count}")

if bad:
    print(f"Bad images: {len(bad)}")
    for p, e in bad[:20]:
        print(p, e)

