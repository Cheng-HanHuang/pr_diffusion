#!/usr/bin/env python3
"""CPU-only import of the official DAPS 100-image ImageNet test archive.

Never open other dataset bundles or walk outside the passed ZIP and new target.
Only image-file members of the approved ImageNet archive can be extracted.
Inputs/outputs remain under PAC root; source archive is immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

ROOT = Path("/egr/research-pac/huang248")
OUT = ROOT/"outputs/pr_diffusion/track_r"
TARGET = ROOT/"data/imagenet/test-imagenet"
OFFICIAL_ID = "1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs"
BAD = ("confirmation", "confirm305", "test-ffhq", "demo-ffhq")
EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_IMAGE_BYTES = 150_000_000
MAX_TOTAL_BYTES = 2_000_000_000

def guard(path, file=False):
    p = Path(path)
    if any(t in str(p).lower() for t in BAD):
        raise RuntimeError("STOP_FORBIDDEN_SOURCE")
    actual = p.resolve(strict=False)
    if any(t in str(actual).lower() for t in BAD):
        raise RuntimeError("STOP_FORBIDDEN_RESOLVED_SOURCE")
    root = ROOT.resolve(strict=False)
    if actual != root and root not in actual.parents:
        raise RuntimeError("STOP_SOURCE_OUTSIDE_PAC")
    if file and not p.is_file():
        raise RuntimeError("STOP_MISSING_INPUT_ARCHIVE")
    return p

def digest(path):
    h = hashlib.sha256()
    with guard(path, True).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024), b""):
            h.update(part)
    return h.hexdigest()

def image_members(z):
    chosen = []
    total = 0
    names = set()
    for info in z.infolist():
        raw = info.filename.replace("\\", "/")
        if any(t in raw.lower() for t in BAD):
            raise RuntimeError("STOP_ARCHIVE_FORBIDDEN_MEMBER")
        path = PurePosixPath(raw)
        parts = path.parts
        if (path.is_absolute() or ".." in parts or
            any(x in ("", ".") for x in parts)):
            raise RuntimeError("STOP_ARCHIVE_TRAVERSAL")
        mode = (info.external_attr >> 16)
        if stat.S_ISLNK(mode):
            raise RuntimeError("STOP_ARCHIVE_SYMLINK")
        if info.is_dir():
            continue
        if info.flag_bits & 1:
            raise RuntimeError("STOP_ARCHIVE_ENCRYPTED")
        if info.file_size > MAX_IMAGE_BYTES:
            raise RuntimeError("STOP_ARCHIVE_OVERSIZED_ENTRY")
        # The archive itself is ImageNet-specific. Never interpret a separate
        # FFHQ archive or permit another dataset in this importer.
        if path.suffix.lower() not in EXTENSIONS:
            if path.name.startswith(".") or path.name in ("README","LICENSE"):
                continue
            raise RuntimeError("STOP_ARCHIVE_UNEXPECTED_FILE "+path.name)
        if len(parts)>1 and not any("imagenet" in x.lower() for x in parts[:-1]):
            raise RuntimeError("STOP_ARCHIVE_UNEXPECTED_DATASET_ROOT")
        name = path.name
        if name in names:
            raise RuntimeError("STOP_ARCHIVE_DUPLICATE_BASENAME")
        names.add(name)
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise RuntimeError("STOP_ARCHIVE_TOTAL_BYTES")
        chosen.append((info,name))
    chosen.sort(key=lambda pair:pair[1])
    if len(chosen)!=100:
        raise RuntimeError("STOP_ARCHIVE_EXPECT_100_IMAGENET_GOT_"+str(len(chosen)))
    return chosen

def main(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_CPU_ONLY")
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                   text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],
                                    text=True, timeout=30).strip()
    if head!=args.expected_head or dirty:
        raise RuntimeError("STOP_GIT_HEAD_OR_DIRTY")
    archive=guard(args.archive, True)
    if archive.suffix.lower()!=".zip":
        raise RuntimeError("STOP_NOT_ZIP")
    destination=guard(TARGET)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("STOP_IMAGENET_ALREADY_EXISTS_REINVENTORY_INSTEAD")
    stage=guard(args.out)
    if not str(stage).startswith(str(OUT/"xp_0_imagenet_")) or stage.exists():
        raise RuntimeError("STOP_STAGE_OUTPUT")
    stage.mkdir(parents=True, exist_ok=False)
    temporary=guard(destination.parent/(".xp_imagenet_new_"+stage.name))
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("STOP_STAGING_COLLISION")
    with zipfile.ZipFile(archive) as z:
        # Validate ALL member names before decompressing a single ZIP member.
        # ZipFile.open checks CRC of each approved image as it is read.
        entries=image_members(z)
        temporary.mkdir(parents=True,exist_ok=False)
        manifest=[]
        from PIL import Image
        for i,(entry,name) in enumerate(entries):
            dst=temporary/name
            with z.open(entry,"r") as src, dst.open("xb") as sink:
                h=hashlib.sha256()
                while True:
                    blob=src.read(1024*1024)
                    if not blob: break
                    h.update(blob)
                    sink.write(blob)
            with Image.open(dst) as im:
                im.verify()
            manifest.append({"index":i,"basename":name,"sha256":h.hexdigest(),
                             "bytes":entry.file_size})
    if len(list(temporary.iterdir()))!=100:
        raise RuntimeError("STOP_STAGE_IMAGE_COUNT")
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary.rename(destination)
    payload={"status":"XP_IMAGENET100_IMPORTED_PASS","source":"DAPS README test-imagenet",
             "google_drive_file_id":OFFICIAL_ID,
             "archive_sha256":digest(archive),"image_count":100,
             "head":head,"destination":str(destination),
             "images":manifest}
    (stage/"XP_IMAGENET100_IMPORT_RECEIPT.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("XP_IMAGENET100_IMPORTED_PASS n=100 archive_sha256="+payload["archive_sha256"])
    print("IMAGE_MANIFEST_SHA256="+digest(stage/"XP_IMAGENET100_IMPORT_RECEIPT.json"))
    print("OK")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--expected-head",required=True)
    try:main(parser.parse_args())
    except Exception as exc:
        print("ACTION NEEDED: XP_IMAGENET_IMPORT "+str(exc),flush=True)
        raise SystemExit(8)
