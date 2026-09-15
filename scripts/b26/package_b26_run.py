#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,tarfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args(); run=a.run.resolve()
 comp=run/'B26_COMPLETE.json'
 if not comp.is_file() or json.load(open(comp)).get('status')!='PASS': raise RuntimeError('B26_COMPLETE PASS required')
 archive=run.with_suffix('.tar.gz'); side=Path(str(archive)+'.sha256')
 if archive.exists() or side.exists(): raise FileExistsError(archive)
 files=[]
 for root,dirs,names in os.walk(run,followlinks=False):
  rootp=Path(root)
  for d in dirs:
   p=rootp/d
   if p.is_symlink(): raise RuntimeError(f'symlink directory rejected: {p}')
  for n in names:
   p=rootp/n
   if p.is_symlink(): raise RuntimeError(f'symlink file rejected: {p}')
   if p.name=='SHA256SUMS.txt': continue
   files.append(p)
 files=sorted(files)
 lines=[f"{sha(p)}  {p.relative_to(run).as_posix()}" for p in files]
 sums=run/'SHA256SUMS.txt'; sums.write_text('\n'.join(lines)+'\n')
 members=files+[sums]
 with tarfile.open(archive,'w:gz') as tf:
  for p in members: tf.add(p,arcname=f"{run.name}/{p.relative_to(run).as_posix()}",recursive=False)
 # Reopen, reject unsafe names, and verify internal members against SHA256SUMS.
 expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in lines}; verified=0
 with tarfile.open(archive,'r:gz') as tf:
  names=tf.getnames()
  for name in names:
   pp=Path(name)
   if pp.is_absolute() or '..' in pp.parts: raise RuntimeError(f'unsafe archive member {name}')
  for rel,digest in expected.items():
   member=tf.extractfile(f"{run.name}/{rel}")
   if member is None: raise RuntimeError(f'missing archive member {rel}')
   h=hashlib.sha256()
   for b in iter(lambda:member.read(1<<20),b''): h.update(b)
   if h.hexdigest()!=digest: raise RuntimeError(f'archive digest mismatch {rel}')
   verified+=1
 ad=sha(archive); side.write_text(f"{ad}  {archive.name}\n")
 out={'status':'PASS','run':str(run),'archive':str(archive),'sidecar':str(side),'archive_sha256':ad,'archive_safe':True,'file_count':len(files)+1,'archive_member_count':len(members),'internal_checksums':str(sums),'internal_checksums_verified':verified}
 print(json.dumps(out,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
