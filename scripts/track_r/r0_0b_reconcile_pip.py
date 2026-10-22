#!/usr/bin/env python3
"""Reconcile only missing pip dependencies inside a cloned, isolated R0 env.

Source environment is NEVER mutated. Unresolved new dependency conflicts stop.
Pre-existing identical source conflicts are logged as inherited warnings.
Avoid all `pip install --upgrade` and never install a replacement for an
existing package (no core torch/numpy version drift by dependency resolution).
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import re
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

MISSING = re.compile(
    r"^(?P<req>.+?)\s+(?P<version>\S+)\s+requires\s+"
    r"(?P<dep>[-a-zA-Z0-9_.]+), which is not installed\.$",
    re.I,
)
PROHIBITED = {
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "triton",
}
MAX_NEW_PACKAGES = 12
MAX_PASSES = 5
SETUPTOOLS_PIN = "80.9.0"
TORCH_SETUPTOOLS_CONFLICT = re.compile(
    r"^torch (?P<torch_version>\S+) has requirement setuptools<82, "
    r"but you have setuptools (?P<installed>\S+)\.$",
    re.I,
)
CORE_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "triton")


def pinned_setuptools_for(issue):
    """Authorize exactly the known torch/setuptools incompatibility.

    Only the isolated target's setuptools may be downgraded, and only
    to the R0.0b handoff's pinned version. All other conflicts fail.
    """
    found = TORCH_SETUPTOOLS_CONFLICT.fullmatch(issue)
    if not found:
        return None
    try:
        current = metadata.version("setuptools")
        torch_version = metadata.version("torch")
        torch_dist = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        return None
    if current != found.group("installed") or torch_version != found.group("torch_version"):
        return None
    if Version(current) < Version("82") or Version(SETUPTOOLS_PIN) >= Version("82"):
        return None
    requested = []
    for item in torch_dist.requires or ():
        requirement = Requirement(item)
        if canonicalize_name(requirement.name) != "setuptools":
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        requested.append(requirement)
    if not requested or any(
        Version(SETUPTOOLS_PIN) not in r.specifier for r in requested
    ):
        return None
    return "setuptools==" + SETUPTOOLS_PIN


def installed_versions(python, names):
    """Retrieve metadata only; don't import torch or activate CUDA."""
    script = (
        "import json,importlib.metadata as m,sys\n"
        "out={}\n"
        "for name in sys.argv[1:]:\n"
        "  try: out[name]=m.version(name)\n"
        "  except m.PackageNotFoundError: out[name]=None\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    output = subprocess.check_output(
        [python, "-c", script, *names], text=True, timeout=45
    )
    import json
    return json.loads(output)


# R0.0 imported NumPy 2.4.6 from prdiff_ffhq. Conda and pip
# distributions can leave multiple numpy *.dist-info directories, in
# which case importlib.metadata.version() need not reflect the imported
# binary module. This exception is limited to NumPy and checked against
# the prior observed runtime, file hashes, and extension binary hashes.
NUMPY_INVENTORY_RUNTIME = "2.4.6"


def numpy_runtime_evidence(python):
    script = (
        "import hashlib,json,pathlib,numpy as np\\n"
        "import numpy._core._multiarray_umath as ext\\n"
        "def digest(path):\\n"
        "  h=hashlib.sha256()\\n"
        "  with open(path,'rb') as f:\\n"
        "    for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)\\n"
        "  return h.hexdigest()\\n"
        "print('R0B_NUMPY_EVIDENCE='+json.dumps({"
        "'version':np.__version__,"
        "'module_sha256':digest(np.__file__),"
        "'extension_sha256':digest(ext.__file__),"
        "'module_path':str(pathlib.Path(np.__file__).resolve()),"
        "'extension_path':str(pathlib.Path(ext.__file__).resolve())},"
        "sort_keys=True))\\n"
    )
    result = subprocess.run(
        [python, "-c", script], capture_output=True, text=True,
        timeout=120, check=False
    )
    if result.returncode:
        raise RuntimeError("STOP_NUMPY_RUNTIME_IMPORT_RC "+str(result.returncode)
                           +" stderr="+result.stderr[-600:])
    import json
    rows = [line for line in result.stdout.splitlines()
            if line.startswith("R0B_NUMPY_EVIDENCE=")]
    if len(rows)!=1:
        raise RuntimeError("STOP_NUMPY_RUNTIME_EVIDENCE_UNAVAILABLE")
    return json.loads(rows[0].split("=",1)[1])


def numpy_metadata_ambiguity_safe(source, target):
    src=numpy_runtime_evidence(source)
    dst=numpy_runtime_evidence(target)
    for scope, record, prefix in (
        ("source",src,"/egr/research-pac/huang248/conda-envs/prdiff_ffhq/"),
        ("clone",dst,"/egr/research-pac/huang248/conda-envs/prdiff_track_r/"),
    ):
        print("R0B_NUMPY_RUNTIME",scope,record,flush=True)
        for field in ("module_path","extension_path"):
            if not os.path.realpath(record[field]).startswith(prefix):
                print("STOP_R0B_NUMPY_OUTSIDE_ENV",scope,field,flush=True)
                return False
    if src["version"]!=NUMPY_INVENTORY_RUNTIME or dst["version"]!=src["version"]:
        print("STOP_R0B_NUMPY_RUNTIME_VERSION_DRIFT",flush=True)
        return False
    if (src["module_sha256"]!=dst["module_sha256"] or
            src["extension_sha256"]!=dst["extension_sha256"]):
        print("STOP_R0B_NUMPY_RUNTIME_BINARY_DRIFT",flush=True)
        return False
    print("R0B_NUMPY_RUNTIME_IDENTICAL_METADATA_DISAGREES",flush=True)
    return True


def core_versions_match(source, target):
    src=installed_versions(source,CORE_PACKAGES)
    dst=installed_versions(target,CORE_PACKAGES)
    for name in CORE_PACKAGES:
        if src[name]==dst[name]:
            continue
        if name=="numpy":
            if numpy_metadata_ambiguity_safe(source,target):
                print("R0B_WARN_NUMPY_DISTINFO_DISAGREEMENT",
                      src[name],dst[name],flush=True)
                continue
        print("STOP_R0B_CORE_CLONE_DRIFT",name,src[name],dst[name],flush=True)
        return False
    print("R0B_CLONE_CORE_VERSIONS_MATCH_OR_VERIFIED_NUMPY_METADATA",src,dst,flush=True)
    return True


def check(py):
    p = subprocess.run([py, "-m", "pip", "check"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=120, check=False)
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    return p.returncode, lines


def missing_spec(line):
    m = MISSING.match(line)
    return m.groupdict() if m else None


def requirement_for(requiring, dep):
    """Retrieve version constraint from installed requesting distribution."""
    try:
        dist = metadata.distribution(requiring)
    except metadata.PackageNotFoundError:
        raise RuntimeError("STOP_CANNOT_IDENTIFY_REQUESTER " + requiring)
    matches = []
    for spec in dist.requires or ():
        requirement = Requirement(spec)
        if canonicalize_name(requirement.name) != canonicalize_name(dep):
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        matches.append(str(requirement.name) + str(requirement.specifier))
    if not matches:
        raise RuntimeError("STOP_NO_RUNTIME_REQUIREMENT " + requiring + " -> " + dep)
    if len(set(matches)) > 1:
        raise RuntimeError("STOP_AMBIGUOUS_REQUIREMENT " + str(matches))
    return matches[0]


def reconcile(source, target):
    if os.path.realpath(source) == os.path.realpath(target):
        raise RuntimeError("STOP_RECONCILER_TARGET_IS_SOURCE")
    if not os.path.isfile(source) or not os.path.isfile(target):
        raise RuntimeError("STOP_INTERPRETER_MISSING")
    root = "/egr/research-pac/huang248/"
    if not (os.path.realpath(source).startswith(root+"conda-envs/prdiff_ffhq/") and
            os.path.realpath(target).startswith(root+"conda-envs/prdiff_track_r/")):
        raise RuntimeError("STOP_RECONCILER_BAD_PREFIX")
    if not core_versions_match(source, target):
        return 9
    rc_base, base = check(source)
    print("R0B_SOURCE_PIP_CHECK_RC",rc_base,flush=True)
    for x in base: print("R0B_SOURCE_PIP_CHECK",x,flush=True)
    baseline = set(base)
    added=[]
    for i in range(MAX_PASSES):
        rc, rows = check(target)
        print("R0B_TARGET_PIP_CHECK_PASS",i,"RC",rc,flush=True)
        for x in rows: print("R0B_TARGET_PIP_CHECK",x,flush=True)
        if rc==0:
            print("R0B_DEPENDENCIES_VERIFIED",flush=True)
            return 0
        new = [line for line in rows if line not in baseline]
        if not new:
            print("R0B_INHERITED_ONLY_PIP_WARN",
                  "Original clone's pip check is equally inconsistent; "
                  "no package replacement was attempted",flush=True)
            return 0
        specs=[]
        for issue in new:
            detail=missing_spec(issue)
            if not detail:
                pin=pinned_setuptools_for(issue)
                if pin:
                    print("R0B_APPROVED_ISOLATED_SETUPTOOLS_PIN",pin,flush=True)
                    if pin not in specs: specs.append(pin)
                    continue
                print("STOP_R0B_NEW_UNRESOLVED_DEPENDENCY",issue,flush=True)
                return 3
            dep = canonicalize_name(detail["dep"])
            if dep in PROHIBITED or dep.startswith("nvidia-"):
                print("STOP_R0B_CORE_DEPENDENCY_ABSENT",issue,flush=True)
                return 4
            req=requirement_for(detail["req"], detail["dep"])
            if req not in specs: specs.append(req)
        if len(specs)+len(added)>MAX_NEW_PACKAGES:
            print("STOP_R0B_TOO_MANY_DEPENDENCIES",specs,flush=True)
            return 5
        print("R0B_INSTALL_MISSING_OR_PINNED_ONLY",*specs,flush=True)
        p=subprocess.run([target,"-m","pip","install","--no-input","--no-deps",
                          *specs],timeout=600,check=False)
        if p.returncode:
            print("STOP_R0B_DEPENDENCY_INSTALL_RC",p.returncode,flush=True)
            return p.returncode
        added.extend(specs)
    print("STOP_R0B_DEPENDENCY_LOOP_EXHAUSTED",flush=True)
    return 6


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source",required=True)
    a.add_argument("--target",required=True)
    args=a.parse_args()
    try:
        return reconcile(args.source,args.target)
    except (RuntimeError,subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        print("ENV_FAILED_R0B_DEPENDENCY_RECONCILE",repr(exc),flush=True)
        return 20


if __name__=="__main__":
    sys.exit(main())
