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
        print("R0B_INSTALL_MISSING_ONLY",*specs,flush=True)
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
    except (RuntimeError,subprocess.TimeoutExpired) as exc:
        print("ENV_FAILED_R0B_DEPENDENCY_RECONCILE",repr(exc),flush=True)
        return 20


if __name__=="__main__":
    sys.exit(main())
