#!/usr/bin/env python3
"""A1.0 deterministic candidate and cohort protocol; CPU + stdlib only.

No filesystem reads, GPU, models, tensors, FFT, or measurements. The caller MUST
supply a preapproved eligible ID set and allowed exposed/reserved exclusions.
This module neither knows nor can query locked confirmation membership.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

VERSION = "PRDIFF-A1-V1"
DEV_RANK_DOMAIN = "PRDIFF-A1-DEV-IMAGE-RANK-V1"
ARMS = ("DAPS", "SITCOM", "NP_LF", "NP_S2")
K = {"DAPS": 4, "SITCOM": 4, "NP_LF": 4, "NP_S2": 4}
EXTRA_OBSERVED = ("NP_MIXED8",)
EXPECTED_UNIQUE_CANDIDATES = 16
ID_RE = re.compile(r"^[0-9]{5}$")


class ProtocolStop(ValueError):
    """Source, seed or approved-panel preflight failed; no run allowed."""


def image_id(value: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ProtocolStop(f"invalid five-digit image ID: {value!r}")
    return value


def seed63(tag: str, identifier: str, index: int) -> int:
    identifier = image_id(identifier)
    if tag not in ARMS + ("A1_MEAS_V1",):
        raise ProtocolStop(f"unknown seed domain {tag!r}")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ProtocolStop("seed replicate must be a nonnegative integer")
    payload = f"{VERSION}|{tag}|{identifier}|{index}".encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def seed32_numpy(tag: str, identifier: str, index: int) -> int:
    """Legacy NumPy global RNG requires an unsigned 32-bit seed.

    Full 63-bit seed63 belongs to torch / compatible generators; use this
    independent domain-separated digest for numpy.random.seed().
    """
    identifier = image_id(identifier)
    if tag not in ARMS + ("A1_MEAS_V1",):
        raise ProtocolStop(f"unknown numpy seed domain {tag!r}")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ProtocolStop("numpy replicate must be a nonnegative integer")
    payload = f"{VERSION}|numpy32|{tag}|{identifier}|{index}".encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class Candidate:
    image_id: str
    family: str
    config: str
    replicate: int
    candidate_id: str
    seed: int


def candidate_matrix(identifier: str) -> list[Candidate]:
    identifier = image_id(identifier)
    result: list[Candidate] = []
    for family in ARMS:
        for j in range(K[family]):
            result.append(Candidate(
                image_id=identifier,
                family=family,
                config="lf" if family == "NP_LF" else
                       "s2_preproj_lam001" if family == "NP_S2" else "native",
                replicate=j,
                candidate_id=f"{identifier}:{family}:{j}",
                seed=seed63(family, identifier, j),
            ))
    verify_matrix(result)
    return result


def verify_matrix(rows: list[Candidate]) -> None:
    if len(rows) != EXPECTED_UNIQUE_CANDIDATES:
        raise ProtocolStop("missing/extra unique native trajectories")
    identities = [c.candidate_id for c in rows]
    seeds = [c.seed for c in rows]
    if len(set(identities)) != len(identities):
        raise ProtocolStop("duplicated candidate identity")
    if len(set(seeds)) != len(seeds):
        raise ProtocolStop("seed collision across configs or native families")
    ids = {c.image_id for c in rows}
    if len(ids) != 1:
        raise ProtocolStop("mixed image IDs")
    for c in rows:
        if c.family not in ARMS or not 0 <= c.replicate < K[c.family]:
            raise ProtocolStop("unknown family/replicate")
        if seed63(c.family, c.image_id, c.replicate) != c.seed:
            raise ProtocolStop("candidate seed drift")
    if Counter(c.family for c in rows) != Counter(K):
        # Counter(K) uses K's dict values as counts.
        raise ProtocolStop("native family candidate counts differ")


def mixed_np8(rows: list[Candidate]) -> list[Candidate]:
    verify_matrix(rows)
    lf = {c.replicate: c for c in rows if c.family == "NP_LF"}
    s2 = {c.replicate: c for c in rows if c.family == "NP_S2"}
    # Fixed interleaving, exactly four candidates of each config; NO extra runs.
    mixed = [c for j in range(4) for c in (lf[j], s2[j])]
    if len({c.candidate_id for c in mixed}) != 8:
        raise ProtocolStop("NP mixed8 candidate identities are not unique")
    return mixed


def mixed_np4(rows: list[Candidate]) -> list[Candidate]:
    return mixed_np8(rows)[:4]


def select_dev_ids(available_public_ids: list[str],
                   exposed_allowlisted_ids: set[str],
                   track_reserved_ids: set[str], n: int = 8) -> list[str]:
    """Select only after caller separately certifies allowed inventory sources.

    This is NOT a certificate that inputs exclude any locked cohort. Never
    provide / derive IDs from excluded B24 confirmation305 in this function.
    """
    if n != 8:
        raise ProtocolStop("frozen A1.1 dev smoke must be 8 images")
    candidates = [image_id(x) for x in available_public_ids]
    exposed = {image_id(x) for x in exposed_allowlisted_ids}
    reserved = {image_id(x) for x in track_reserved_ids}
    if len(set(candidates)) != len(candidates):
        raise ProtocolStop("duplicate available image ID")
    eligible = [s for s in candidates if s not in exposed and s not in reserved]
    if len(eligible) < n:
        raise ProtocolStop("not enough independently eligible public DEV images")
    rank = lambda s: (hashlib.sha256(
        f"{DEV_RANK_DOMAIN}|{s}".encode("utf8")).hexdigest(), s)
    selected = sorted(eligible, key=rank)[:n]
    if set(selected) & (exposed | reserved):
        raise ProtocolStop("selection leaked exposed or reserved IDs")
    return selected
