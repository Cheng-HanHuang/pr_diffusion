#!/usr/bin/env python3
"""Strict JSON helpers for B22 artifacts.

B22 result files must be standards-compliant JSON.  Some diagnostic statistics
are mathematically undefined for fixed configurations (for example, the
post-projection first-vs-second candidate margin when ``hard_candidates=1``).
Those values are represented as JSON ``null`` rather than non-standard NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite scalar diagnostics with ``None``.

    Scientific tensors and arrays must be summarized before reaching this
    function.  Scalar NumPy values are converted to their Python equivalents.
    Strings and bytes are treated as atomic values rather than sequences.
    """

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_json(item) for item in value]

    return value


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    """Return dotted paths of non-finite scalar values before sanitization."""

    paths: list[str] = []

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            paths.append(prefix or "<root>")
        return paths

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(nonfinite_paths(item, child))
        return paths

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(nonfinite_paths(item, child))

    return paths
