"""Dependency-independent validation for the Draft 2020-12 vocabulary used by B23.

The validator fails if a B23 schema introduces an unsupported assertion keyword.
When ``jsonschema`` is installed, callers can additionally run the official
Draft202012Validator; the custom path is always executed.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    """A schema or instance violates the supported B23 Draft 2020-12 contract."""


ANNOTATIONS = {
    "$schema", "$id", "$defs", "title", "description", "default", "examples",
}
ASSERTIONS = {
    "$ref", "type", "const", "enum", "required", "properties",
    "additionalProperties", "items", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "allOf", "anyOf", "oneOf", "not",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": _is_number(value),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, Mapping),
    }.get(expected, False)


def _resolve(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"only local JSON pointers are supported: {reference}")
    value: Any = root
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            raise SchemaValidationError(f"unresolved schema reference: {reference}")
        value = value[token]
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"schema reference is not an object: {reference}")
    return value


def check_supported_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise SchemaValidationError("B23 schemas must declare Draft 2020-12")

    def walk(node: Any, *, is_property_map: bool = False) -> None:
        if isinstance(node, Mapping):
            if not is_property_map:
                unknown = [
                    key for key in node
                    if key not in ANNOTATIONS and key not in ASSERTIONS and not key.startswith("x-")
                ]
                if unknown:
                    raise SchemaValidationError(f"unsupported schema keywords: {sorted(unknown)}")
            for key, child in node.items():
                if key in {"properties", "$defs"}:
                    walk(child, is_property_map=True)
                elif is_property_map:
                    walk(child)
                elif key in {"items", "additionalProperties", "allOf", "anyOf", "oneOf", "not"}:
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)


def validate(instance: Any, schema: Mapping[str, Any]) -> None:
    """Validate an instance against every assertion keyword used by B23."""

    check_supported_schema(schema)

    def visit(value: Any, node: Mapping[str, Any], path: str) -> None:
        if "$ref" in node:
            visit(value, _resolve(schema, node["$ref"]), path)
        expected = node.get("type")
        if expected is not None:
            expected_types = [expected] if isinstance(expected, str) else expected
            if not isinstance(expected_types, list) or not all(isinstance(x, str) for x in expected_types):
                raise SchemaValidationError(f"{path}: malformed type assertion")
            if not any(_type_matches(value, item) for item in expected_types):
                raise SchemaValidationError(f"{path}: expected type {expected_types}")
        if "const" in node and value != node["const"]:
            raise SchemaValidationError(f"{path}: const mismatch")
        if "enum" in node and value not in node["enum"]:
            raise SchemaValidationError(f"{path}: value is outside enum")
        if _is_number(value):
            bounds = (
                ("minimum", lambda a, b: a >= b),
                ("maximum", lambda a, b: a <= b),
                ("exclusiveMinimum", lambda a, b: a > b),
                ("exclusiveMaximum", lambda a, b: a < b),
            )
            for key, predicate in bounds:
                if key in node and not predicate(value, node[key]):
                    raise SchemaValidationError(f"{path}: violates {key}")
        if isinstance(value, str):
            if len(value) < node.get("minLength", 0):
                raise SchemaValidationError(f"{path}: shorter than minLength")
            if "maxLength" in node and len(value) > node["maxLength"]:
                raise SchemaValidationError(f"{path}: longer than maxLength")
            if "pattern" in node and re.search(node["pattern"], value) is None:
                raise SchemaValidationError(f"{path}: pattern mismatch")
        if isinstance(value, list):
            if len(value) < node.get("minItems", 0):
                raise SchemaValidationError(f"{path}: fewer than minItems")
            if "maxItems" in node and len(value) > node["maxItems"]:
                raise SchemaValidationError(f"{path}: more than maxItems")
            if node.get("uniqueItems"):
                canonical = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
                if len(canonical) != len(set(canonical)):
                    raise SchemaValidationError(f"{path}: items are not unique")
            if isinstance(node.get("items"), Mapping):
                for index, item in enumerate(value):
                    visit(item, node["items"], f"{path}[{index}]")
        if isinstance(value, Mapping):
            required = node.get("required", [])
            missing = [key for key in required if key not in value]
            if missing:
                raise SchemaValidationError(f"{path}: missing required fields {missing}")
            properties = node.get("properties", {})
            for key, child in properties.items():
                if key in value:
                    visit(value[key], child, f"{path}.{key}")
            extras = set(value) - set(properties)
            additional = node.get("additionalProperties", True)
            if extras and additional is False:
                raise SchemaValidationError(f"{path}: additional properties {sorted(extras)}")
            if isinstance(additional, Mapping):
                for key in extras:
                    visit(value[key], additional, f"{path}.{key}")
        for child in node.get("allOf", []):
            visit(value, child, path)
        if "anyOf" in node and not any(_accepts(value, child, path) for child in node["anyOf"]):
            raise SchemaValidationError(f"{path}: no anyOf branch matched")
        if "oneOf" in node and sum(_accepts(value, child, path) for child in node["oneOf"]) != 1:
            raise SchemaValidationError(f"{path}: oneOf did not match exactly once")
        if "not" in node and _accepts(value, node["not"], path):
            raise SchemaValidationError(f"{path}: forbidden by not")

    def _accepts(value: Any, node: Mapping[str, Any], path: str) -> bool:
        try:
            visit(value, node, path)
        except SchemaValidationError:
            return False
        return True

    visit(instance, schema, "$")


def validate_with_mode(instance: Any, schema: Mapping[str, Any]) -> str:
    """Always run the equivalent validator, then optionally the official one."""

    validate(instance, schema)
    try:
        import jsonschema
    except ImportError:
        return "CUSTOM_DRAFT2020_SUBSET_EQUIVALENT"
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(instance)
    return "JSONSCHEMA_DRAFT_2020_12_PLUS_CUSTOM_EQUIVALENCE"
