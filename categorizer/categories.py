"""Authoritative category list + optional per-category instructions.

Loaded from config/categories.yaml at import. This is domain config, not a
"tool", so it lives in its own module and is consumed by agent, pipeline, eval,
and tools.
"""

import os

import yaml

from categorizer.paths import CATEGORIES_PATH


def _load_categories(path: str) -> tuple[tuple[str, ...], dict[str, str]]:
    """Load the authoritative category list and per-category instructions from
    categories.yaml. Returns (names tuple, {name: instructions}).

    Raises rather than degrading to a partial/empty list: a malformed or
    missing file is a setup error, not something to silently work around.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Category definitions not found at {path}. Copy categories.sample.yaml "
            "to categories.yaml and fill in any per-category instructions. This file "
            "is required; it holds the authoritative category list."
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty YAML list of categories.")

    names: list[str] = []
    instructions: dict[str, str] = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{path} entry #{i + 1} is missing a 'name' field.")
        name = entry["name"]
        # Absent or null instructions default to "" (i.e. no special handling),
        # which is the intended "optional" behavior — the field can be omitted.
        instr = (entry.get("instructions") or "").strip()
        names.append(name)
        instructions[name] = instr

    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate category names.")

    return tuple(names), instructions


CATEGORIES, CATEGORY_INSTRUCTIONS = _load_categories(CATEGORIES_PATH)
