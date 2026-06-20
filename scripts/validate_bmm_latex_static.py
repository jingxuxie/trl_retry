#!/usr/bin/env python
"""Static checks for the BMM-TRL LaTeX manuscript."""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = REPO_ROOT / "paper" / "bmm_trl" / "main.tex"
BIB_PATH = REPO_ROOT / "paper" / "bmm_trl" / "references.bib"


def fail(message: str) -> None:
    raise AssertionError(message)


def brace_balance(text: str) -> None:
    depth = 0
    escaped = False
    for idx, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth < 0:
            fail(f"Unbalanced closing brace near byte {idx}")
    if depth != 0:
        fail(f"Unbalanced braces: final depth {depth}")


def collect_citations(tex: str) -> set[str]:
    citations = set()
    for match in re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
        citations.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return citations


def collect_bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", bib))


def collect_labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", tex)


def collect_refs(tex: str) -> set[str]:
    refs = set()
    for match in re.finditer(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", tex):
        refs.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return refs


def validate_surface(tex: str) -> None:
    forbidden = [
        "TODO",
        "TBD",
        "FIXME",
        "placeholder",
        "goblin",
        "gremlin",
    ]
    for needle in forbidden:
        if needle.lower() in tex.lower():
            fail(f"Forbidden manuscript marker found: {needle}")

    required = [
        "Budgeted Max-Min Transitive RL",
        "non-expansive",
        "Budget-holdout",
        "Product controls",
        "Scene-Play support-graph transfer",
        "Fixed-controller hierarchical planning",
        "Reproducibility",
        "\\path{BMM_TRL_REPRO_COMMANDS.md}",
        "52/75 (69.3\\%)",
        "58/75",
        "55/75 (73.3\\%)",
        "66/75 (88.0\\%)",
        "63/75 (84.0\\%)",
        "not as a single uniform policy-extraction or pure-BMM",
    ]
    for needle in required:
        if needle not in tex:
            fail(f"Missing required manuscript phrase: {needle}")


def validate() -> None:
    tex = TEX_PATH.read_text()
    bib = BIB_PATH.read_text()
    brace_balance(tex)

    citations = collect_citations(tex)
    bib_keys = collect_bib_keys(bib)
    missing_citations = sorted(citations - bib_keys)
    if missing_citations:
        fail(f"Missing bibliography entries: {missing_citations}")

    labels = collect_labels(tex)
    label_counts = Counter(labels)
    duplicates = sorted(label for label, count in label_counts.items() if count > 1)
    if duplicates:
        fail(f"Duplicate labels: {duplicates}")

    missing_refs = sorted(collect_refs(tex) - set(labels))
    if missing_refs:
        fail(f"Missing referenced labels: {missing_refs}")

    table_count = len(re.findall(r"\\begin\{table\}", tex))
    caption_count = len(re.findall(r"\\caption\{", tex))
    if table_count != caption_count:
        fail(f"Expected every table to have one caption, got {table_count} tables and {caption_count} captions")

    validate_surface(tex)


def main() -> None:
    validate()
    print("PASS: BMM-TRL LaTeX static checks passed.")


if __name__ == "__main__":
    main()
