"""Keep the algorithms page's per-algorithm help in step with ALGORITHMS.

The page itself can't be imported here — ui/ needs a running QGIS — so the
help dictionary is read out of the source with `ast` instead. That is enough
to catch the one failure mode worth guarding: adding a tenth algorithm and
forgetting to describe it, which would otherwise show as an empty box under a
bare heading and only ever be noticed by a user hovering over it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from sdm_plugin.core.config import ALGORITHMS

PAGE = Path(__file__).parent.parent / "ui" / "pages" / "algorithms_page.py"


def _literal_dict(source: str, name: str) -> dict:
    """The value of a module-level `name = {...}` assignment, evaluated as a
    literal. Implicit string concatenation across lines is a literal too, so
    the page's multi-line help strings come through intact."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {PAGE.name}")


def test_every_algorithm_has_help_text():
    help_text = _literal_dict(PAGE.read_text(encoding="utf-8"), "_ALGORITHM_HELP")
    assert set(help_text) == set(ALGORITHMS), (
        "every algorithm offered on the page needs a description, and a "
        "description with no algorithm is dead text"
    )


def test_help_text_keeps_its_three_part_shape():
    """Each entry is what-it-is / Best for: / Watch out for:, matching the
    cross-validation and ensemble pages. A missing part reads as an oversight
    next to its neighbours."""
    help_text = _literal_dict(PAGE.read_text(encoding="utf-8"), "_ALGORITHM_HELP")
    for algo, text in help_text.items():
        assert "<b>Best for:</b>" in text, f"{algo} is missing its 'Best for' line"
        assert "<b>Watch out for:</b>" in text, f"{algo} is missing its 'Watch out for' line"
        assert text.count("<p>") == 3, f"{algo} should be exactly three paragraphs"
