"""M12 tests — the recruiter-facing packaging rules. Skips until artifacts exist."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
METHOD = REPO / "reports" / "method_note.md"
PRES = REPO / "docs" / "presentation.md"


def _read(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p.relative_to(REPO)} not written yet (M12)")
    return p.read_text(encoding="utf-8")


def test_readme_has_required_sections_in_order():
    text = _read(README)
    if "Status" in text and "Phase 0" in text:
        pytest.skip("README still the P00 placeholder — M12 rewrite pending")
    heads = [h.strip() for h in re.findall(r"^##\s+(.+)$", text, flags=re.MULTILINE)]
    joined = " | ".join(heads).lower()
    for expected in ("method", "limitations", "reproduc", "ethic"):
        assert expected in joined, f"README missing a '{expected}' section (M12 §1)"
    assert heads, "README has no ## sections"


def test_readme_shows_the_money_chart():
    text = _read(README)
    if "Phase 0" in text:
        pytest.skip("README still the P00 placeholder — M12 rewrite pending")
    assert "fig_calibration.png" in text, "the calibration chart must be above the fold"


def test_readme_has_results_table_with_baseline():
    text = _read(README)
    if "Phase 0" in text:
        pytest.skip("README still the P00 placeholder — M12 rewrite pending")
    assert "constant_0.5" in text, "show the trivial baseline — it makes the numbers credible"


def test_readme_has_no_placeholders():
    text = _read(README)
    if "Phase 0" in text:
        pytest.skip("README still the P00 placeholder — M12 rewrite pending")
    lowered = text.lower()
    for banned in ("todo", "lorem", "<insert", "tbd"):
        assert banned not in lowered, f"placeholder text left in README: {banned}"


def test_method_note_length():
    text = _read(METHOD)
    words = len(text.split())
    assert 300 <= words <= 700, f"method note should be ~1 page, got {words} words"


def test_presentation_script_minimum():
    text = _read(PRES)
    assert len(text.split()) >= 300, "walkthrough script should be a real 3-minute script"
