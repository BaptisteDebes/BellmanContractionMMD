"""Test of the bold row titles of the figure scripts, on cases worked out by hand."""

import pytest

from paper_figures.figure_style import convert_to_bold_mathtext


def test_bold_mathtext_writes_each_line_in_bold_with_its_spaces_escaped() -> None:
    """A two-line row title of Figure 3 becomes two bold mathtext lines."""
    title = "1D Same Mean,\nDifferent Std"

    expected = r"$\mathbf{1D\ Same\ Mean,}$" + "\n" + r"$\mathbf{Different\ Std}$"
    assert convert_to_bold_mathtext(title) == expected


def test_bold_mathtext_refuses_a_hyphen() -> None:
    """Mathtext would draw the hyphen of "Two-Gaussian" as a minus sign."""
    with pytest.raises(ValueError, match="Two-Gaussian"):
        convert_to_bold_mathtext("Two-Gaussian")
