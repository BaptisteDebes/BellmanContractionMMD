"""Colors and axis helpers shared by the scripts that draw the paper's figures."""

import re

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes

TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"
BACKGROUND_COLOR = "#FFFFFF"


def build_cell_edges(centers: Float[np.ndarray, " cells"]) -> Float[np.ndarray, " edges"]:
    """Build the cell boundaries of an evenly spaced grid from its cell centers.

    Args:
        centers: The grid values, in increasing order: shape (cells,).

    Returns:
        The midpoints between neighbours, with half a step added at both ends:
        shape (cells + 1,).
    """
    # Midpoints between neighbouring centers, shape (cells − 1,)
    midpoints = 0.5 * (centers[1:] + centers[:-1])
    # First edge, half a step below the first center
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    # Last edge, half a step above the last center
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    # All edges, shape (cells + 1,)
    return np.concatenate(([first], midpoints, [last]))


def style_axes(axes: Axes) -> None:
    """Draw a light grid and keep only the left and bottom spines."""
    axes.grid(True, color=GRID_COLOR, alpha=0.88, linewidth=0.95, linestyle="-")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(TEXT_COLOR)
        axes.spines[side].set_linewidth(1.45)


def convert_to_bold_mathtext(text: str) -> str:
    """Write each line of a title as bold mathtext, so that it can share a label with plain lines.

    A Matplotlib label has a single font weight. Bold mathtext uses the same font, DejaVu Sans
    Bold, so the title looks as if it were set with fontweight="bold".

    Args:
        text: Letters, digits, spaces, commas and line breaks.

    Returns:
        The text with each line written as $\\mathbf{…}$ and its spaces escaped.

    Raises:
        ValueError: If the text holds a character that mathtext would read as a symbol.
    """
    if re.fullmatch(r"[A-Za-z0-9 ,\n]*", text) is None:
        raise ValueError(f"only letters, digits, spaces, commas and line breaks: {text!r}")
    return "\n".join(r"$\mathbf{" + line.replace(" ", r"\ ") + "}$" for line in text.split("\n"))
