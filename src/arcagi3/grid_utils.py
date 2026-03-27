"""Utilities for converting ARC-AGI-3 grids to images and text."""

import base64
import io
import json

import numpy as np
from PIL import Image

# Official ARC-AGI-3 16-color palette (RGBA)
COLOR_PALETTE = {
    0: (255, 255, 255, 255),   # White
    1: (204, 204, 204, 255),   # Off-white
    2: (153, 153, 153, 255),   # Neutral light
    3: (102, 102, 102, 255),   # Neutral
    4: (51, 51, 51, 255),      # Off-black
    5: (0, 0, 0, 255),         # Black
    6: (229, 58, 163, 255),    # Magenta
    7: (255, 123, 204, 255),   # Magenta light
    8: (249, 60, 49, 255),     # Red
    9: (30, 147, 255, 255),    # Blue
    10: (136, 216, 241, 255),  # Blue light
    11: (255, 220, 0, 255),    # Yellow
    12: (255, 133, 27, 255),   # Orange
    13: (146, 18, 49, 255),    # Maroon
    14: (79, 204, 48, 255),    # Green
    15: (163, 86, 214, 255),   # Purple
}


def grid_to_image(grid: np.ndarray, scale: int = 2) -> Image.Image:
    """Convert a 64x64 grid (values 0-15) to a PIL Image.

    Args:
        grid: 2D numpy array with shape (64, 64), values 0-15.
        scale: Upscale factor (2 = 128x128 output).

    Returns:
        PIL Image in RGBA mode.
    """
    h, w = grid.shape
    img = Image.new("RGBA", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = COLOR_PALETTE.get(int(grid[y, x]), (0, 0, 0, 255))
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    return img


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def grid_to_base64(grid: np.ndarray, scale: int = 2) -> str:
    """Convert grid directly to base64 PNG string."""
    return image_to_base64(grid_to_image(grid, scale))


def image_diff(prev: np.ndarray, curr: np.ndarray, scale: int = 2) -> Image.Image:
    """Create a diff image highlighting changed pixels between two grids.

    Unchanged pixels are dimmed (50% opacity), changed pixels are full brightness.
    """
    h, w = curr.shape
    img = Image.new("RGBA", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            color = COLOR_PALETTE.get(int(curr[y, x]), (0, 0, 0, 255))
            if prev is not None and prev[y, x] == curr[y, x]:
                # Dim unchanged pixels
                color = (color[0] // 2, color[1] // 2, color[2] // 2, 128)
            pixels[x, y] = color
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    return img


def grid_to_text(grid: np.ndarray) -> str:
    """Convert grid to compact text representation (JSON matrix)."""
    return json.dumps(grid.tolist())


def grid_to_text_compact(grid: np.ndarray) -> str:
    """Convert grid to very compact text — one hex char per cell, rows as lines.

    Uses hex (0-f) for values 0-15. Much shorter than JSON for 64x64 grids.
    """
    lines = []
    for row in grid:
        lines.append("".join(format(int(v), "x") for v in row))
    return "\n".join(lines)
