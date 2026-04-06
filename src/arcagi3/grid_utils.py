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


# ANSI true-color (24-bit) versions of the palette for terminal rendering
ANSI_PALETTE = {
    0: (255, 255, 255),  # White
    1: (204, 204, 204),  # Off-white
    2: (153, 153, 153),  # Neutral light
    3: (102, 102, 102),  # Neutral
    4: (51, 51, 51),     # Off-black
    5: (0, 0, 0),        # Black
    6: (229, 58, 163),   # Magenta
    7: (255, 123, 204),  # Magenta light
    8: (249, 60, 49),    # Red
    9: (30, 147, 255),   # Blue
    10: (136, 216, 241), # Blue light
    11: (255, 220, 0),   # Yellow
    12: (255, 133, 27),  # Orange
    13: (146, 18, 49),   # Maroon
    14: (79, 204, 48),   # Green
    15: (163, 86, 214),  # Purple
}


def grid_to_ansi(grid: np.ndarray, downsample: int = 2) -> str:
    """Render grid as colored blocks in the terminal using ANSI true-color.

    Uses background-colored spaces, one char per pixel (after downsampling).

    Args:
        grid: 2D array (64x64), values 0-15.
        downsample: Factor to reduce resolution (2 = 32x32 output).

    Returns:
        String with ANSI escape codes ready to print.
    """
    h, w = grid.shape
    lines = []
    step = downsample
    for y in range(0, h, step):
        line = []
        for x in range(0, w, step):
            val = int(grid[y, x])
            r, g, b = ANSI_PALETTE.get(val, (0, 0, 0))
            line.append(f"\033[48;2;{r};{g};{b}m  ")
        lines.append("".join(line) + "\033[0m")
    return "\n".join(lines)


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


# --- Frame analysis and diff utilities ---

COLOR_NAMES = {
    0: "white", 1: "light-gray", 2: "gray", 3: "dark-gray",
    4: "near-black", 5: "black", 6: "magenta", 7: "pink",
    8: "red", 9: "blue", 10: "light-blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}


def grid_hash(grid: np.ndarray) -> str:
    """Fast hash of a grid for state deduplication."""
    return hash(grid.tobytes())


def find_objects(grid: np.ndarray, background: int | None = None) -> list[dict]:
    """Find connected components (objects) in the grid using flood fill.

    Returns list of objects with: color, pixels, bbox, centroid, size.
    Skips the background color (most common color if not specified).
    """
    h, w = grid.shape
    if background is None:
        # Most common color is background
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[np.argmax(counts)])

    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or int(grid[y, x]) == background:
                continue
            # Flood fill
            color = int(grid[y, x])
            stack = [(y, x)]
            pixels = []
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or int(grid[cy, cx]) != color:
                    continue
                visited[cy, cx] = True
                pixels.append((cx, cy))
                stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])

            if len(pixels) < 2:
                continue  # Skip single pixels (noise)

            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            obj = {
                "color": color,
                "color_name": COLOR_NAMES.get(color, f"color-{color}"),
                "size": len(pixels),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "centroid": (round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)),
            }
            objects.append(obj)

    # Sort by size descending
    objects.sort(key=lambda o: o["size"], reverse=True)
    return objects


def compute_diff(prev: np.ndarray, curr: np.ndarray) -> dict:
    """Compute a structured diff between two frames.

    Returns: changed_cells count, changed_regions, moved_objects summary.
    """
    if prev is None:
        return {"type": "initial", "description": "First frame, no previous to compare."}

    diff_mask = prev != curr
    n_changed = int(np.sum(diff_mask))

    if n_changed == 0:
        return {"type": "no_change", "changed_cells": 0, "description": "NO pixels changed. Action had NO visible effect."}

    total = prev.shape[0] * prev.shape[1]
    pct = round(100 * n_changed / total, 1)

    # Find changed regions (bounding boxes of connected changed areas)
    changed_ys, changed_xs = np.where(diff_mask)
    regions = []
    if n_changed <= 200:
        # Small change: describe precisely
        # Find what colors appeared/disappeared
        old_colors = set(int(prev[y, x]) for y, x in zip(changed_ys, changed_xs))
        new_colors = set(int(curr[y, x]) for y, x in zip(changed_ys, changed_xs))

        old_names = [COLOR_NAMES.get(c, f"c{c}") for c in old_colors]
        new_names = [COLOR_NAMES.get(c, f"c{c}") for c in new_colors]

        # Bounding box of changes
        x_min, x_max = int(changed_xs.min()), int(changed_xs.max())
        y_min, y_max = int(changed_ys.min()), int(changed_ys.max())

        regions.append({
            "bbox": (x_min, y_min, x_max, y_max),
            "center": (round((x_min + x_max) / 2), round((y_min + y_max) / 2)),
            "old_colors": old_names,
            "new_colors": new_names,
        })

    # Detect movement: same-color object disappeared in one spot, appeared in another
    movements = _detect_movements(prev, curr, diff_mask)

    result = {
        "type": "changed",
        "changed_cells": n_changed,
        "changed_pct": pct,
        "regions": regions,
        "movements": movements,
    }

    # Detect swaps: two objects that exchanged positions
    swaps = _detect_swaps(movements)

    result["swaps"] = swaps

    # Build human-readable description
    desc_parts = [f"{n_changed} pixels changed ({pct}% of grid)."]

    # Report swaps first (more informative than raw movements)
    swap_colors = set()
    for s in swaps:
        desc_parts.append(
            f"SWAP: {s['color_a_name']} at ({s['pos_a_x']},{s['pos_a_y']}) <-> "
            f"{s['color_b_name']} at ({s['pos_b_x']},{s['pos_b_y']}) exchanged positions."
        )
        swap_colors.add(s["color_a"])
        swap_colors.add(s["color_b"])

    # Report non-swap movements
    for m in movements:
        if m["color"] not in swap_colors:
            desc_parts.append(
                f"{m['color_name']} object moved from ({m['from_x']},{m['from_y']}) "
                f"to ({m['to_x']},{m['to_y']}), delta=({m['dx']},{m['dy']})."
            )
    if not movements and not swaps and regions:
        r = regions[0]
        desc_parts.append(
            f"Changes in area ({r['bbox'][0]},{r['bbox'][1]})-({r['bbox'][2]},{r['bbox'][3]}). "
            f"Colors before: {', '.join(r['old_colors'])}. After: {', '.join(r['new_colors'])}."
        )
    result["description"] = " ".join(desc_parts)
    return result


def _detect_movements(prev: np.ndarray, curr: np.ndarray, diff_mask: np.ndarray) -> list[dict]:
    """Detect objects that moved between frames."""
    movements = []

    # For each non-background color in the changed area
    changed_ys, changed_xs = np.where(diff_mask)
    if len(changed_ys) == 0:
        return movements

    colors_in_diff = set()
    for y, x in zip(changed_ys, changed_xs):
        colors_in_diff.add(int(prev[y, x]))
        colors_in_diff.add(int(curr[y, x]))

    # Get background (most common in whole grid)
    vals, counts = np.unique(curr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    for color in colors_in_diff:
        if color == bg:
            continue

        # Where was this color before (in changed area only)?
        old_positions = [(int(x), int(y)) for y, x in zip(changed_ys, changed_xs) if int(prev[y, x]) == color]
        new_positions = [(int(x), int(y)) for y, x in zip(changed_ys, changed_xs) if int(curr[y, x]) == color]

        if old_positions and new_positions:
            # Compute centroids
            old_cx = sum(p[0] for p in old_positions) / len(old_positions)
            old_cy = sum(p[1] for p in old_positions) / len(old_positions)
            new_cx = sum(p[0] for p in new_positions) / len(new_positions)
            new_cy = sum(p[1] for p in new_positions) / len(new_positions)

            dx = round(new_cx - old_cx)
            dy = round(new_cy - old_cy)

            if abs(dx) > 0 or abs(dy) > 0:
                movements.append({
                    "color": color,
                    "color_name": COLOR_NAMES.get(color, f"color-{color}"),
                    "from_x": round(old_cx),
                    "from_y": round(old_cy),
                    "to_x": round(new_cx),
                    "to_y": round(new_cy),
                    "dx": dx,
                    "dy": dy,
                })

    return movements


def _detect_swaps(movements: list[dict]) -> list[dict]:
    """Detect pairs of objects that exchanged positions (swaps)."""
    swaps = []
    used = set()
    for i, a in enumerate(movements):
        if i in used:
            continue
        for j, b in enumerate(movements):
            if j <= i or j in used:
                continue
            # Check if A went to where B was and B went to where A was
            if (a["from_x"] == b["to_x"] and a["from_y"] == b["to_y"]
                    and a["to_x"] == b["from_x"] and a["to_y"] == b["from_y"]):
                swaps.append({
                    "color_a": a["color"],
                    "color_a_name": a["color_name"],
                    "color_b": b["color"],
                    "color_b_name": b["color_name"],
                    "pos_a_x": a["from_x"],
                    "pos_a_y": a["from_y"],
                    "pos_b_x": b["from_x"],
                    "pos_b_y": b["from_y"],
                })
                used.add(i)
                used.add(j)
                break
    return swaps


def describe_frame(grid: np.ndarray) -> str:
    """Generate a text description of the current frame: objects, layout, colors."""
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)

    # Color distribution (skip tiny amounts)
    total = h * w
    color_dist = []
    bg_color = int(vals[np.argmax(counts)])
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        pct = round(100 * int(c) / total, 1)
        if pct >= 0.5:
            name = COLOR_NAMES.get(int(v), f"color-{int(v)}")
            color_dist.append(f"{name}: {pct}%")

    objects = find_objects(grid, background=bg_color)

    lines = [f"Background: {COLOR_NAMES.get(bg_color, f'color-{bg_color}')}"]
    lines.append(f"Colors present: {', '.join(color_dist)}")
    if objects:
        lines.append(f"Objects found ({len(objects)}):")
        for i, obj in enumerate(objects[:15]):  # Limit to top 15
            bbox = obj["bbox"]
            cx, cy = obj["centroid"]
            lines.append(
                f"  #{i+1}: {obj['color_name']} object, "
                f"{obj['size']}px, center=({cx},{cy}), "
                f"bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"
            )
    return "\n".join(lines)
