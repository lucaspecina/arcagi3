"""Deterministic trackers for harness intelligence.

These are NOT LLM-based — they use simple heuristics on frame data
to identify key game elements: avatar, health bars, goals.
"""

import numpy as np
from .grid_utils import find_objects, compute_diff, COLOR_NAMES


class AvatarTracker:
    """Track which object the player controls by correlating actions with movement.

    Logic: if the same object/color consistently moves after actions,
    it's probably the avatar.
    """

    def __init__(self):
        self.movement_log: list[dict] = []  # {action, movements}
        self.avatar_candidate: dict | None = None
        self.confidence: int = 0  # How many consistent observations

    def update(self, action: str, diff_result: dict) -> None:
        """Update tracker with new action-diff pair."""
        movements = diff_result.get("movements", [])
        if not movements:
            return

        self.movement_log.append({"action": action, "movements": movements})

        # Find the color that moves most consistently
        color_counts: dict[int, int] = {}
        for entry in self.movement_log:
            for m in entry["movements"]:
                c = m["color"]
                color_counts[c] = color_counts.get(c, 0) + 1

        if color_counts:
            best_color = max(color_counts, key=color_counts.get)
            count = color_counts[best_color]

            if count >= 2:
                self.avatar_candidate = {
                    "color": best_color,
                    "color_name": COLOR_NAMES.get(best_color, f"color-{best_color}"),
                    "observations": count,
                }
                self.confidence = count

    def get_avatar_info(self) -> str | None:
        """Return avatar description if confident enough."""
        if self.avatar_candidate and self.confidence >= 2:
            ac = self.avatar_candidate
            return (
                f"AVATAR DETECTED: The {ac['color_name']} object moves when you act "
                f"({ac['observations']} observations). This is likely what you control."
            )
        return None

    def get_action_map(self) -> dict[str, str]:
        """Return action→direction mapping based on avatar movements."""
        if not self.avatar_candidate:
            return {}

        avatar_color = self.avatar_candidate["color"]
        action_dirs: dict[str, list[tuple[int, int]]] = {}

        for entry in self.movement_log:
            action = entry["action"]
            for m in entry["movements"]:
                if m["color"] == avatar_color:
                    if action not in action_dirs:
                        action_dirs[action] = []
                    action_dirs[action].append((m["dx"], m["dy"]))

        result = {}
        for action, deltas in action_dirs.items():
            avg_dx = sum(d[0] for d in deltas) / len(deltas)
            avg_dy = sum(d[1] for d in deltas) / len(deltas)

            if abs(avg_dy) > abs(avg_dx):
                direction = "UP" if avg_dy < 0 else "DOWN"
            elif abs(avg_dx) > abs(avg_dy):
                direction = "LEFT" if avg_dx < 0 else "RIGHT"
            else:
                direction = f"dx={avg_dx:.0f},dy={avg_dy:.0f}"

            result[action] = f"moves avatar {direction} (avg delta: dx={avg_dx:.0f}, dy={avg_dy:.0f})"

        return result


class BarTracker:
    """Track horizontal bar-like structures that change monotonically.

    These are likely health/energy bars (resource cost) NOT progress.
    """

    def __init__(self):
        self.bar_history: list[dict] = []  # {step, bars: [{region, length, color}]}
        self.detected_bars: list[dict] = []

    def update(self, grid: np.ndarray, step: int) -> None:
        """Scan for horizontal bars and track their changes."""
        h, w = grid.shape
        bars = []

        # Scan each row for long horizontal runs of a single color
        for y in range(h):
            run_start = 0
            run_color = int(grid[y, 0])
            for x in range(1, w):
                cell = int(grid[y, x])
                if cell != run_color:
                    if x - run_start >= 8:  # Minimum bar length
                        bars.append({
                            "y": y,
                            "x_start": run_start,
                            "x_end": x - 1,
                            "length": x - run_start,
                            "color": run_color,
                            "color_name": COLOR_NAMES.get(run_color, f"color-{run_color}"),
                        })
                    run_start = x
                    run_color = cell
            # Check end of row
            if w - run_start >= 8:
                bars.append({
                    "y": y,
                    "x_start": run_start,
                    "x_end": w - 1,
                    "length": w - run_start,
                    "color": run_color,
                    "color_name": COLOR_NAMES.get(run_color, f"color-{run_color}"),
                })

        self.bar_history.append({"step": step, "bars": bars})
        self._detect_monotonic()

    def _detect_monotonic(self) -> None:
        """Detect bars that are shrinking monotonically (= resource bars)."""
        if len(self.bar_history) < 3:
            return

        # Group bars by approximate y-position and color
        bar_series: dict[tuple[int, int], list[int]] = {}
        for entry in self.bar_history:
            for bar in entry["bars"]:
                key = (bar["y"], bar["color"])
                if key not in bar_series:
                    bar_series[key] = []
                bar_series[key].append(bar["length"])

        self.detected_bars = []
        for (y, color), lengths in bar_series.items():
            if len(lengths) >= 3:
                # Check if monotonically decreasing
                is_decreasing = all(a >= b for a, b in zip(lengths, lengths[1:]))
                total_decrease = lengths[0] - lengths[-1]

                if is_decreasing and total_decrease > 0:
                    self.detected_bars.append({
                        "y": y,
                        "color": color,
                        "color_name": COLOR_NAMES.get(color, f"color-{color}"),
                        "initial_length": lengths[0],
                        "current_length": lengths[-1],
                        "decrease": total_decrease,
                        "type": "RESOURCE_COST",
                    })

    def get_bar_warnings(self) -> str | None:
        """Return warnings about detected resource bars."""
        if not self.detected_bars:
            return None

        lines = []
        for bar in self.detected_bars:
            lines.append(
                f"RESOURCE BAR at y={bar['y']}: {bar['color_name']} bar shrinking "
                f"({bar['initial_length']}->{bar['current_length']}px, lost {bar['decrease']}px). "
                f"This is a COST/HEALTH indicator, NOT progress! "
                f"Depleting it is BAD, not a goal."
            )
        return "\n".join(lines)
