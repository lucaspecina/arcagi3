"""Exploration controller — deterministic policy layer.

Tracks avatar movement, visited regions, blocked transitions, and frontiers.
Provides action recommendations and vetoes to the LLM actor.
"""

from dataclasses import dataclass, field


# Discretize positions to ~5px grid cells for region tracking
REGION_SIZE = 5


def pos_to_region(x: float, y: float) -> tuple[int, int]:
    """Convert pixel position to region coordinate."""
    return (int(x) // REGION_SIZE, int(y) // REGION_SIZE)


@dataclass
class ExplorationController:
    """External exploration policy that guides the LLM actor."""

    # Position tracking
    avatar_positions: list[tuple[float, float]] = field(default_factory=list)
    visited_regions: set[tuple[int, int]] = field(default_factory=set)

    # Blocked transitions: (region, action) → True
    blocked: set[tuple[tuple[int, int], str]] = field(default_factory=set)

    # Action effects per region: (region, action) → (dx, dy)
    action_effects: dict[tuple[tuple[int, int], str], tuple[float, float]] = field(default_factory=dict)

    # Oscillation detection
    last_actions: list[str] = field(default_factory=list)

    # Co-moving entities: color → list of (action, dx, dy)
    entity_movements: dict[int, list[tuple[str, float, float]]] = field(default_factory=dict)
    co_movers: set[int] = field(default_factory=set)  # Colors that co-move with avatar
    avatar_color: int | None = None

    # Health tracking
    health_at_step: list[float] = field(default_factory=list)

    def update(
        self,
        action: str,
        avatar_pos: tuple[float, float] | None,
        prev_pos: tuple[float, float] | None,
        movements: list[dict],
        health_bar_length: float | None = None,
    ) -> None:
        """Update controller state after an action."""
        self.last_actions.append(action)
        if len(self.last_actions) > 20:
            self.last_actions = self.last_actions[-20:]

        if health_bar_length is not None:
            self.health_at_step.append(health_bar_length)

        if avatar_pos is None:
            return

        self.avatar_positions.append(avatar_pos)
        region = pos_to_region(*avatar_pos)
        self.visited_regions.add(region)

        # Check if movement happened
        if prev_pos is not None:
            dx = avatar_pos[0] - prev_pos[0]
            dy = avatar_pos[1] - prev_pos[1]
            prev_region = pos_to_region(*prev_pos)

            if abs(dx) < 1 and abs(dy) < 1:
                # No movement → blocked
                self.blocked.add((prev_region, action))
            else:
                self.action_effects[(prev_region, action)] = (dx, dy)

        # Track entity co-movement
        for m in movements:
            color = m["color"]
            if color not in self.entity_movements:
                self.entity_movements[color] = []
            self.entity_movements[color].append((action, m["dx"], m["dy"]))

        self._detect_co_movers()

    def _detect_co_movers(self) -> None:
        """Detect entities that always move with the avatar."""
        if len(self.avatar_positions) < 3:
            return

        # Find avatar color (most consistently moving entity)
        if self.avatar_color is None:
            best_color = None
            best_count = 0
            for color, moves in self.entity_movements.items():
                if len(moves) > best_count:
                    best_count = len(moves)
                    best_color = color
            if best_color is not None and best_count >= 2:
                self.avatar_color = best_color

        if self.avatar_color is None:
            return

        # Check which other colors ALWAYS move with same deltas as avatar
        avatar_moves = self.entity_movements.get(self.avatar_color, [])
        if len(avatar_moves) < 2:
            return

        avatar_by_action = {}
        for action, dx, dy in avatar_moves:
            avatar_by_action[action] = (dx, dy)

        for color, moves in self.entity_movements.items():
            if color == self.avatar_color:
                continue
            if len(moves) < 2:
                continue

            match_count = 0
            total = 0
            for action, dx, dy in moves:
                if action in avatar_by_action:
                    total += 1
                    a_dx, a_dy = avatar_by_action[action]
                    if abs(dx - a_dx) < 2 and abs(dy - a_dy) < 2:
                        match_count += 1

            if total >= 2 and match_count / total >= 0.8:
                self.co_movers.add(color)

    def get_current_region(self) -> tuple[int, int] | None:
        """Get the current region of the avatar."""
        if not self.avatar_positions:
            return None
        return pos_to_region(*self.avatar_positions[-1])

    def get_blocked_actions(self) -> list[str]:
        """Get actions that are blocked at the current position."""
        region = self.get_current_region()
        if region is None:
            return []
        return [action for (r, action) in self.blocked if r == region]

    def get_allowed_actions(self, available: list[str]) -> list[str]:
        """Filter available actions, removing blocked ones."""
        blocked = set(self.get_blocked_actions())
        return [a for a in available if a not in blocked]

    def detect_oscillation(self) -> bool:
        """Detect if agent is oscillating (A-B-A-B pattern)."""
        if len(self.last_actions) < 4:
            return False
        last4 = self.last_actions[-4:]
        return last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]

    def get_frontier_regions(self) -> list[tuple[int, int]]:
        """Get unexplored regions adjacent to visited regions."""
        frontiers = set()
        for rx, ry in self.visited_regions:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (rx + dx, ry + dy)
                if neighbor not in self.visited_regions:
                    # Basic bounds check (64x64 grid / REGION_SIZE)
                    max_r = 64 // REGION_SIZE
                    if 0 <= neighbor[0] < max_r and 0 <= neighbor[1] < max_r:
                        frontiers.add(neighbor)
        return sorted(frontiers)

    def suggest_direction(self) -> str | None:
        """Suggest a direction to explore based on frontiers and avatar position."""
        if not self.avatar_positions:
            return None

        region = self.get_current_region()
        if region is None:
            return None

        frontiers = self.get_frontier_regions()
        if not frontiers:
            return None

        # Find nearest frontier
        best = None
        best_dist = float("inf")
        for fx, fy in frontiers:
            dist = abs(fx - region[0]) + abs(fy - region[1])
            if dist < best_dist:
                best_dist = dist
                best = (fx, fy)

        if best is None:
            return None

        dx = best[0] - region[0]
        dy = best[1] - region[1]

        # Map to direction
        if abs(dy) > abs(dx):
            return "DOWN" if dy > 0 else "UP"
        else:
            return "RIGHT" if dx > 0 else "LEFT"

    def get_exploration_report(self) -> str:
        """Generate a concise exploration report for the LLM."""
        lines = []

        # Position info
        if self.avatar_positions:
            x, y = self.avatar_positions[-1]
            lines.append(f"Avatar position: ({x:.0f}, {y:.0f})")

        # Visited coverage
        total_possible = (64 // REGION_SIZE) ** 2
        visited_pct = round(100 * len(self.visited_regions) / total_possible, 1) if total_possible > 0 else 0
        lines.append(f"Explored: {len(self.visited_regions)} regions ({visited_pct}%)")

        # Blocked actions here
        blocked = self.get_blocked_actions()
        if blocked:
            lines.append(f"⛔ BLOCKED here: {', '.join(blocked)} (walls/obstacles)")

        # Oscillation warning
        if self.detect_oscillation():
            lines.append("⚠ OSCILLATION DETECTED! You are going back and forth. CHANGE DIRECTION!")

        # Frontier suggestion
        direction = self.suggest_direction()
        if direction:
            lines.append(f"→ SUGGESTED: explore {direction} (nearest unexplored area)")

        # Co-movers warning
        if self.co_movers:
            from .grid_utils import COLOR_NAMES
            names = [COLOR_NAMES.get(c, f"color-{c}") for c in self.co_movers]
            lines.append(
                f"⚠ CO-MOVERS: {', '.join(names)} objects move WITH your avatar. "
                f"They are NOT separate targets!"
            )

        # Health trend
        if len(self.health_at_step) >= 3:
            recent = self.health_at_step[-3:]
            if all(a >= b for a, b in zip(recent, recent[1:])):
                loss_per_step = (recent[0] - recent[-1]) / len(recent)
                remaining = recent[-1]
                steps_left = int(remaining / max(loss_per_step, 0.1))
                lines.append(f"⚠ HEALTH DECLINING: ~{loss_per_step:.0f}px/step, ~{steps_left} steps left before empty")

        return "\n".join(lines)
