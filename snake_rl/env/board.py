from __future__ import annotations

import random
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Dict


class Cell(Enum):
    """Enumeration of all possible cell contents on the board."""

    EMPTY = auto()
    WALL = auto()  # virtual; returned by _get_cell when out‑of‑bounds
    GREEN_APPLE = auto()
    RED_APPLE = auto()
    SNAKE_HEAD = auto()
    SNAKE_BODY = auto()


@dataclass(frozen=True)
class Pos:
    """Grid coordinate (x, y) with helper addition."""

    x: int
    y: int

    def __add__(self, other: Tuple[int, int]) -> "Pos":
        dx, dy = other
        return Pos(self.x + dx, self.y + dy)


# Cardinal direction deltas (x, y)
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# Game constants
BOARD_SIZE = 10
GREEN_APPLE_COUNT = 2
RED_APPLE_COUNT = 1
START_LENGTH = 3


class Board:
    """Environment logic (no rendering). Provides reset() and step()."""

    def __init__(self, size: int = BOARD_SIZE, seed: int | None = None):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    # -----------------------------------------------------------
    #  Initialization helpers
    # -----------------------------------------------------------
    def reset(self) -> None:
        """Reset board, spawn apples & snake, clear state."""
        self.grid: List[List[Cell]] = [
            [Cell.EMPTY for _ in range(self.size)] for _ in range(self.size)
        ]
        self.snake: List[Pos] = self._spawn_snake()
        for p in self.snake:
            self._set_cell(p, Cell.SNAKE_BODY)
        self._set_cell(self.snake[0], Cell.SNAKE_HEAD)

        for _ in range(GREEN_APPLE_COUNT):
            self._spawn_item(Cell.GREEN_APPLE)
        for _ in range(RED_APPLE_COUNT):
            self._spawn_item(Cell.RED_APPLE)

        self.direction: str = "RIGHT"  # arbitrary start dir
        self.done = False
        self.length_target = 10

    def _spawn_snake(self) -> List[Pos]:
        """Random contiguous 3‑cell snake (horizontal or vertical)."""
        horizontal = self.rng.choice([True, False])
        if horizontal:
            row = self.rng.randrange(self.size)
            col = self.rng.randrange(self.size - START_LENGTH)
            return [Pos(col + i, row) for i in reversed(range(START_LENGTH))]
        col = self.rng.randrange(self.size)
        row = self.rng.randrange(self.size - START_LENGTH)
        return [Pos(col, row + i) for i in reversed(range(START_LENGTH))]

    def _spawn_item(self, cell_type: Cell) -> None:
        empty = [
            Pos(x, y)
            for x in range(self.size)
            for y in range(self.size)
            if self.grid[y][x] == Cell.EMPTY
        ]
        self._set_cell(self.rng.choice(empty), cell_type)

    # -----------------------------------------------------------
    #  Internal cell helpers
    # -----------------------------------------------------------
    def _get_cell(self, pos: Pos) -> Cell:
        if not (0 <= pos.x < self.size and 0 <= pos.y < self.size):
            return Cell.WALL
        return self.grid[pos.y][pos.x]

    def _set_cell(self, pos: Pos, cell_type: Cell) -> None:
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.grid[pos.y][pos.x] = cell_type

    # -----------------------------------------------------------
    #  Public API – RL‑style
    # -----------------------------------------------------------
    def step(self, action: str):
        """Advance one tick. Returns (state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Game finished; call reset().")
        if action not in DIRECTIONS:
            raise ValueError(f"Invalid action {action}")

        self.direction = action
        next_head = self.snake[0] + DIRECTIONS[action]
        target_cell = self._get_cell(next_head)

        # collision handling
        if target_cell in {Cell.WALL, Cell.SNAKE_BODY}:
            self.done = True
            return self._vision(), -5, True, {"event": "death"}

        # Move snake head
        self.snake.insert(0, next_head)
        self._set_cell(next_head, Cell.SNAKE_HEAD)
        # Turn previous head into body segment
        self._set_cell(self.snake[1], Cell.SNAKE_BODY)

        reward = -0.05  # default small penalty for time pressure
        if target_cell == Cell.GREEN_APPLE:
            reward = 1
            self._spawn_item(Cell.GREEN_APPLE)
        elif target_cell == Cell.RED_APPLE:
            reward = -1
            # additional shrink of 1 (total ‑1 length)
            self.snake.pop()
            if len(self.snake) == 0:
                self.done = True
        else:
            # ordinary move → remove last tail segment
            tail = self.snake.pop()
            self._set_cell(tail, Cell.EMPTY)

        if len(self.snake) >= self.length_target:
            reward += 2  # bonus for hitting goal length

        return self._vision(), reward, self.done, {}

    # -----------------------------------------------------------
    #  Snake vision (state abstraction)
    # -----------------------------------------------------------
    def _vision(self) -> Dict[str, Cell]:
        """Return first non‑empty object along each cardinal ray."""

        def look(dir_name: str) -> Cell:
            delta = DIRECTIONS[dir_name]
            pos = self.snake[0] + delta
            while True:
                cell = self._get_cell(pos)
                if cell == Cell.EMPTY:
                    pos = pos + delta
                    continue
                return cell

        return {d: look(d) for d in DIRECTIONS}

    # -----------------------------------------------------------
    #  Debug helpers
    # -----------------------------------------------------------
    def render_ascii(self) -> str:
        symbols = {
            Cell.EMPTY: " ",
            Cell.WALL: "#",
            Cell.GREEN_APPLE: "G",
            Cell.RED_APPLE: "R",
            Cell.SNAKE_HEAD: "H",
            Cell.SNAKE_BODY: "S",
        }
        return "\n".join(
            "".join(symbols[self._get_cell(Pos(x, y))] for x in range(self.size))
            for y in range(self.size)
        )


# ---------------------------------------------------------------------------
#  Manual quick‑run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    board = Board(seed=42)
    print(board.render_ascii())
    done = False
    while not done:
        action = board.rng.choice(list(DIRECTIONS))
        state, reward, done, info = board.step(action)
        print(f"\nAction: {action}  Reward: {reward}")
        print(board.render_ascii())
