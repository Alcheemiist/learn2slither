from __future__ import annotations

"""State encoder transforming board vision into a hashable tuple usable by the
Q‑table.  The encoder is strictly limited to the four cardinal observations
returned by ``Board._vision()`` so we remain compliant with spec (no peeking at
full grid)."""

from enum import IntEnum
from typing import Dict, Tuple, Sequence

from snake_rl.env.board import Cell, Board


class Enc(IntEnum):
    WALL = 0
    GREEN = 1
    RED = 2
    BODY = 3  # snake body segment (includes head except when we shift it)
    EMPTY = 4


# Mapping from raw Cell to compact integer code
CELL2ENC = {
    Cell.WALL: Enc.WALL,
    Cell.GREEN_APPLE: Enc.GREEN,
    Cell.RED_APPLE: Enc.RED,
    Cell.SNAKE_HEAD: Enc.BODY,  # treat head as obstacle
    Cell.SNAKE_BODY: Enc.BODY,
    Cell.EMPTY: Enc.EMPTY,
}

# Fixed direction order ensures stable keys
_DIR_ORDER: Sequence[str] = ("UP", "LEFT", "DOWN", "RIGHT")


def encode_state(board: Board) -> Tuple[int, int, int, int]:
    """Return 4‑int tuple representing first object in each direction.

    Example: (0, 4, 1, 3) means WALL ahead, EMPTY left, GREEN below, BODY right.
    The returned tuple is directly usable as a dictionary key in a Q‑table.
    """

    vision: Dict[str, Cell] = board._vision()  # allowed helper
    return tuple(int(CELL2ENC[vision[d]]) for d in _DIR_ORDER)  # type: ignore


# ---------------------------------------------------------------------------
#  Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    b = Board(seed=0)
    print(b.render_ascii())
    print("Encoded state:", encode_state(b))
