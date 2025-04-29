from __future__ import annotations

"""Pygame-based visualisation & basic key‑board controls for Learn2Slither.

This module is optional during headless training but mandatory for human‑readable
inspection / defence.  It renders one Board instance and lets you:
  • watch at realtime or accelerated FPS (\-speed flag, default 8)  
  • pause / resume with SPACE  
  • single‑step when paused with N  
  • quit with ESC

Keyboard control for manual testing (arrow keys) is also provided; when not in
"manual" mode, an external agent should call ``tick(action)``.
"""

import sys
import pygame
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

# local import guarded to avoid heavy deps when importing board logic alone
from snake_rl.env.board import Board, Cell, DIRECTIONS, Pos

CELL_PX = 40  # size of one cell in pixels
MARGIN = 2    # grid line thickness

COLORS = {
    Cell.EMPTY: (30, 30, 30),
    Cell.GREEN_APPLE: (0, 200, 0),
    Cell.RED_APPLE: (200, 0, 0),
    Cell.SNAKE_HEAD: (50, 150, 255),
    Cell.SNAKE_BODY: (30, 90, 200),
}
BG = (15, 15, 15)
GRID = (50, 50, 50)
TEXT = (220, 220, 220)

pygame.font.init()
FONT = pygame.font.SysFont("consolas", 16)

def _draw_board(screen: pygame.Surface, board: Board) -> None:
    size_px = board.size * (CELL_PX + MARGIN) + MARGIN
    screen.fill(BG)

    # cells
    for y in range(board.size):
        for x in range(board.size):
            rect = pygame.Rect(
                MARGIN + x * (CELL_PX + MARGIN),
                MARGIN + y * (CELL_PX + MARGIN),
                CELL_PX,
                CELL_PX,
            )
            cell = board._get_cell(Pos(x, y))
            color = COLORS.get(cell, BG)
            pygame.draw.rect(screen, color, rect)

    # grid lines (optional aesthetics)
    for i in range(board.size + 1):
        # vertical
        pygame.draw.line(
            screen,
            GRID,
            (MARGIN // 2 + i * (CELL_PX + MARGIN), 0),
            (MARGIN // 2 + i * (CELL_PX + MARGIN), size_px),
            1,
        )
        # horizontal
        pygame.draw.line(
            screen,
            GRID,
            (0, MARGIN // 2 + i * (CELL_PX + MARGIN)),
            (size_px, MARGIN // 2 + i * (CELL_PX + MARGIN)),
            1,
        )


def _overlay_text(screen: pygame.Surface, msg: str, x: int = 5, y: int = 5) -> None:
    surf = FONT.render(msg, True, TEXT)
    screen.blit(surf, (x, y))


class SnakeGameApp:
    """Standalone window for manual play / visualisation."""

    def __init__(self, board: Optional[Board] = None, fps: int = 8, manual: bool = False):
        self.board = board or Board()
        self.manual = manual
        self.fps_target = fps
        self.clock = pygame.time.Clock()

        size_px = self.board.size * (CELL_PX + MARGIN) + MARGIN
        self.screen = pygame.display.set_mode((size_px, size_px))
        pygame.display.set_caption("Learn2Slither")

        self.paused = False

    # ------------------------------------------------------------------
    #  Public loop helpers
    # ------------------------------------------------------------------
    def run(self):
        while True:
            action = self._handle_events()
            if not self.paused or action is not None:
                self._step(action)
            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps_target)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _handle_events(self) -> Optional[str]:
        action = None
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if ev.key == pygame.K_n and self.paused:
                    # step once when paused
                    self.paused = True  # keep paused
                    action = self._random_action()
                if self.manual:
                    if ev.key == pygame.K_UP:
                        action = "UP"
                    elif ev.key == pygame.K_DOWN:
                        action = "DOWN"
                    elif ev.key == pygame.K_LEFT:
                        action = "LEFT"
                    elif ev.key == pygame.K_RIGHT:
                        action = "RIGHT"
        return action

    def _random_action(self) -> str:
        return self.board.rng.choice(list(DIRECTIONS))

    def _step(self, action: Optional[str]):
        if action is None:
            action = self._random_action() if not self.manual else self.board.direction
        if not self.board.done:
            self.board.step(action)
        else:
            self.board.reset()

    def _draw(self):
        _draw_board(self.screen, self.board)
        _overlay_text(
            self.screen,
            f"Len: {len(self.board.snake)}  Done: {self.board.done}  FPS: {self.fps_target}",
        )


# ----------------------------------------------------------------------
#  CLI entry‑point: python -m snake_rl.ui.renderer --manual --speed 8
# ----------------------------------------------------------------------

def _cli():
    ap = ArgumentParser(description="Learn2Slither visualiser")
    ap.add_argument("--manual", action="store_true", help="Play yourself with arrow keys")
    ap.add_argument("--speed", type=int, default=8, help="Target FPS (visual speed)")
    args = ap.parse_args()

    pygame.init()
    app = SnakeGameApp(fps=args.speed, manual=args.manual)
    app.run()


if __name__ == "__main__":
    _cli()
