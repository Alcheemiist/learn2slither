from __future__ import annotations

"""Training loop + CLI flags.

Usage examples
--------------
# Train for 100 sessions headless, auto‑save checkpoint
python -m snake_rl.trainer.run --sessions 100 --save models/100.txt --visual off

# Evaluate a trained model for 10 games with learning OFF and step‑through
python -m snake_rl.trainer.run --load models/100.txt --sessions 10 --dontlearn --visual on --step
"""

import argparse
from pathlib import Path

from snake_rl.env.board import Board, DIRECTIONS
from snake_rl.agent.vision import encode_state
from snake_rl.agent.q_table import QTableAgent
from snake_rl.ui.renderer import SnakeGameApp


class Runner:
    def __init__(
        self,
        sessions: int,
        visual: bool,
        step_mode: bool,
        save_path: Path | None,
        load_path: Path | None,
        dontlearn: bool,
    ) -> None:
        self.sessions = sessions
        self.visual = visual
        self.step_mode = step_mode
        self.save_path = save_path
        self.load_path = load_path
        self.dontlearn = dontlearn

        self.board = Board()
        self.agent = QTableAgent(actions=list(DIRECTIONS))
        if self.load_path:
            self.agent = QTableAgent.load(self.load_path, list(DIRECTIONS))
            print(f"Loaded model from {self.load_path}")
            if self.dontlearn:
                self.agent.epsilon = 0.0

        self.app = None
        if self.visual:
            from pygame import init as pginit

            pginit()
            self.app = SnakeGameApp(self.board, fps=6, manual=False)

    # --------------------------------------------------------------
    def run(self):
        lengths = []
        durations = []
        for ep in range(self.sessions):
            total_steps = 0
            while not self.board.done:
                state = encode_state(self.board)
                a = self.agent.choose_action(state, explore=not self.dontlearn)
                next_state, reward, done, _ = self.board.step(a)
                if not self.dontlearn:
                    self.agent.learn(state, a, reward, next_state, done)
                total_steps += 1
                if self.visual:
                    self.app._draw()
                    if self.step_mode:
                        self.app.paused = True  # force pause until key
                    self.app._handle_events()
                    self.app.clock.tick(self.app.fps_target)
            lengths.append(len(self.board.snake))
            durations.append(total_steps)
            print(
                f"Session {ep+1}/{self.sessions} → length={lengths[-1]} steps={durations[-1]}"
            )
            self.board.reset()
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent.save(self.save_path)
            print(f"Saved model to {self.save_path}")


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Learn2Slither trainer/evaluator")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--load", type=Path)
    ap.add_argument("--visual", choices=["on", "off"], default="off")
    ap.add_argument("--dontlearn", action="store_true")
    ap.add_argument("--step", action="store_true", help="Step‑by‑step mode when visual on")
    args = ap.parse_args()

    runner = Runner(
        sessions=args.sessions,
        visual=args.visual == "on",
        step_mode=args.step,
        save_path=args.save,
        load_path=args.load,
        dontlearn=args.dontlearn,
    )
    runner.run()


if __name__ == "__main__":
    main()
