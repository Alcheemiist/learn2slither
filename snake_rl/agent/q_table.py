from __future__ import annotations

"""Simple ε‑greedy Q‑table agent suitable for discrete state/action spaces.

The agent is *environment‑agnostic*: it expects caller to supply hashable
`state` (we use the encoded 4‑tuple) and a list/tuple of valid action names
(`UP`, `LEFT`, ...).  Q‑values live in a nested dict:  Q[s][a] -> float.

Save/load is JSON for human readability.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

import random


class QTableAgent:
    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.actions = actions
        self.alpha = alpha  # learning‑rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # Q‑table: state -> action -> value
        self.Q: Dict[Any, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

    # ------------------------------------------------------------------
    #  Policy
    # ------------------------------------------------------------------
    def choose_action(self, state: Any, explore: bool = True) -> str:
        """Return an action according to ε‑greedy policy."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        # exploitation – pick argmax (ties broken randomly)
        q_vals = self.Q[state]
        max_q = max(q_vals.values())
        best = [a for a, q in q_vals.items() if q == max_q]
        return self.rng.choice(best)

    # ------------------------------------------------------------------
    #  Learning update
    # ------------------------------------------------------------------
    def learn(self, state, action: str, reward: float, next_state, done: bool):
        q_sa = self.Q[state][action]
        max_next = 0 if done else max(self.Q[next_state].values())
        td_target = reward + self.gamma * max_next
        self.Q[state][action] = q_sa + self.alpha * (td_target - q_sa)

        # Handle ε decay after each step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    #  Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        data = {
            "Q": self.Q,
            "meta": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        # Need to convert keys to strings for JSON (tuples -> repr)
        serialisable = {str(k): v for k, v in self.Q.items()}
        data["Q"] = serialisable
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path, actions: List[str]):
        raw = json.loads(Path(path).read_text())
        agent = cls(actions, **raw["meta"])
        # eval string tuple keys back to Python tuples
        agent.Q = defaultdict(
            lambda: {a: 0.0 for a in actions},
            {eval(k): v for k, v in raw["Q"].items()},
        )
        return agent
