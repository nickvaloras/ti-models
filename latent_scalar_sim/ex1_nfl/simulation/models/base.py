"""Shared interface for rating-based transitive-inference models.

Every model here learns ONE scalar per item (a rank / value / rating) and
predicts a pairwise choice from the difference of the two scalars -- the
"additive representation" case in Lippl et al. (2024). Such models are
transitive by construction, so inference on never-played pairs comes for free.

New models (betasort, RL value-updating, networks with conjunctive units, ...)
subclass RatingModel and implement fit() and pair_prob(). A model that is not
a single scalar per item can still fit this interface by overriding
prob_matrix_from() and storing whatever state it needs in `history`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

ELO_SCALE = 400 / np.log(10)   # natural-log BT units -> Elo points


class RatingModel(ABC):
    name: str = "model"

    def __init__(self, n_teams: int):
        self.n = n_teams
        self.ratings = np.zeros(n_teams)
        # history[k] = ratings after week k (row 0 = before any games)
        self.history: list[np.ndarray] = []

    @abstractmethod
    def fit(self, games) -> "RatingModel":
        ...

    @abstractmethod
    def pair_prob(self, r: np.ndarray, i: int, j: int) -> float:
        """P(i beats j) under rating vector r."""

    # -- convenience -----------------------------------------------------------
    def prob(self, i: int, j: int) -> float:
        return self.pair_prob(self.ratings, i, j)

    def prob_matrix_from(self, r: np.ndarray) -> np.ndarray:
        P = np.array([[self.pair_prob(r, i, j) for j in range(self.n)]
                      for i in range(self.n)], dtype=float)
        np.fill_diagonal(P, np.nan)
        return P

    def prob_matrix(self) -> np.ndarray:
        return self.prob_matrix_from(self.ratings)

    def elo_scale(self, r: np.ndarray | None = None) -> np.ndarray:
        """Ratings on a common Elo-like scale (mean 1500) for side-by-side plots."""
        return self.ratings if r is None else r

    def history_elo_scale(self) -> np.ndarray:
        return np.array([self.elo_scale(r) for r in self.history])

    def rank_of(self) -> np.ndarray:
        """rank_of[i] = 0 for the model's top team."""
        return np.argsort(np.argsort(-self.ratings))
