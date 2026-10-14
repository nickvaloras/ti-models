"""Elo rating (online / one game at a time).

    E_i  = 1 / (1 + 10^(-(R_i - R_j)/400))       expected score (same logistic as BT)
    R_i <- R_i + K (S_i - E_i)                    S_i = 1 win, 0 loss

With binary outcomes, Elo is stochastic gradient ascent on the Bradley-Terry
log-likelihood, one game at a time, with learning rate K. So BT vs. Elo here is
purely "whole-season fit" vs. "incremental updating":

  * order-dependent: later games move ratings as much as early ones, and early
    games are judged against still-uninformed ratings (a recency bias), like
    trial-by-trial value-updating models of TI (Jensen et al.'s implicit
    value-updating / betasort family),
  * no explicit prior: shrinkage comes from starting everyone at 1500 and taking
    finitely many small steps (implicit regularization / early stopping),
  * passes > 1 replays the season (like repeated training blocks); with enough
    passes and a small K it approaches the unregularized BT fit.
"""
from __future__ import annotations

import numpy as np

from .base import RatingModel


class Elo(RatingModel):
    name = "Elo"

    def __init__(self, n_teams: int, k: float = 20.0, start: float = 1500.0,
                 passes: int = 1):
        super().__init__(n_teams)
        self.k, self.start, self.passes = k, start, passes
        self.log: list[dict] = []

    @staticmethod
    def expected(ri: float, rj: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(ri - rj) / 400))

    def fit(self, games) -> "Elo":
        games = sorted(games, key=lambda g: g.week)
        R = np.full(self.n, self.start, dtype=float)
        self.history, self.log = [R.copy()], []
        weeks = sorted({g.week for g in games})
        for p in range(self.passes):
            for wk in weeks:
                for g in (g for g in games if g.week == wk):
                    w, l = g.winner, g.loser
                    e_w = self.expected(R[w], R[l])
                    delta = self.k * (1.0 - e_w)
                    R[w] += delta
                    R[l] -= delta
                    self.log.append(dict(pass_=p, week=wk, winner=w, loser=l,
                                         expected=e_w, delta=delta))
                if p == self.passes - 1:
                    self.history.append(R.copy())
        self.ratings = R
        return self

    def pair_prob(self, r, i, j) -> float:
        return self.expected(r[i], r[j])

    def prob_matrix_from(self, r):
        P = 1.0 / (1.0 + 10 ** (-(r[:, None] - r[None, :]) / 400))
        np.fill_diagonal(P, np.nan)
        return P
