"""Bradley-Terry model (batch / whole-season).

    P(i beats j) = sigmoid(theta_i - theta_j)

Fit by maximum a posteriori: win/loss log-likelihood plus a Gaussian prior
theta ~ N(0, prior_sd^2). The prior matters: with 10 games per team an
undefeated or winless team has an MLE of +/- infinity (the same divergence a
strict, noiseless TI hierarchy produces). The prior is the simplest form of the
"norm minimization" inductive bias in Lippl et al. (2024). The default
prior_sd = 1.0 is close to the spread of the simulated true strengths (sd ~0.92).

Uncertainty: Laplace approximation, cov = inverse negative Hessian at the MAP.

BT uses all games jointly, so the order of games does not matter (unlike Elo).
`history` holds the MAP refit on all games through each week.
"""
from __future__ import annotations

import numpy as np

from .base import ELO_SCALE, RatingModel


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BradleyTerry(RatingModel):
    name = "Bradley-Terry"

    def __init__(self, n_teams: int, prior_sd: float = 1.0, tol: float = 1e-10,
                 max_iter: int = 100):
        super().__init__(n_teams)
        self.prior_sd, self.tol, self.max_iter = prior_sd, tol, max_iter
        self.cov = np.eye(n_teams) * prior_sd ** 2
        self.cov_history: list[np.ndarray] = []

    def _fit_once(self, winners: np.ndarray, losers: np.ndarray):
        n, lam = self.n, 1.0 / self.prior_sd ** 2
        theta = np.zeros(n)
        for _ in range(self.max_iter):
            p = _sigmoid(theta[winners] - theta[losers])
            r = 1.0 - p
            grad = -lam * theta
            np.add.at(grad, winners, r)
            np.add.at(grad, losers, -r)
            w = p * (1 - p)
            H = -lam * np.eye(n)
            np.add.at(H, (winners, winners), -w)
            np.add.at(H, (losers, losers), -w)
            np.add.at(H, (winners, losers), w)
            np.add.at(H, (losers, winners), w)
            step = np.linalg.solve(H, grad)
            theta = theta - step
            if np.max(np.abs(step)) < self.tol:
                break
        return theta, np.linalg.inv(-H)

    def fit(self, games) -> "BradleyTerry":
        games = list(games)
        self.history = [np.zeros(self.n)]
        self.cov_history = [np.eye(self.n) * self.prior_sd ** 2]
        for wk in sorted({g.week for g in games}):
            seen = [g for g in games if g.week <= wk]
            theta, cov = self._fit_once(np.array([g.winner for g in seen]),
                                        np.array([g.loser for g in seen]))
            self.history.append(theta)
            self.cov_history.append(cov)
        self.ratings, self.cov = self.history[-1], self.cov_history[-1]
        return self

    def pair_prob(self, r, i, j) -> float:
        return float(_sigmoid(r[i] - r[j]))

    def prob_matrix_from(self, r):
        P = _sigmoid(r[:, None] - r[None, :])
        np.fill_diagonal(P, np.nan)
        return P

    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def elo_scale(self, r=None) -> np.ndarray:
        r = self.ratings if r is None else r
        return 1500 + ELO_SCALE * r
