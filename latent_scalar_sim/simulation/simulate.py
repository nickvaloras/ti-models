"""Season simulator: binary win/loss only.

Generative model (the hidden "world"):

    true strength   s_i   evenly spaced, s_0 (NE) highest ... s_15 (CLE) lowest
    P(i beats j)  = sigmoid(s_i - s_j)

This is exactly the Bradley-Terry / Elo likelihood, so both models are correctly
specified; any difference between them comes from *how they learn*
(whole-season fit vs. one game at a time), not from a wrong world model.

With the default spread of 3.0 logits the best team beats an average opponent
about 80% of the time (~8-2 over 10 games) and adjacent teams are near coin
flips (P = 0.55) -- roughly NFL parity.

Schedule: weeks 1..T are perfect matchings of the 16 teams (8 games/week), no
repeated matchups. Week T+1 is the *transitive-inference test week*: another
perfect matching drawn only from pairs that never met in weeks 1..T. It is
built together with the season so it always exists.

tau controls schedule locality: opponents are sampled with weight
exp(-(d - 1) / tau), where d is the true rank distance. tau = inf (default)
gives a uniformly random schedule; small tau favors adjacent "premise" pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .teams import N_TEAMS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Game:
    week: int
    a: int
    b: int
    winner: int

    @property
    def loser(self) -> int:
        return self.b if self.winner == self.a else self.a


@dataclass
class Season:
    strengths: np.ndarray            # hidden true strengths (logits)
    true_rank: np.ndarray            # true_rank[i] = 0 for the best team
    games: list[Game]                # training games, weeks 1..weeks
    test_week: list[Game]            # week weeks+1: never-played matchups
    weeks: int
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.strengths)

    # --- ground truth ----------------------------------------------------------
    def true_prob(self, i: int, j: int) -> float:
        return float(sigmoid(self.strengths[i] - self.strengths[j]))

    def true_prob_matrix(self) -> np.ndarray:
        P = sigmoid(self.strengths[:, None] - self.strengths[None, :])
        np.fill_diagonal(P, np.nan)
        return P

    # --- data views ------------------------------------------------------------
    def played_pairs(self, through_week: int | None = None) -> set[frozenset]:
        w = self.weeks if through_week is None else through_week
        return {frozenset((g.a, g.b)) for g in self.games if g.week <= w}

    def unplayed_pairs(self) -> list[tuple[int, int]]:
        played = self.played_pairs()
        return [(i, j) for i, j in combinations(range(self.n), 2)
                if frozenset((i, j)) not in played]

    def records(self) -> np.ndarray:
        """(n, 2) array of wins, losses over the training weeks."""
        rec = np.zeros((self.n, 2), dtype=int)
        for g in self.games:
            rec[g.winner, 0] += 1
            rec[g.loser, 1] += 1
        return rec

    def record_score(self) -> np.ndarray:
        """Wins, tie-broken by strength of schedule (opponents' total wins).
        A model-free baseline ranking."""
        rec = self.records()
        sos = np.zeros(self.n)
        for g in self.games:
            sos[g.a] += rec[g.b, 0]
            sos[g.b] += rec[g.a, 0]
        return rec[:, 0] + 1e-3 * sos


# -----------------------------------------------------------------------------
def make_strengths(n: int = N_TEAMS, spread: float = 3.0, jitter: float = 0.0,
                   rng=None) -> np.ndarray:
    """Evenly spaced strengths from +spread/2 (best) to -spread/2 (worst)."""
    rng = np.random.default_rng(rng)
    s = np.linspace(spread / 2, -spread / 2, n)
    if jitter:
        s = np.sort(s + rng.normal(0, jitter, n))[::-1]
    return s - s.mean()


def _sample_matching(available: set[frozenset], rank: np.ndarray, tau: float,
                     rng: np.random.Generator) -> list[tuple[int, int]] | None:
    """Randomized backtracking search for one perfect matching."""
    def solve(unmatched: list[int]):
        if not unmatched:
            return []
        a = unmatched[0]
        cands = [b for b in unmatched[1:] if frozenset((a, b)) in available]
        if not cands:
            return None
        d = np.array([abs(rank[a] - rank[b]) for b in cands], dtype=float)
        w = np.ones_like(d) if np.isinf(tau) else np.exp(-(d - 1) / tau)
        for k in rng.choice(len(cands), size=len(cands), replace=False, p=w / w.sum()):
            b = cands[k]
            rest = solve([u for u in unmatched if u not in (a, b)])
            if rest is not None:
                return [(a, b)] + rest
        return None

    return solve(list(rng.permutation(len(rank))))


def make_schedule(rank: np.ndarray, weeks: int = 10, tau: float = np.inf,
                  rng=None, max_tries: int = 5000) -> list[list[tuple[int, int]]]:
    """`weeks` training matchings + 1 test matching, no pair repeated."""
    rng = np.random.default_rng(rng)
    n = len(rank)
    for _ in range(max_tries):
        available = {frozenset(p) for p in combinations(range(n), 2)}
        sched = []
        for w in range(weeks + 1):
            # the test week is drawn uniformly from whatever is left
            m = _sample_matching(available, rank, np.inf if w == weeks else tau, rng)
            if m is None:
                break
            sched.append(m)
            available -= {frozenset(p) for p in m}
        if len(sched) == weeks + 1:
            return sched
    raise RuntimeError("Could not build a schedule; try a larger tau or fewer weeks.")


def play(week: int, a: int, b: int, s: np.ndarray, rng) -> Game:
    return Game(week, a, b, a if rng.random() < sigmoid(s[a] - s[b]) else b)


def simulate_season(weeks: int = 10, spread: float = 3.0, jitter: float = 0.0,
                    tau: float = np.inf, n_teams: int = N_TEAMS,
                    seed: int | None = 7) -> Season:
    rng = np.random.default_rng(seed)
    s = make_strengths(n_teams, spread=spread, jitter=jitter, rng=rng)
    true_rank = np.argsort(np.argsort(-s))
    sched = make_schedule(true_rank, weeks=weeks, tau=tau, rng=rng)
    games = [play(w + 1, a, b, s, rng) for w, wk in enumerate(sched[:-1]) for a, b in wk]
    test = [play(weeks + 1, a, b, s, rng) for a, b in sched[-1]]
    return Season(strengths=s, true_rank=true_rank, games=games, test_week=test,
                  weeks=weeks, params=dict(weeks=weeks, spread=spread, jitter=jitter,
                                           tau=tau, seed=seed))


def simulate_future(season: Season, pairs, n_sims: int = 1, seed=None) -> list[Game]:
    """Extra games between `pairs` in the same hidden world (held-out outcomes)."""
    rng = np.random.default_rng(seed)
    return [play(season.weeks + 1, a, b, season.strengths, rng)
            for _ in range(n_sims) for a, b in pairs]
