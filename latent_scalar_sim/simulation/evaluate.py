"""Evaluation: how well do the models infer never-played matchups?

The TI framing:
  train pairs = matchups played in weeks 1..10
  test pairs  = the 40 matchups never played; week 11 plays 8 of them
  symbolic distance = |true_rank_i - true_rank_j|

Signatures to look for:
  * symbolic distance effect -- P(stronger team wins) and accuracy rise with
    distance on test pairs, even though those pairs were never observed,
  * calibration -- are the models as confident as the true probabilities?
  * learning curves -- how fast do ratings converge on the hidden order?
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .simulate import Season, simulate_future, simulate_season
from .teams import ABBRS


def _acc(p):
    """Share picking the truly stronger team; a 50/50 call counts as half."""
    p = np.asarray(p)
    return float(np.mean((p > 0.5) + 0.5 * (p == 0.5)))


def pair_table(season: Season, models: dict) -> pd.DataFrame:
    """One row per unordered pair, oriented so `b` (better) is the truly stronger team."""
    played = season.played_pairs()
    wk11 = {frozenset((g.a, g.b)) for g in season.test_week}
    mats = {k: m.prob_matrix() for k, m in models.items()}
    rows = []
    for i, j in combinations(range(season.n), 2):
        b, w = (i, j) if season.strengths[i] > season.strengths[j] else (j, i)
        key = frozenset((i, j))
        row = dict(better=ABBRS[b], worse=ABBRS[w], b=b, w=w,
                   distance=int(abs(season.true_rank[b] - season.true_rank[w])),
                   split="train" if key in played else "test",
                   week11=key in wk11, p_true=season.true_prob(b, w))
        for k, P in mats.items():
            row[f"p_{k}"] = P[b, w]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(season: Season, models: dict, n_future: int = 200, seed: int = 0):
    """Per-model metrics on train vs test pairs.

    acc     : picks the truly stronger team
    mae_p   : |P_model - P_true|
    logloss : on n_future freshly simulated games per pair (held-out outcomes)
    """
    df = pair_table(season, models)
    out = []
    for split in ("train", "test"):
        d = df[df.split == split]
        fut = simulate_future(season, list(zip(d.b, d.w)), n_sims=n_future, seed=seed)
        for name, m in models.items():
            p = np.array([m.prob(g.winner, g.loser) for g in fut])
            out.append(dict(model=name, split=split, n_pairs=len(d),
                            acc=_acc(d[f"p_{name}"]),
                            mae_p=float((d[f"p_{name}"] - d.p_true).abs().mean()),
                            logloss=float(-np.log(np.clip(p, 1e-9, 1)).mean())))
        p = np.array([season.true_prob(g.winner, g.loser) for g in fut])
        out.append(dict(model="True", split=split, n_pairs=len(d), acc=1.0,
                        mae_p=0.0, logloss=float(-np.log(p).mean())))
    return pd.DataFrame(out)


def week11_table(season: Season, models: dict) -> pd.DataFrame:
    """The TI test week: predictions vs. the simulated result of each game."""
    rows = []
    for g in season.test_week:
        b, w = (g.a, g.b) if season.strengths[g.a] > season.strengths[g.b] else (g.b, g.a)
        row = dict(better=ABBRS[b], worse=ABBRS[w],
                   distance=int(abs(season.true_rank[b] - season.true_rank[w])),
                   p_true=season.true_prob(b, w), winner=ABBRS[g.winner],
                   upset=g.winner == w)
        for k, m in models.items():
            row[f"p_{k}"] = m.prob(b, w)
            row[f"hit_{k}"] = (m.prob(g.winner, g.loser) > 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_agreement(season: Season, models: dict) -> pd.DataFrame:
    rows = [dict(model=k, spearman=spearmanr(m.ratings, season.strengths)[0],
                 kendall=kendalltau(m.ratings, season.strengths)[0])
            for k, m in models.items()]
    rs = season.record_score()
    rows.append(dict(model="Record", spearman=spearmanr(rs, season.strengths)[0],
                     kendall=kendalltau(rs, season.strengths)[0]))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# replication across many seasons
# -----------------------------------------------------------------------------
def replicate(model_factories: dict[str, Callable], n_seasons: int = 300,
              seed0: int = 1000, **season_kwargs):
    """Run many independent seasons.

    Returns
      pairs    : stacked end-of-season pair tables (one row per pair per season)
      learning : per season x week x model -- rank correlation with the truth and
                 accuracy / |P - P_true| on that season's fixed test pairs
    """
    pair_frames, learn_rows = [], []
    for r in range(n_seasons):
        s = simulate_season(seed=seed0 + r, **season_kwargs)
        models = {k: f(s.n).fit(s.games) for k, f in model_factories.items()}
        df = pair_table(s, models)
        df["season"] = r
        pair_frames.append(df)

        test = df[df.split == "test"]
        tb, tw = test.b.to_numpy(), test.w.to_numpy()
        pt = test.p_true.to_numpy()
        for k, m in models.items():
            for wk, rat in enumerate(m.history):
                P = m.prob_matrix_from(rat)
                p = P[tb, tw]
                rho = 0.0 if np.allclose(rat, rat[0]) else spearmanr(rat, s.strengths)[0]
                learn_rows.append(dict(season=r, week=wk, model=k, spearman=rho,
                                       acc_test=_acc(p), mae_test=float(np.abs(p - pt).mean())))
        wins = np.zeros(s.n)
        learn_rows.append(dict(season=r, week=0, model="Record", spearman=0.0))
        for wk in range(1, s.weeks + 1):
            for g in s.games:
                if g.week == wk:
                    wins[g.winner] += 1
            learn_rows.append(dict(season=r, week=wk, model="Record",
                                   spearman=spearmanr(wins, s.strengths)[0]))
    return pd.concat(pair_frames, ignore_index=True), pd.DataFrame(learn_rows)


def distance_curve(pairs: pd.DataFrame, model_names, split: str | None = "test"):
    """Mean P(stronger wins) and accuracy by symbolic distance."""
    d = pairs if split is None else pairs[pairs.split == split]
    g = d.groupby("distance")
    out = g[["p_true"] + [f"p_{n}" for n in model_names]].mean()
    for n in model_names:
        out[f"acc_{n}"] = g[f"p_{n}"].apply(_acc)
    out["n"] = g.size()
    return out.reset_index()


def learning_curve(learning: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% interval (across seasons) per week and model."""
    def q(x, a):
        return np.nanquantile(x, a)
    rows = []
    for (m, wk), d in learning.groupby(["model", "week"]):
        row = dict(model=m, week=wk)
        for col in ("spearman", "acc_test", "mae_test"):
            if col in d and d[col].notna().any():
                x = d[col].dropna().to_numpy()
                se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
                row[col] = x.mean()
                row[f"{col}_lo"], row[f"{col}_hi"] = x.mean() - 1.96 * se, x.mean() + 1.96 * se
                row[f"{col}_q10"], row[f"{col}_q90"] = q(x, .1), q(x, .9)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration(pairs: pd.DataFrame, model_names, split: str = "test",
                bins: int = 10) -> pd.DataFrame:
    """Reliability table: bin by predicted P, compare with the true P.

    Both orientations of every pair are used, so the table is symmetric about 0.5.
    """
    d = pairs[pairs.split == split]
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for n in model_names:
        pm = np.concatenate([d[f"p_{n}"], 1 - d[f"p_{n}"]])
        pt = np.concatenate([d.p_true, 1 - d.p_true])
        idx = np.clip(np.digitize(pm, edges) - 1, 0, bins - 1)
        for k in range(bins):
            sel = idx == k
            if sel.sum() == 0:
                continue
            rows.append(dict(model=n, bin=k, p_pred=pm[sel].mean(),
                             p_true=pt[sel].mean(), n=int(sel.sum())))
    return pd.DataFrame(rows)
