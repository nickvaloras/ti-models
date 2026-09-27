"""Bradley-Terry vs. Elo on a simulated 16-team, 10-week season (binary W/L),
with week 11 as the transitive-inference test week.

    python run_nfl.py                     # defaults
    python run_nfl.py --seed 3            # another season
    python run_nfl.py --spread 5          # less parity (easier)
    python run_nfl.py --tau 1             # adjacent-heavy (TI-style) schedule
    python run_nfl.py --k 40 --passes 3   # hotter Elo, replay the season 3x
    python run_nfl.py --reps 1000         # more seasons for the summary figures

Outputs (figures/ and results/ next to this file):
    01_results_matrix.png     played (train) vs never-played (test) matchups
    02_rankings_bump.png      true order vs record vs BT vs Elo
    03_ratings.png            ratings on one Elo scale (+ BT uncertainty)
    04_trajectories.png       week-by-week ratings, batch vs online
    05_prob_matrices.png      P(row beats col) for every pair
    06_symbolic_distance.png  distance effect, many seasons
    07_test_week.png          week-11 predictions for never-played matchups
    08_learning_curves.png    rank recovery / accuracy / probability error by week
    09_calibration.png        predicted vs true probability
    results/*.csv             games, ratings, pair predictions, metrics, curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import plots
from simulation.evaluate import (calibration, distance_curve, learning_curve,
                                 pair_table, rank_agreement, replicate, summarize,
                                 week11_table)
from simulation.models import BradleyTerry, Elo
from simulation.simulate import simulate_season
from simulation.teams import ABBRS

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weeks", type=int, default=10, help="training weeks")
    ap.add_argument("--spread", type=float, default=3.0,
                    help="best-to-worst true strength gap (logits)")
    ap.add_argument("--tau", type=float, default=float("inf"),
                    help="schedule locality; inf = random, small = adjacent-heavy")
    ap.add_argument("--prior-sd", type=float, default=1.0, help="BT Gaussian prior sd")
    ap.add_argument("--k", type=float, default=20.0, help="Elo K-factor")
    ap.add_argument("--passes", type=int, default=1, help="Elo passes over the season")
    ap.add_argument("--reps", type=int, default=300, help="seasons for summary figures")
    ap.add_argument("--out", type=Path, default=HERE)
    return ap.parse_args()


def main():
    a = parse()
    season_kw = dict(weeks=a.weeks, spread=a.spread, tau=a.tau)
    factories = {"Bradley-Terry": lambda n: BradleyTerry(n, prior_sd=a.prior_sd),
                 "Elo": lambda n: Elo(n, k=a.k, passes=a.passes)}
    figs, res = a.out / "figures", a.out / "results"
    res.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    # ---- one showcase season -------------------------------------------------
    season = simulate_season(seed=a.seed, **season_kw)
    bt = factories["Bradley-Terry"](season.n).fit(season.games)
    elo = factories["Elo"](season.n).fit(season.games)
    models = {"Bradley-Terry": bt, "Elo": elo}

    print(f"\n=== Season (seed {a.seed}) ===")
    for wk in range(1, a.weeks + 1):
        print(f"W{wk:<2} " + "  ".join(f"{ABBRS[g.winner]}>{ABBRS[g.loser]}"
                                      for g in season.games if g.week == wk))

    rec = season.records()
    table = pd.DataFrame({
        "team": ABBRS, "true_rank": season.true_rank + 1,
        "true_strength": season.strengths.round(3), "W": rec[:, 0], "L": rec[:, 1],
        "BT_theta": bt.ratings.round(3), "BT_se": bt.se().round(3),
        "BT_elo_scale": bt.elo_scale().round(0), "BT_rank": bt.rank_of() + 1,
        "Elo": elo.ratings.round(0), "Elo_rank": elo.rank_of() + 1,
    }).sort_values("true_rank")
    print("\n=== Ratings ===")
    print(table.to_string(index=False))
    print("\n=== Rank agreement with the hidden order ===")
    print(rank_agreement(season, models).round(3).to_string(index=False))
    metrics = summarize(season, models)
    print("\n=== Played (train) vs never-played (test) pairs ===")
    print(metrics.round(3).to_string(index=False))
    wk = week11_table(season, models)
    print(f"\n=== Week {a.weeks + 1}: TI test games ===")
    print(wk.round(2).to_string(index=False))

    # ---- many seasons ---------------------------------------------------------
    print(f"\nReplicating {a.reps} seasons ...")
    rep_pairs, rep_learn = replicate(factories, n_seasons=a.reps, **season_kw)
    names = list(factories)
    curve_test = distance_curve(rep_pairs, names, split="test")
    curve_train = distance_curve(rep_pairs, names, split="train")
    lc = learning_curve(rep_learn)
    cal = calibration(rep_pairs, names)
    print("\nDistance effect (never-played pairs):")
    print(curve_test.round(3).to_string(index=False))
    print("\nLearning curve (end of season):")
    print(lc[lc.week == a.weeks][["model", "spearman", "acc_test", "mae_test"]]
          .round(3).to_string(index=False))

    # ---- save -----------------------------------------------------------------
    pd.DataFrame([dict(week=g.week, team_a=ABBRS[g.a], team_b=ABBRS[g.b],
                       winner=ABBRS[g.winner]) for g in season.games + season.test_week]
                 ).to_csv(res / "games.csv", index=False)
    table.to_csv(res / "ratings.csv", index=False)
    pair_table(season, models).to_csv(res / "pair_predictions.csv", index=False)
    metrics.to_csv(res / "metrics.csv", index=False)
    wk.to_csv(res / "test_week.csv", index=False)
    curve_test.to_csv(res / "distance_curve_test.csv", index=False)
    curve_train.to_csv(res / "distance_curve_train.csv", index=False)
    lc.to_csv(res / "learning_curve.csv", index=False)
    cal.to_csv(res / "calibration.csv", index=False)

    out = [plots.fig_results_matrix(season, figs),
           plots.fig_rankings(season, models, figs),
           plots.fig_ratings(season, bt, elo, figs),
           plots.fig_trajectories(season, bt, elo, figs),
           plots.fig_prob_matrices(season, models, figs),
           plots.fig_distance(curve_test, curve_train, a.reps, figs),
           plots.fig_test_week(season, models, figs),
           plots.fig_learning_curves(lc, a.reps, a.weeks, figs),
           plots.fig_calibration(cal, a.reps, figs)]
    print("\nSaved:\n  " + "\n  ".join(str(p) for p in out))


if __name__ == "__main__":
    main()
