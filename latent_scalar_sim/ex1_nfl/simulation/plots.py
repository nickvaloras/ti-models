"""Team-color visuals.

Team identity is never carried by color alone: every team mark is paired with
its abbreviation badge (primary fill, secondary ring), because several NFL
palettes collide (NE / CHI / NYG / HOU navies; KC / ARI reds).
Model identity uses a separate fixed palette (True gray, BT blue, Elo orange,
Record = dark dashed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .models import ELO_SCALE
from .teams import ABBRS, TEAMS, text_color_on

INK, INK2, MUTED, GRID = "#1F2328", "#57606A", "#8C959F", "#E6E8EB"
MODEL_COLORS = {"True": "#6E7781", "Bradley-Terry": "#2F6BD8", "Elo": "#E07B28",
                "Record": INK}
MODEL_LS = {"True": "-", "Bradley-Terry": "-", "Elo": "-", "Record": (0, (4, 2))}
DIVERGING = LinearSegmentedColormap.from_list(
    "fav", ["#B2182B", "#EF8A62", "#E8E8E8", "#67A9CF", "#2166AC"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })


_style()


def badge(ax, x, y, idx, fontsize=9, **kw):
    t = TEAMS[idx]
    return ax.text(x, y, t.abbr, ha="center", va="center", fontsize=fontsize,
                   fontweight="bold", color=text_color_on(t.primary), zorder=6,
                   bbox=dict(boxstyle="round,pad=0.32", fc=t.primary, ec=t.secondary,
                             lw=2), **kw)


def _suptitle(fig, title, sub=None):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14,
             fontweight="bold", color=INK)
    if sub:
        fig.text(0.012, 1 - 0.42 / h, sub, ha="left", va="top", fontsize=10, color=INK2)


def _save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _repel(y, min_gap):
    """Nudge label positions apart while keeping their order."""
    order = np.argsort(y)
    ys = np.array(y, dtype=float)[order]
    for _ in range(500):
        moved = False
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                mid = (ys[k] + ys[k - 1]) / 2
                ys[k - 1], ys[k] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(ys)
    out[order] = ys
    return out


# -----------------------------------------------------------------------------
# 1. results matrix: observed (train) vs. to-be-inferred (test)
# -----------------------------------------------------------------------------
def fig_results_matrix(season, outdir):
    n = season.n
    order = np.argsort(season.true_rank)
    fig, ax = plt.subplots(figsize=(11, 10.4))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    played = {}
    for g in season.games:
        played[(g.a, g.b)] = played[(g.b, g.a)] = g
    wk11 = {}
    for g in season.test_week:
        wk11[(g.a, g.b)] = wk11[(g.b, g.a)] = g

    for r, i in enumerate(order):
        for c, j in enumerate(order):
            if i == j:
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#2B2F36", ec="white", lw=1.5))
                continue
            g = played.get((i, j))
            if g is None:
                test = (i, j) in wk11
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#F4F5F7",
                                       ec="#D5D9DE", lw=0, hatch="///", zorder=1))
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="none", ec="white",
                                       lw=1.5, zorder=1.5))
                if test:
                    ax.add_patch(Rectangle((c - .42, r - .42), .84, .84, fc="white",
                                           ec=INK, lw=1.4, zorder=2))
                    ax.text(c, r - 0.08, "?", ha="center", va="center", color=INK,
                            fontsize=11, fontweight="bold", zorder=3)
                    ax.text(c, r + 0.27, f"W{season.weeks + 1}", ha="center",
                            va="center", color=INK2, fontsize=5.5, zorder=3)
                else:
                    ax.text(c, r, "?", ha="center", va="center", color=MUTED,
                            fontsize=10, fontweight="bold", zorder=2)
                continue
            win = g.winner
            fc = TEAMS[win].primary
            tc = text_color_on(fc)
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc=fc, ec="white", lw=1.5))
            ax.text(c, r - 0.06, "W" if win == i else "L", ha="center", va="center",
                    color=tc, fontsize=9.5, fontweight="bold")
            ax.text(c, r + 0.3, f"wk {g.week}", ha="center", va="center", color=tc,
                    fontsize=5, alpha=0.85)

    for k, t in enumerate(order):
        badge(ax, -1.05, k, t, fontsize=8, clip_on=False)
        badge(ax, k, -1.05, t, fontsize=7, clip_on=False, rotation=90)
    rec = season.records()
    for k, t in enumerate(order):
        ax.text(n - 0.3, k, f"{rec[t,0]}-{rec[t,1]}", ha="left", va="center",
                fontsize=9, color=INK2)
    ax.text(n - 0.3, -1.05, "W-L", ha="left", va="center", fontsize=8, color=MUTED)

    _suptitle(fig, f"Weeks 1-{season.weeks}: observed vs. inferred matchups",
              f"Filled = played (cell color = winner; W/L from the row team's view). "
              f"Hatched '?' = {len(season.unplayed_pairs())} never-played pairs.\n"
              f"Boxed '?' = the {len(season.test_week)} week-{season.weeks + 1} TI test "
              f"games. Teams in true-strength order.")
    fig.subplots_adjust(top=0.855)
    return _save(fig, outdir, "01_results_matrix")


# -----------------------------------------------------------------------------
# 2. bump chart of rankings
# -----------------------------------------------------------------------------
def fig_rankings(season, models, outdir):
    n = season.n
    cols = {"True\nstrength": season.strengths,
            "Win-loss\nrecord": season.record_score()}
    for name, m in models.items():
        cols[name] = m.ratings
    ranks = {k: np.argsort(np.argsort(-v)) for k, v in cols.items()}
    xs = np.arange(len(cols)) * 1.6

    fig, ax = plt.subplots(figsize=(8.6, 10))
    for t in range(n):
        ys = [ranks[k][t] for k in cols]
        ax.plot(xs, ys, color=TEAMS[t].primary, lw=4, alpha=0.5, zorder=2,
                solid_capstyle="round")
        for x, y in zip(xs, ys):
            badge(ax, x, y, t, fontsize=8.5)
    for x, k in zip(xs, cols):
        ax.text(x, -1.1, k, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=INK)
    for r in range(n):
        ax.text(xs[0] - 0.7, r, f"#{r+1}", ha="right", va="center", color=MUTED, fontsize=9)
    ax.set_xlim(xs[0] - 1.0, xs[-1] + 0.6)
    ax.set_ylim(n - 0.4, -1.5)
    ax.axis("off")
    _suptitle(fig, "Who's #1? True order vs. what each model inferred",
              "Lines track each team across rankings; crossings are ranking errors.\n"
              "Record = wins, tie-broken by opponents' wins (no model).")
    fig.subplots_adjust(top=0.9)
    return _save(fig, outdir, "02_rankings_bump")


# -----------------------------------------------------------------------------
# 3. rating bars with uncertainty
# -----------------------------------------------------------------------------
def fig_ratings(season, bt, elo, outdir):
    order = np.argsort(season.true_rank)
    rec = season.records()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1.25, 1.25], wspace=0.08))
    panels = [
        ("True strength", ELO_SCALE * season.strengths, None, "Elo points vs. average"),
        ("Bradley-Terry (MAP ± 1 SE)", bt.elo_scale() - 1500, ELO_SCALE * bt.se(),
         "rating − 1500 (Elo scale)"),
        (f"Elo (K={elo.k:g})", elo.ratings - 1500, None, "rating − 1500"),
    ]
    lim = max(np.max(np.abs(v) + (0 if e is None else e)) for _, v, e, _ in panels) * 1.1
    for ax, (title, vals, err, xl) in zip(axes, panels):
        v = vals[order]
        ax.barh(y, v, height=0.68, color=[TEAMS[t].primary for t in order],
                edgecolor=[TEAMS[t].secondary for t in order], lw=1.5, zorder=3)
        if err is not None:
            ax.errorbar(v, y, xerr=err[order], fmt="none", ecolor=INK, elinewidth=1.1,
                        capsize=2.5, zorder=4)
        ax.axvline(0, color=INK2, lw=1, zorder=2)
        ax.grid(axis="x", zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{ABBRS[t]}  {rec[t,0]}-{rec[t,1]}" for t in order],
                            fontweight="bold", color=INK)
    axes[0].invert_yaxis()
    _suptitle(fig, f"Ratings after {season.weeks} weeks (all on one Elo scale)",
              "Teams sorted by hidden true strength. BT error bars: Laplace approximation. "
              "Elo's shorter bars = shrinkage toward 1500.")
    fig.subplots_adjust(top=0.87)
    return _save(fig, outdir, "03_ratings")


# -----------------------------------------------------------------------------
# 4. week-by-week trajectories
# -----------------------------------------------------------------------------
def fig_trajectories(season, bt, elo, outdir):
    H = {"Bradley-Terry (refit on all games so far)": bt.history_elo_scale(),
         "Elo (one online update per game)": elo.history_elo_scale()}
    lo = min(h.min() for h in H.values()) - 15
    hi = max(h.max() for h in H.values()) + 15
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True,
                             gridspec_kw=dict(wspace=0.16))
    for ax, (title, h) in zip(axes, H.items()):
        wk = np.arange(h.shape[0])
        for t in range(h.shape[1]):
            ax.plot(wk, h[:, t], color=TEAMS[t].primary, lw=2, zorder=3)
            ax.scatter(wk, h[:, t], s=22, color=TEAMS[t].secondary,
                       edgecolor=TEAMS[t].primary, lw=1.1, zorder=4)
        ly = _repel(h[-1], (hi - lo) * 0.041)
        for t in range(h.shape[1]):
            ax.plot([wk[-1], wk[-1] + 0.5], [h[-1, t], ly[t]], color=TEAMS[t].primary,
                    lw=1, alpha=0.7)
            badge(ax, wk[-1] + 0.95, ly[t], t, fontsize=7)
        ax.axhline(1500, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(-0.3, wk[-1] + 1.5)
        ax.set_ylim(lo, hi)
        ax.set_xticks(wk)
        ax.set_xticklabels(["start"] + [str(k) for k in wk[1:]])
        ax.set_xlabel("week")
        ax.grid(axis="y")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("rating (Elo scale)")
    _suptitle(fig, "How the ratings evolved, week by week",
              "BT re-solves the whole season each week (order-free); Elo takes one "
              "small step per game (order-dependent, shrunk toward 1500).")
    fig.subplots_adjust(top=0.87)
    return _save(fig, outdir, "04_trajectories")


# -----------------------------------------------------------------------------
# 5. probability matrices
# -----------------------------------------------------------------------------
def fig_prob_matrices(season, models, outdir):
    order = np.argsort(season.true_rank)
    mats = {"True": season.true_prob_matrix()}
    mats.update({k: m.prob_matrix() for k, m in models.items()})
    played = season.played_pairs()
    wk11 = {frozenset((g.a, g.b)) for g in season.test_week}
    n = len(order)
    fig, axes = plt.subplots(1, len(mats), figsize=(7 * len(mats), 7.4))
    for ax, (name, P) in zip(axes, mats.items()):
        M = P[np.ix_(order, order)]
        im = ax.imshow(M, cmap=DIVERGING, vmin=0, vmax=1)
        for r in range(n):
            for c in range(n):
                if r == c:
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#2B2F36", ec="white"))
                    continue
                v = M[r, c]
                ax.text(c, r, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v - 0.5) > 0.3 else INK)
                key = frozenset((order[r], order[c]))
                if key in wk11:
                    ax.add_patch(Rectangle((c - .45, r - .45), .9, .9, fc="none",
                                           ec=INK, lw=2.0))
                elif key not in played:
                    ax.add_patch(Rectangle((c - .42, r - .42), .84, .84, fc="none",
                                           ec=INK, lw=0.9, ls=(0, (2, 1.5))))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([ABBRS[t] for t in order], rotation=90, fontsize=8)
        ax.set_yticklabels([ABBRS[t] for t in order], fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name, color=MODEL_COLORS.get(name, INK))
    fig.subplots_adjust(top=0.84, right=0.92, wspace=0.14)
    cax = fig.add_axes([0.935, 0.18, 0.009, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P(row team beats column team)")
    cb.outline.set_visible(False)
    _suptitle(fig, "Predicted win probability for every matchup",
              f"Dashed box = never played (a transitive inference); bold box = "
              f"week-{season.weeks + 1} test game; unmarked = played. Numbers are P×100.")
    return _save(fig, outdir, "05_prob_matrices")


# -----------------------------------------------------------------------------
# 6. symbolic distance effect
# -----------------------------------------------------------------------------
def fig_distance(curve_test, curve_train, n_seasons, outdir,
                 model_names=("Bradley-Terry", "Elo")):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(wspace=0.22))
    ax = axes[0]
    x = curve_test["distance"]
    ax.plot(x, curve_test["p_true"], color=MODEL_COLORS["True"], lw=2, marker="o",
            ms=7, label="True", zorder=3)
    for n in model_names:
        ax.plot(x, curve_test[f"p_{n}"], color=MODEL_COLORS[n], lw=2, marker="o", ms=7,
                label=f"{n} (never played)", zorder=4)
        ax.plot(curve_train["distance"], curve_train[f"p_{n}"], color=MODEL_COLORS[n],
                lw=1.5, ls=(0, (3, 2)), marker="s", ms=4, alpha=0.7,
                label=f"{n} (played)", zorder=3)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(range(1, int(x.max()) + 1))
    ax.set_xlabel("symbolic distance |rank_i − rank_j|")
    ax.set_ylabel("mean P(stronger team wins)")
    ax.set_title("Confidence grows with distance")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1]
    w = 0.38
    for k, n in enumerate(model_names):
        ax.bar(x + (k - 0.5) * w, curve_test[f"acc_{n}"], width=w - 0.04,
               color=MODEL_COLORS[n], label=n, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, int(x.max()) + 1))
    ax.set_xlabel("symbolic distance")
    ax.set_ylabel("share picking the truly stronger team")
    ax.set_title("Accuracy on never-played pairs")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.97))
    for xi, nn in zip(x, curve_test["n"]):
        ax.text(xi, 1.03, f"{nn}", ha="center", fontsize=6.5, color=MUTED)
    ax.text(x.max() + 0.6, 1.03, "n", fontsize=6.5, color=MUTED)
    _suptitle(fig, "The symbolic distance effect",
              f"Averaged over {n_seasons} simulated seasons. Solid = never-played "
              "(inferred) pairs; dashed = pairs that were played.")
    fig.subplots_adjust(top=0.83)
    return _save(fig, outdir, "06_symbolic_distance")


# -----------------------------------------------------------------------------
# 7. TI test week, tug-of-war style
# -----------------------------------------------------------------------------
def fig_test_week(season, models, outdir):
    games = sorted(season.test_week,
                   key=lambda g: min(season.true_rank[g.a], season.true_rank[g.b]))
    names = list(models)
    row_h = len(names) * 0.42 + 0.6
    fig, ax = plt.subplots(figsize=(11, row_h * len(games) * 0.95 + 1.2))
    hits = {n: 0 for n in names}
    for r, g in enumerate(games):
        a, b = sorted((g.a, g.b), key=lambda t: season.true_rank[t])   # a = stronger
        y0 = r * row_h
        mid = y0 + (len(names) - 1) * 0.21
        badge(ax, -0.09, mid, a, fontsize=10)
        badge(ax, 1.09, mid, b, fontsize=10)
        pt = season.true_prob(a, b)
        for k, name in enumerate(names):
            yy = y0 + k * 0.42
            p = models[name].prob(a, b)
            ax.barh(yy, p, left=0, height=0.32, color=TEAMS[a].primary,
                    edgecolor="white", lw=2)
            ax.barh(yy, 1 - p, left=p, height=0.32, color=TEAMS[b].primary,
                    edgecolor="white", lw=2)
            ax.text(0.012, yy, f"{p*100:.0f}%", va="center", ha="left", fontsize=8.5,
                    fontweight="bold", color=text_color_on(TEAMS[a].primary))
            ax.text(0.988, yy, f"{(1-p)*100:.0f}%", va="center", ha="right", fontsize=8.5,
                    fontweight="bold", color=text_color_on(TEAMS[b].primary))
            ax.text(-0.2, yy, name, va="center", ha="right", fontsize=8.5, color=INK2)
            hit = models[name].prob(g.winner, g.loser) > 0.5
            hits[name] += hit
        ax.plot([pt, pt], [y0 - 0.22, y0 + (len(names) - 1) * 0.42 + 0.22], color=INK,
                lw=2, zorder=5)
        ax.text(pt, y0 - 0.27, f"true {pt*100:.0f}%", ha="center", va="bottom",
                fontsize=7.5, color=INK)
        upset = g.winner == b
        dist = abs(season.true_rank[a] - season.true_rank[b])
        ax.text(1.2, mid - 0.13, f"Winner: {ABBRS[g.winner]}" + ("  (upset)" if upset else ""),
                va="center", ha="left", fontsize=9.5, color=INK,
                fontweight="bold" if upset else "normal")
        ax.text(1.2, mid + 0.17, f"rank distance {dist}", va="center", ha="left",
                fontsize=8, color=MUTED)
    ax.set_xlim(-0.55, 1.62)
    ax.set_ylim(len(games) * row_h - 0.35, -0.45)
    ax.axis("off")
    score = ",  ".join(f"{n} {hits[n]}/{len(games)}" for n in names)
    _suptitle(fig, f"Week {season.weeks + 1}: the transitive-inference test",
              f"Every game is a matchup neither team has played. Bar split = model's "
              f"win probability; black tick = true probability.\n"
              f"Picks correct this week: {score}.")
    fig.subplots_adjust(top=0.9)
    return _save(fig, outdir, "07_test_week")


# -----------------------------------------------------------------------------
# 8. learning curves across many seasons
# -----------------------------------------------------------------------------
def fig_learning_curves(lc, n_seasons, weeks, outdir,
                        model_names=("Bradley-Terry", "Elo")):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(wspace=0.25))
    specs = [("spearman", "rank correlation with true order", "Recovering the hierarchy",
              list(model_names) + ["Record"]),
             ("acc_test", "accuracy on never-played pairs", "Inference accuracy",
              list(model_names)),
             ("mae_test", "|P_model − P_true| on never-played pairs", "Probability error",
              list(model_names))]
    for ax, (col, yl, title, names) in zip(axes, specs):
        for n in names:
            d = lc[lc.model == n].sort_values("week")
            if col not in d or d[col].isna().all():
                continue
            if n != "Record":
                ax.fill_between(d.week, d[f"{col}_lo"], d[f"{col}_hi"],
                                color=MODEL_COLORS[n], alpha=0.25, lw=0)
            ax.plot(d.week, d[col], color=MODEL_COLORS[n], ls=MODEL_LS[n], lw=2,
                    marker="o", ms=4, label=n)
        ax.set_xticks(range(0, weeks + 1))
        ax.set_xlabel("weeks of games observed")
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(axis="y")
    axes[1].axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    axes[2].set_ylim(bottom=0)
    axes[0].legend(loc="lower right")
    axes[2].legend(loc="upper right")
    _suptitle(fig, "Learning curves: how many games does each model need?",
              f"Lines = mean over {n_seasons} seasons; bands = 95% CI of the mean. "
              f"Test pairs are that season's never-played matchups.")
    fig.subplots_adjust(top=0.78)
    return _save(fig, outdir, "08_learning_curves")


# -----------------------------------------------------------------------------
# 9. calibration
# -----------------------------------------------------------------------------
def fig_calibration(cal, n_seasons, outdir, model_names=("Bradley-Terry", "Elo")):
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(0.8, 0.745, "perfectly calibrated", fontsize=8, color=MUTED, ha="center",
            va="center", rotation=45, rotation_mode="anchor")
    nmax = cal.n.max()
    for n in model_names:
        d = cal[cal.model == n].sort_values("p_pred")
        ax.plot(d.p_pred, d.p_true, color=MODEL_COLORS[n], lw=2, zorder=3, label=n)
        ax.scatter(d.p_pred, d.p_true, s=20 + 160 * d.n / nmax, color=MODEL_COLORS[n],
                   edgecolor="white", lw=1.5, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("model's predicted P(win)")
    ax.set_ylabel("true P(win)")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05, "steeper than the diagonal =\nunderconfident "
            "(predictions too close to 50%)", ha="right", fontsize=8.5, color=INK2)
    _suptitle(fig, "Calibration on never-played pairs",
              f"Binned predictions pooled over {n_seasons} seasons; dot size = count.")
    fig.subplots_adjust(top=0.86)
    return _save(fig, outdir, "09_calibration")
