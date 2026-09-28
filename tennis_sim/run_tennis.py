"""
run_tennis.py — Bradley–Terry vs Elo on a simulated tennis tour.

A teaching simulation. Six players with hidden, evenly spaced skills play a
tour of round-robin tournaments. Two learners estimate the skills from wins and
losses only:

    Bradley–Terry (BT)  batch: refit after every tournament on ALL matches so far
    Elo                 online: update after EVERY match, then forget the match

Both use the same outcome rule, P(i beats j) = sigmoid(v_i - v_j), which is also
the rule that generates the data. So the models are correctly specified, and
any difference between them is a difference of ESTIMATOR, not of theory.

The players are real, the results are not: skills are illustrative, not predictions.

Run:    python run_tennis.py          (writes figures/ and results/summary.txt)
Needs:  numpy, matplotlib
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 11
GAP           = 1.1      # logits between neighbours in the ladder → ~75% favourite
N_TOURNAMENTS = 6        # each tournament = one full round robin (15 matches)
K_VALUES      = (20, 60) # two Elo step sizes, to show what K does
ELO_START     = 1500
PRIOR_SD      = 3.0      # BT Gaussian prior sd (logits). Wider than the NFL sim's
                         # 1.0 because this world spans 5.5 logits; sd 1 would
                         # squash everyone together. It is what keeps an unbeaten
                         # player's rating finite (see concept figure C5).
N_SHUFFLES    = 200      # re-orderings of the same season for the order-dependence figure

FIG_DIR = "figures"
RES_DIR = "results"
FOOTNOTE = "Simulated: skills are illustrative, not predictions."

ELO_PER_LOGIT = 400 / np.log(10)   # ≈ 173.7: 1 BT logit = 173.7 Elo points

# ─────────────────────────────────────────────────────────────────────────────
# Players: ladder order = hidden true order (1 = strongest)
# Colours: flag colours. `core` is the line/marker fill, `edge` the outline,
# so each player reads as a two-colour flag. Unique marker per player, and every
# line gets a direct name label, so nobody has to decode colour alone.
# ─────────────────────────────────────────────────────────────────────────────
PLAYERS = ["Sinner", "Alcaraz", "Zverev", "Djokovic", "Shelton", "de Minaur"]
STYLE = {
    "Sinner":    dict(core="#009246", edge="#CE2B37", marker="o", country="Italy"),
    "Alcaraz":   dict(core="#FFC400", edge="#AA151B", marker="s", country="Spain"),
    "Zverev":    dict(core="#000000", edge="#FF9E1B", marker="D", country="Germany"),
    "Djokovic":  dict(core="#C6363C", edge="#0C4076", marker="^", country="Serbia"),
    "Shelton":   dict(core="#1F3A93", edge="#B22234", marker="v", country="USA"),
    "de Minaur": dict(core="#00843D", edge="#FFCD00", marker="P", country="Australia"),
}
N = len(PLAYERS)
S_TRUE = GAP * ((N - 1) / 2 - np.arange(N))   # centred: +2.75 … −2.75


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def two_tone(lw=2.2):
    """Path effect: outline stroke in the flag's second colour."""
    return lambda edge: [pe.Stroke(linewidth=lw + 2.4, foreground=edge), pe.Normal()]


# ─────────────────────────────────────────────────────────────────────────────
# World: generate a season
# ─────────────────────────────────────────────────────────────────────────────
def round_robin_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def simulate_season(rng):
    """Returns a list of matches in play order: (winner, loser, tournament)."""
    matches = []
    for t in range(N_TOURNAMENTS):
        pairs = round_robin_pairs()
        order = rng.permutation(len(pairs))          # random order within a tournament
        for k in order:
            i, j = pairs[k]
            if rng.random() < sigmoid(S_TRUE[i] - S_TRUE[j]):
                matches.append((i, j, t))
            else:
                matches.append((j, i, t))
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Bradley–Terry, MAP with Gaussian prior, fit by Newton's method
#
#   log posterior  ℓ(v) = Σ_matches log σ(v_winner − v_loser) − Σ_k v_k² / (2 τ²)
#   gradient       ∂ℓ/∂v_k = Σ_{matches of k} (y − σ(v_k − v_opp)) − v_k / τ²
#   Hessian        H = −Σ p(1−p)(e_w − e_l)(e_w − e_l)ᵀ − I/τ²
#   Newton step    v ← v − H⁻¹ ∇ℓ
#   Laplace SE     sqrt(diag((−H)⁻¹))
# ─────────────────────────────────────────────────────────────────────────────
def fit_bt(matches, tau=PRIOR_SD, iters=50):
    v = np.zeros(N)
    for _ in range(iters):
        g = -v / tau**2
        H = -np.eye(N) / tau**2
        for w, l, *_ in matches:
            p = sigmoid(v[w] - v[l])
            g[w] += 1 - p
            g[l] -= 1 - p
            c = p * (1 - p)
            H[w, w] -= c; H[l, l] -= c; H[w, l] += c; H[l, w] += c
        step = np.linalg.solve(H, g)
        v = v - step
        if np.max(np.abs(step)) < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    return v, se


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Elo, online
#
#   expected score  E_w = 1 / (1 + 10^((R_l − R_w)/400))   (= σ((R_w − R_l)/173.7))
#   update          R_w += K (1 − E_w),   R_l −= K (1 − E_w)
#
# Only the two players in the match move. The match is then discarded.
# ─────────────────────────────────────────────────────────────────────────────
def run_elo(matches, K, start=ELO_START, record=False):
    R = np.full(N, float(start))
    history = [R.copy()]
    for w, l, *_ in matches:
        E = 1 / (1 + 10 ** ((R[l] - R[w]) / 400))
        R[w] += K * (1 - E)
        R[l] -= K * (1 - E)
        if record:
            history.append(R.copy())
    return (R, np.array(history)) if record else R


def to_elo_scale(v):
    """BT logits (centred) → Elo points, so both models share one axis."""
    return ELO_START + ELO_PER_LOGIT * (v - v.mean())


def prob_matrix_from_logits(v):
    return sigmoid(v[:, None] - v[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def footnote(fig):
    fig.text(0.99, -0.02, FOOTNOTE, ha="right", va="top", fontsize=7, color="#777777")


def save(fig, name):
    footnote(fig)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def player_marker(ax, x, y, name, size=11, **kw):
    st = STYLE[name]
    ax.plot(x, y, st["marker"], ms=size, mfc=st["core"], mec=st["edge"], mew=2.2,
            linestyle="none", **kw)


def fig1_world():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))
    # ladder
    for k, name in enumerate(PLAYERS):
        player_marker(a, S_TRUE[k], 0, name, size=14, zorder=3)
        a.text(S_TRUE[k], 0.13 if k % 2 == 0 else -0.17, name, ha="center", fontsize=9)
    a.axhline(0, color="#999999", lw=0.8, zorder=1)
    a.set_ylim(-0.3, 0.25); a.set_yticks([]); a.set_aspect("auto")
    a.invert_xaxis()
    a.set_xlabel("hidden skill sₖ (logits)")
    a.set_title(f"The ladder: evenly spaced, {GAP} logits apart")

    # true probability matrix
    P = prob_matrix_from_logits(S_TRUE)
    P_show = np.where(np.eye(N, dtype=bool), np.nan, P)
    im = b.imshow(P_show, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(N):
        for j in range(N):
            if i != j:
                lab = f"{P[i, j]:.3f}" if (P[i, j] > 0.99 or P[i, j] < 0.01) else f"{P[i, j]:.2f}"
                b.text(j, i, lab, ha="center", va="center", fontsize=8,
                       color="white" if P[i, j] > 0.7 else "black")
    b.set_xticks(range(N)); b.set_xticklabels(PLAYERS, rotation=40, ha="right")
    b.set_yticks(range(N)); b.set_yticklabels(PLAYERS)
    b.set_title("True P(row beats column) = σ(s_row − s_col)")
    fig.colorbar(im, ax=b, fraction=0.046)
    fig.tight_layout()
    save(fig, "01_world.png")


def fig2_results(matches):
    W = np.zeros((N, N), int)
    for w, l, _ in matches:
        W[w, l] += 1
    meetings = N_TOURNAMENTS
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    frac = np.where(np.eye(N, dtype=bool), np.nan, W / meetings)
    im = ax.imshow(frac, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            upset = i > j and W[i, j] > 0      # lower-ranked row beat higher-ranked column
            ax.text(j, i, f"{W[i, j]}", ha="center", va="center",
                    fontsize=12 if upset else 10,
                    fontweight="bold" if upset else "normal",
                    color="white" if frac[i, j] > 0.7 else "black")
            if upset:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="#1a1a1a", lw=2))
    ax.set_xticks(range(N)); ax.set_xticklabels(PLAYERS, rotation=40, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(PLAYERS)
    n_up = int(sum(W[i, j] for i in range(N) for j in range(i)))
    ax.set_title(f"Season results: row's wins vs column (out of {meetings})\n"
                 f"boxed = upsets (lower-ranked player won); {n_up} of {len(matches)} matches")
    fig.colorbar(im, ax=ax, fraction=0.046, label="win fraction")
    fig.tight_layout()
    save(fig, "02_results_matrix.png")
    return n_up


def spread_labels(y, min_gap):
    """Nudge label heights apart so names don't overlap (order preserved)."""
    order = np.argsort(-y)
    out = y.astype(float).copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[a] - out[b] < min_gap:
            out[b] = out[a] - min_gap
    return out


def fig3_trajectories(matches):
    """Three panels on one Elo-scale axis: BT (per tournament), Elo K=20, Elo K=60."""
    n_per_t = len(round_robin_pairs())
    true_elo = to_elo_scale(S_TRUE)
    bt = np.array([to_elo_scale(fit_bt(matches[: t * n_per_t])[0])
                   for t in range(1, N_TOURNAMENTS + 1)])
    bt = np.vstack([np.full(N, ELO_START), bt])          # everyone starts level
    x_t = np.arange(N_TOURNAMENTS + 1)
    x_match = np.arange(len(matches) + 1) / n_per_t      # match index in tournaments

    panels = [("Bradley–Terry (refit after each tournament)", x_t, bt)]
    for K in K_VALUES:
        _, hist = run_elo(matches, K, record=True)
        panels.append((f"Elo, K = {K} (update after every match)", x_match, hist))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, (title, x, Y) in zip(axes, panels):
        for k, name in enumerate(PLAYERS):
            st = STYLE[name]
            ax.axhline(true_elo[k], color=st["core"], ls=":", lw=1.1, alpha=0.8)
            ax.plot(x, Y[:, k], color=st["core"], lw=2,
                    path_effects=two_tone()(st["edge"]),
                    marker=st["marker"] if Y is bt else None, ms=7,
                    mfc=st["core"], mec=st["edge"], mew=1.5)
        ly = spread_labels(Y[-1], min_gap=38)
        for k, name in enumerate(PLAYERS):
            ax.text(N_TOURNAMENTS + 0.15, ly[k], name, va="center", fontsize=9)
        ax.set_xlim(0, N_TOURNAMENTS + 1.3)
        ax.set_xticks(range(N_TOURNAMENTS + 1))
        ax.set_xlabel("tournaments completed")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("rating (Elo scale)")
    fig.suptitle("Dotted lines = true skill on the same scale. "
                 "BT reaches the truth's spread; Elo climbs toward it at a speed set by K.",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "03_trajectories.png")


def fig4_order_shuffle(matches, rng):
    v_bt, _ = fit_bt(matches)
    bt_final = to_elo_scale(v_bt)
    finals = {K: [] for K in K_VALUES}
    for _ in range(N_SHUFFLES):
        perm = [matches[i] for i in rng.permutation(len(matches))]
        for K in K_VALUES:
            finals[K].append(run_elo(perm, K))
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(12, 4.8), sharey=True)
    for ax, K in zip(axes, K_VALUES):
        F = np.array(finals[K])
        for k, name in enumerate(PLAYERS):
            st = STYLE[name]
            jitter = rng.normal(0, 0.06, len(F))
            ax.scatter(np.full(len(F), k) + jitter, F[:, k], s=8, color=st["core"],
                       alpha=0.35, edgecolor="none")
            ax.hlines(bt_final[k], k - 0.35, k + 0.35, color="black", lw=2.5,
                      label="BT (any order)" if k == 0 else None)
            player_marker(ax, k, np.median(F[:, k]), name, size=9, zorder=4)
        ax.set_xticks(range(N)); ax.set_xticklabels(PLAYERS, rotation=30, ha="right")
        spread = np.mean(F.max(0) - F.min(0))
        ax.set_title(f"Elo K = {K}: {N_SHUFFLES} orderings, mean range {spread:.0f} pts",
                     fontsize=10)
    axes[0].set_ylabel("final rating (Elo scale)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Same 90 matches, shuffled order: Elo's answer moves, BT's does not", y=1.02)
    fig.tight_layout()
    save(fig, "04_order_shuffle.png")
    return {K: float(np.mean(np.array(finals[K]).max(0) - np.array(finals[K]).min(0)))
            for K in K_VALUES}


def fig5_confidence(matches):
    P_true = prob_matrix_from_logits(S_TRUE)
    v_bt, _ = fit_bt(matches)
    models = {"BT": prob_matrix_from_logits(v_bt)}
    for K in K_VALUES:
        R = run_elo(matches, K)
        models[f"Elo K={K}"] = prob_matrix_from_logits(R / ELO_PER_LOGIT)
    iu = np.triu_indices(N, 1)                       # favourite = row (higher rank)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0.4, 1], [0.4, 1], "--", color="black", lw=1, label="perfect")
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
    marks = {"BT": ("o", "#B33A2B"), f"Elo K={K_VALUES[0]}": ("s", "#F6C28B"),
             f"Elo K={K_VALUES[1]}": ("^", "#E8873C")}
    err = {}
    for off, (name, P) in zip((-0.006, 0.0, 0.006), models.items()):
        m, c = marks[name]
        ax.scatter(P_true[iu] + off, P[iu], marker=m, s=55, color=c, edgecolor="black",
                   lw=0.6, label=name, zorder=3)
        err[name] = float(np.mean(np.abs(P[iu] - P_true[iu])))
    ax.set_xlabel("true P(favourite wins)")
    ax.set_ylabel("model's P(favourite wins)")
    ax.set_xlim(0.7, 1.02); ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Confidence after the season (all 15 pairs)\n"
                 "below the diagonal = underconfident; below 0.5 = picks the wrong favourite")
    fig.tight_layout()
    save(fig, "05_confidence.png")
    return err


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    matches = simulate_season(rng)

    fig1_world()
    n_upsets = fig2_results(matches)
    fig3_trajectories(matches)
    spreads = fig4_order_shuffle(matches, np.random.default_rng(SEED + 1))
    errs = fig5_confidence(matches)

    # summary
    v_bt, se = fit_bt(matches)
    lines = [f"seed {SEED}, gap {GAP} logits, {N_TOURNAMENTS} tournaments, "
             f"{len(matches)} matches, {n_upsets} upsets", ""]
    lines.append(f"{'player':<10} {'true':>7} {'BT':>7} " +
                 " ".join(f"{'Elo K=' + str(K):>9}" for K in K_VALUES) + "   (Elo scale)")
    elo_f = {K: run_elo(matches, K) for K in K_VALUES}
    tru = to_elo_scale(S_TRUE); bt = to_elo_scale(v_bt)
    wins = np.zeros(N, int)
    for w, *_ in matches:
        wins[w] += 1
    for k, name in enumerate(PLAYERS):
        lines.append(f"{name:<10} {tru[k]:7.0f} {bt[k]:7.0f} " +
                     " ".join(f"{elo_f[K][k]:9.0f}" for K in K_VALUES) +
                     f"   record {wins[k]}-{N_TOURNAMENTS * (N - 1) - wins[k]}")
    lines += ["", "mean |P - P_true| over the 15 pairs: " +
              ", ".join(f"{m} {e:.3f}" for m, e in errs.items()),
              "mean final-rating range across shuffled orders (Elo pts): " +
              ", ".join(f"K={K} {s:.0f}" for K, s in spreads.items()) + ", BT 0"]
    text = "\n".join(lines)
    with open(os.path.join(RES_DIR, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
