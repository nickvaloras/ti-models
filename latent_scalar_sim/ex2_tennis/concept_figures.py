"""
concept_figures.py — teaching figures for the tennis README.

These are CONCEPT figures: closed-form calculations and tiny worked examples
that explain one idea each. Simulation results come from run_tennis.py. This
file reuses its players, colours and model code so the two always agree.

Run:    python concept_figures.py      (writes figures/concept_*.png)
"""
import numpy as np
import matplotlib.pyplot as plt

from run_tennis import (PLAYERS, STYLE, N, S_TRUE, ELO_PER_LOGIT, ELO_START,
                        PRIOR_SD, SEED, sigmoid, fit_bt, run_elo, simulate_season,
                        round_robin_pairs, to_elo_scale, save, two_tone, FIG_DIR)
import os

LIGHT, MID, DARK = "#F6C28B", "#E8873C", "#B33A2B"
GREY = "#777777"


# ─────────────────────────────────────────────────────────────────────────────
# C1  One model, two unit systems
# ─────────────────────────────────────────────────────────────────────────────
def c1_same_curve():
    fig, ax = plt.subplots(figsize=(7, 4.3))
    dR = np.linspace(-800, 800, 400)
    ax.plot(dR, 1 / (1 + 10 ** (-dR / 400)), color=DARK, lw=2.5,
            label="Elo: 1 / (1 + 10^(−ΔR/400))")
    ax.plot(dR[::20], sigmoid(dR[::20] / ELO_PER_LOGIT), "o", color=LIGHT, mec="black",
            label="BT: σ(Δv),  Δv = ΔR · ln10/400")
    ax.axhline(0.5, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel("rating difference ΔR (Elo points)")
    ax.set_ylabel("P(i beats j)")
    top = ax.secondary_xaxis("top", functions=(lambda r: r / ELO_PER_LOGIT,
                                               lambda v: v * ELO_PER_LOGIT))
    top.set_xlabel("same difference in BT logits (1 logit ≈ 173.7 Elo)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("C1. Same model, two unit systems", pad=34)
    fig.tight_layout()
    save(fig, "concept_c1_same_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# C2  The order flip: two matches, split 1–1, played in both orders
# ─────────────────────────────────────────────────────────────────────────────
def c2_order_flip():
    a, b = 0, 1                                   # Sinner, Alcaraz
    orders = {"Sinner wins, then Alcaraz wins": [(a, b), (b, a)],
              "Alcaraz wins, then Sinner wins": [(b, a), (a, b)]}
    K = 20
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, ms) in zip(axes, orders.items()):
        R = np.full(2, 1500.0)
        steps = [R.copy()]
        for w, l in ms:
            E = 1 / (1 + 10 ** ((R[l] - R[w]) / 400))
            R[w] += K * (1 - E); R[l] -= K * (1 - E)
            steps.append(R.copy())
        steps = np.array(steps)
        for k, name in enumerate(["Sinner", "Alcaraz"]):
            st = STYLE[name]
            ax.plot([0, 1, 2], steps[:, k], color=st["core"], lw=2.2,
                    path_effects=two_tone()(st["edge"]), marker=st["marker"], ms=9,
                    mfc=st["core"], mec=st["edge"], mew=2)
            ax.text(2.08, steps[-1, k], f"{name} {steps[-1, k]:.1f}", va="center", fontsize=9)
        ax.axhline(1500, color="black", lw=2, ls="--")
        ax.text(1.0, 1501.2, "BT: 1–1 → equal, either order", fontsize=9, ha="center")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["start", "match 1", "match 2"])
        ax.set_xlim(-0.2, 2.9)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel(f"Elo rating (K = {K})")
    fig.suptitle("C2. Same two results, opposite orders: Elo ends with the last winner ahead",
                 y=1.02)
    fig.tight_layout()
    save(fig, "concept_c2_order_flip.png")


# ─────────────────────────────────────────────────────────────────────────────
# C3  Local vs global updating: add one upset after two tournaments
# ─────────────────────────────────────────────────────────────────────────────
def c3_update_scope():
    matches = simulate_season(np.random.default_rng(SEED))
    before = matches[: 2 * len(round_robin_pairs())]        # first two tournaments
    upset = (5, 0, 2)                                        # de Minaur beats Sinner
    v0, _ = fit_bt(before); v1, _ = fit_bt(before + [upset])
    d_bt = (v1 - v0) * ELO_PER_LOGIT
    R0 = run_elo(before, 20); R1 = run_elo(before + [upset], 20)
    d_elo = R1 - R0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(N)
    for ax, d, ttl in [(axes[0], d_elo, "Elo (K = 20): only the two players move"),
                       (axes[1], d_bt, "Bradley–Terry refit: everyone can move")]:
        for k, name in enumerate(PLAYERS):
            st = STYLE[name]
            ax.bar(k, d[k], color=st["core"], edgecolor=st["edge"], lw=2.2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(PLAYERS, rotation=30, ha="right")
        ax.set_title(ttl, fontsize=10.5)
    axes[0].set_ylabel("change in rating from one match (Elo points)")
    fig.suptitle("C3. After two tournaments, add one upset: de Minaur beats Sinner", y=1.02)
    fig.tight_layout()
    save(fig, "concept_c3_update_scope.png")
    return d_elo, d_bt


# ─────────────────────────────────────────────────────────────────────────────
# C4  K as memory: how much does a match from n matches ago still count?
#     Linearised Elo: each later match multiplies an old match's influence by
#     (1 − η·p(1−p)), η = K·ln10/400. Here p = 0.75 (a typical favourite).
#     BT weights every match played so far equally.
# ─────────────────────────────────────────────────────────────────────────────
def c4_memory():
    ago = np.arange(0, 61)
    c = 0.75 * 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ago, np.ones_like(ago, float), color="black", lw=2.5,
            label="Bradley–Terry: every match counts the same")
    for K, col in [(20, LIGHT), (60, MID), (150, DARK)]:
        eta = K * np.log(10) / 400
        w = (1 - eta * c) ** ago
        half = np.log(0.5) / np.log(1 - eta * c)
        ax.plot(ago, w, color=col, lw=2.2, label=f"Elo K = {K} (half-life ≈ {half:.0f} matches)")
    ax.axvline(30, color=GREY, ls=":", lw=1)
    ax.text(30.5, 0.9, "one season\n(30 matches\nper player)", fontsize=8, color=GREY)
    ax.set_xlabel("matches ago"); ax.set_ylabel("relative weight of that match today")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title("C4. K is a learning rate, and also a memory (linearised)")
    fig.tight_layout()
    save(fig, "concept_c4_memory.png")


# ─────────────────────────────────────────────────────────────────────────────
# C5  The unbeaten player: why BT needs a prior
#     Sinner goes 6–0 against an opponent fixed at v = 0. Log posterior for his v.
# ─────────────────────────────────────────────────────────────────────────────
def c5_prior():
    v = np.linspace(-1, 12, 500)
    loglik = 6 * np.log(sigmoid(v))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(v, loglik - loglik.max(), color="black", lw=2.2, label="no prior (pure likelihood)")
    for tau, col in [(PRIOR_SD, DARK), (1.0, MID)]:
        lp = loglik - v**2 / (2 * tau**2)
        ax.plot(v, lp - lp.max(), color=col, lw=2.2, label=f"with prior sd = {tau:g}")
        ax.plot(v[np.argmax(lp)], 0, "o", color=col, mec="black", zorder=3)
    ax.set_ylim(-6, 0.5)
    ax.set_xlabel("Sinner's rating v (logits), opponent at 0")
    ax.set_ylabel("log posterior (shifted, max = 0)")
    ax.annotate("keeps rising:\nno finite best value", (11, -0.02), xytext=(7.2, -2.2),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=GREY))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("C5. An unbeaten record (6–0): BT needs a prior to give a finite rating")
    fig.tight_layout()
    save(fig, "concept_c5_prior_unbeaten.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    c1_same_curve(); c2_order_flip()
    d_elo, d_bt = c3_update_scope()
    print("C3 Elo deltas:", np.round(d_elo, 1))
    print("C3 BT  deltas:", np.round(d_bt, 1))
    c4_memory(); c5_prior()
