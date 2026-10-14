"""
concept_figures.py — schematic figures for the betasort example
================================================================

Hand-set memory states, not simulation output. Self-contained (numpy +
matplotlib); the update arithmetic is re-written inline here on purpose so the
schematic can be read without run_betasort.py.

  concept1_position_code.png   what (U, L) encode; consolidate vs shift
  concept2_one_error_trial.png one incorrect CE trial, step by step
                               (the logic of Jensen et al. 2015, Fig 3)
"""

from math import lgamma
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ITEMS = list("ABCDEFG")
FIGDIR = Path(__file__).resolve().parent / "figures"
ITEM_COLORS = plt.cm.YlOrRd(np.linspace(0.30, 0.95, 7))
WARM = ["#F6C28B", "#E8873C", "#B33A2B"]
X = np.linspace(0.001, 0.999, 600)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def beta_pdf(a, b, x=X):
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - logB)


def density(U, L):
    """Betasort's position belief: Beta(U + 1, L + 1)."""
    return beta_pdf(U + 1, L + 1)


def caption(fig, text):
    fig.text(0.01, 0.005, "SCHEMATIC — " + text, fontsize=7.5, color="0.35",
             ha="left", va="bottom")


# ---------------------------------------------------------------------------
def concept_position_code(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    # (a) What U and L mean
    ax = axes[0]
    cases = [((0, 0), "U=0, L=0\nno evidence"),
             ((3, 1), "U=3, L=1 (dashed)\nprobably high"),
             ((12, 4), "U=12, L=4\nsame mean, surer"),
             ((1, 5), "U=1, L=5\nprobably low")]
    cols = ["0.55", WARM[2], WARM[2], WARM[1]]
    label_xy = {(0, 0): (0.5, 1.12), (3, 1): (0.40, 2.05),
                (12, 4): (0.75, 3.95), (1, 5): (0.17, 2.95)}
    for ((U, L), lab), c in zip(cases, cols):
        y = density(U, L)
        ax.plot(X, y, color=c, lw=2, ls="--" if (U, L) == (3, 1) else "-")
        if (U, L) == (0, 0):
            ax.annotate(lab, (0.96, 1.0), xytext=(0.96, 2.55), color=c,
                        ha="center", va="bottom", fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
        else:
            ax.text(*label_xy[(U, L)], lab, color=c, ha="center",
                    va="bottom", fontsize=8.5)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("position on the unit span")
    ax.set_ylabel("density")
    ax.set_title("(a) Position = Beta(U+1, L+1);  mean ≈ U/(U+L)")

    # (b) Feedback consolidates or shifts
    ax = axes[1]
    U0, L0 = 3.0, 3.0
    V0 = U0 / (U0 + L0)
    Uc, Lc = U0 + 6 * V0, L0 + 6 * (1 - V0)        # six rewards: consolidate
    Us, Ls = U0 + 2, L0                            # two pushes up: shift
    for (U, L), c, lab in [((U0, L0), "0.55", "start\nU=3, L=3"),
                           ((Uc, Lc), WARM[2], "after 6 rewards\n(+V, +1−V each)\nnarrower, same place"),
                           ((Us, Ls), WARM[1], "after 2 pushes up\n(U+1 each)\nmoved")]:
        y = density(U, L)
        ax.plot(X, y, color=c, lw=2)
        k = np.argmax(y)
        dx = {"0.55": -0.20, WARM[2]: 0.0, WARM[1]: 0.22}[c]
        ax.text(min(max(X[k] + dx, 0.12), 0.88), y[k] + 0.1, lab, color=c,
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("position on the unit span")
    ax.set_title("(b) Reward consolidates; error shifts")

    caption(fig, "hand-set (U, L) values chosen to illustrate the update rules.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def concept_error_trial(path):
    """
    A well-learned list, then one error: pair CE is shown and E is chosen.
    xi = 0.6 is deliberately low (as in the paper's Fig 3) so every step is
    visible. Reward rates are set high (r = 0.9) for every item.
    """
    V_true = np.linspace(0.92, 0.08, 7)
    strength = 10.0
    U = V_true * strength
    L = (1 - V_true) * strength
    xi, r = 0.6, 0.9
    xi_R = r / (r + 1) + 0.5
    ch, nc = 4, 2                                   # chose E, did not choose C

    states = [("1. Before the trial: well-learned list", U.copy(), L.copy(), [])]

    # Relaxation: every U, L shrinks by xi * xi_R (means unchanged, wider)
    U = U * xi * xi_R
    L = L * xi * xi_R
    states.append(("2. Relaxation: all densities widen, means unchanged",
                   U.copy(), L.copy(), list(range(7))))

    # Explicit feedback: chosen item down, unchosen item up
    V = U / (U + L)
    L[ch] += 1
    U[nc] += 1
    states.append(("3. Explicit feedback: C pushed up, E pushed down",
                   U.copy(), L.copy(), [nc, ch]))

    # Implicit inference on the five absent items (Eq. 7)
    for i in range(7):
        if i in (ch, nc):
            continue
        if V[i] > V[nc]:
            U[i] += 1                                # above the pair: up
        elif V[i] < V[ch]:
            L[i] += 1                                # below the pair: down
        else:
            U[i] += V[i]                             # between: consolidate
            L[i] += 1 - V[i]
    states.append(("4. Implicit inference: A,B up · D consolidated · F,G down",
                   U.copy(), L.copy(), [0, 1, 3, 5, 6]))

    fig, axes = plt.subplots(len(states), 1, figsize=(8.5, 9), sharex=True)
    prevU, prevL = None, None
    for ax, (title, Us, Ls, changed) in zip(axes, states):
        for k in range(7):
            y = density(Us[k], Ls[k])
            emph = k in changed
            ax.plot(X, y, color=ITEM_COLORS[k], lw=2.4 if emph else 1.2,
                    alpha=1.0 if emph or not changed else 0.45)
            m = Us[k] / (Us[k] + Ls[k])
            ax.axvline(m, color=ITEM_COLORS[k], lw=0.8, ls=":",
                       alpha=0.9 if emph else 0.4)
            ax.text(m, y.max() + 0.15, ITEMS[k], ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=ITEM_COLORS[k])
        ax.set_ylim(0, 6.2)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines["left"].set_visible(False)
    axes[0].text(0.99, 0.95, "trial: C vs E  →  chose E  →  no reward",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, color=WARM[2])
    axes[-1].set_xlabel("position on the unit span  (dotted line = mean V)")
    caption(fig, "hand-set starting state, ξ=0.6, reward rate 0.9 for all "
                 "items; mirrors the logic of Jensen et al. (2015) Fig 3.")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    concept_position_code(FIGDIR / "concept1_position_code.png")
    concept_error_trial(FIGDIR / "concept2_one_error_trial.png")
    print(f"Concept figures written to {FIGDIR}")
