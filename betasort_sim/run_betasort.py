"""
run_betasort.py — Betasort on the classic 7-item transitive-inference task
==========================================================================

Self-contained teaching script (numpy + matplotlib only). Implements the
betasort algorithm of Jensen, Muñoz, Alkan, Ferrera & Terrace (2015),
"Implicit value updating explains transitive inference performance: the
betasort model", PLoS Comput Biol 11(9): e1004523.

Task (Jensen et al. 2015, monkey procedure)
-------------------------------------------
    Hidden order:  A > B > C > D > E > F > G
    Phase 1  adjacent pairs only (AB, BC, CD, DE, EF, FG)
             20 blocks x 12 trials (each pair twice per block) = 240 trials
    Phase 2  all 21 pairs, 10 blocks x 21 trials               = 210 trials
    Phase 3  massed FG only                                    = 200 trials
             (the paper's Fig 7 test of robustness to massed trials)

    A choice is rewarded iff the earlier list item is chosen.
    "Transfer" = the moment Phase 2 starts. Critical pairs are the six
    non-adjacent pairs without a terminal item: BD, CE, DF, BE, CF, BF.

Run:   python run_betasort.py            (writes figures/ next to this file)
"""

from math import lgamma
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ITEMS = list("ABCDEFG")           # index 0 = A = top of the hidden order
N_ITEMS = len(ITEMS)
TAU = 0.05                        # noise: P(ignore memory, choose at random)
XI = 0.95                         # recall: per-trial memory retention
N_SESSIONS = 200                  # replicated simulated sessions
EXAMPLE_SEED = 7                  # seed used for single-session figures
PROBE_EVERY = 6                   # compute noiseless-probe accuracy every k trials
GRID = np.linspace(0.0, 1.0, 2001)[1:-1]   # quadrature grid on (0, 1)

FIGDIR = Path(__file__).resolve().parent / "figures"

# Plot style (ti-models conventions): warm palette, direct labels
WARM = ["#F6C28B", "#E8873C", "#B33A2B"]
ITEM_COLORS = plt.cm.YlOrRd(np.linspace(0.30, 0.95, N_ITEMS))
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

ADJACENT = [(i, i + 1) for i in range(N_ITEMS - 1)]
ALL_PAIRS = [(i, j) for i in range(N_ITEMS) for j in range(i + 1, N_ITEMS)]
CRITICAL = [(i, j) for (i, j) in ALL_PAIRS
            if j - i >= 2 and i != 0 and j != N_ITEMS - 1]


def pair_name(i, j):
    return ITEMS[i] + ITEMS[j]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Betasort:
    """
    Memory: four length-N vectors.
      U[i], L[i]  "upper" / "lower" evidence for item i's position on [0, 1].
                  Position belief = Beta(U+1, L+1). The +1 is the paper's prior
                  (Eq. 2); it keeps the density proper when U or L is 0.
      R[i], N[i]  leaky counts of rewarded / unrewarded trials on which item i
                  was *present*. They only control how fast U, L decay.

    Derivation notes (why this does transitive inference):
      * Position, not value. V_i = U_i / (U_i + L_i) is where item i sits on the
        unit span; it is *not* P(reward | i). Under adjacent-pair training every
        non-terminal item is rewarded 50% of the time, so any expected-value
        learner (Q-learning, Rescorla-Wagner) is stuck at 0.5 for B..F.
      * Consolidate vs. shift. A reward adds (V_i, 1 - V_i) to (U_i, L_i): one
        unit of evidence split in proportion to the current position. The mean
        is unchanged (U+V)/(U+L+1) = V, but U+L grows, so the density narrows.
        An error adds a whole unit to one side only, which moves the mean.
      * Implicit updating. On every trial *all* items are updated, including
        the five that were not shown. After an error on pair (ch, nc):
        items believed above the pair are pushed up (U += 1), items believed
        below are pushed down (L += 1), items in between are consolidated.
        This is what spreads B..F apart before any non-adjacent pair is seen.
    """

    def __init__(self, n=N_ITEMS, tau=TAU, xi=XI, rng=None, implicit=True):
        self.n, self.tau, self.xi = n, tau, xi
        self.rng = rng if rng is not None else np.random.default_rng()
        self.implicit = implicit      # kept for experimentation; always True here
        self.U = np.zeros(n)
        self.L = np.zeros(n)
        self.R = np.zeros(n)
        self.N = np.zeros(n)

    # Eq. 5 -----------------------------------------------------------------
    def values(self):
        tot = self.U + self.L
        V = np.full(self.n, 0.5)
        np.divide(self.U, tot, out=V, where=tot > 0)
        return V

    # Eqs. 2-3: choice policy -----------------------------------------------
    def choose(self, i, j):
        if self.rng.random() < self.tau:
            x = self.rng.random(2)                                  # Beta(1, 1)
        else:
            x = self.rng.beta(self.U[[i, j]] + 1, self.L[[i, j]] + 1)
        return i if x[0] > x[1] else j

    # Eqs. 4, 6, 7: updating policy -----------------------------------------
    def update(self, i, j, chosen, rewarded):
        ch = chosen
        nc = j if chosen == i else i
        present = np.zeros(self.n, bool)
        present[[i, j]] = True

        # (1) Relaxation (Eq. 4). All four vectors leak by xi; U and L leak
        #     further by xi_R in [0.5, 1], which is near 1 when the item's
        #     recent reward rate r is high (exploit) and near 0.5 when low
        #     (explore): xi_R = r / (r + 1) + 0.5.
        #     Implementation choice (not specified in the paper): an item with
        #     R + N = 0 is given r = 0.5.
        self.R *= self.xi
        self.N *= self.xi
        tot = self.R + self.N
        r = np.full(self.n, 0.5)
        np.divide(self.R, tot, out=r, where=tot > 0)
        xi_R = r / (r + 1.0) + 0.5
        self.U *= xi_R * self.xi
        self.L *= xi_R * self.xi

        # Reward bookkeeping for the two presented items (both get the credit).
        if rewarded:
            self.R[present] += 1
        else:
            self.N[present] += 1

        # Positions are read *after* relaxation. Relaxation scales U and L by
        # the same factor, so V is unchanged by it.
        V = self.values()

        if rewarded:
            # (2a) Eq. 6: consolidate every item, shown or not.
            self.U += V
            self.L += 1.0 - V
            return

        # (2b) Explicit feedback on an error: push the chosen item down and
        #      the unchosen item up.
        self.L[ch] += 1.0
        self.U[nc] += 1.0
        if not self.implicit:
            return

        # (3) Eq. 7: implicit inference for the absent items. Written exactly
        #     as in the paper, per parameter. In the usual case V_ch < V_nc the
        #     three branches partition the line (above / between / below). If
        #     the model's belief is currently inverted (V_ch > V_nc), an item
        #     lying between them gets U += 1 *and* L += 1: it is pushed to
        #     neither side but loses confidence.
        absent = ~present
        above = absent & (V > V[nc])
        below = absent & (V < V[ch])
        between = absent & (V < V[nc]) & (V > V[ch])
        self.U[above] += 1.0
        self.U[between] += V[between]
        self.L[below] += 1.0
        self.L[between] += 1.0 - V[between]


# ---------------------------------------------------------------------------
# Exact choice probability (Eq. 9) — used for noiseless probes
# ---------------------------------------------------------------------------
def beta_pdf_grid(a, b, x=GRID):
    """Beta(a, b) density on the grid, for vectors a, b (shape n x len(x))."""
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[:, None]
    logB = np.vectorize(lgamma)(a) + np.vectorize(lgamma)(b) \
        - np.vectorize(lgamma)(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - logB)


def choice_matrix(model):
    """
    P[i, j] = P(choose i | pair {i, j}) for the current memory state:
        tau/2 + (1 - tau) * integral pdf_i(x) * cdf_j(x) dx       (Eq. 9)
    Computing this exactly, rather than sampling choices, gives a probe that
    does not disturb learning (the paper's "undisruptive probe trials").
    """
    dx = GRID[1] - GRID[0]
    pdf = beta_pdf_grid(model.U + 1, model.L + 1)
    pdf /= pdf.sum(axis=1, keepdims=True) * dx          # renormalise on grid
    cdf = np.cumsum(pdf, axis=1) * dx
    P = (pdf @ cdf.T) * dx                              # P[i, j] = int f_i F_j
    P = model.tau / 2 + (1 - model.tau) * P
    np.fill_diagonal(P, 0.5)
    return P


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def make_schedule(rng):
    sched, phase = [], []
    for _ in range(20):                                  # Phase 1
        block = ADJACENT * 2
        sched += [block[k] for k in rng.permutation(len(block))]
    phase += [1] * len(sched)
    for _ in range(10):                                  # Phase 2
        sched += [ALL_PAIRS[k] for k in rng.permutation(len(ALL_PAIRS))]
    phase += [2] * (len(sched) - len(phase))
    sched += [(5, 6)] * 200                              # Phase 3: FG
    phase += [3] * (len(sched) - len(phase))
    return sched, np.array(phase)


def run_session(seed):
    rng = np.random.default_rng(seed)
    sched, phase = make_schedule(rng)
    m = Betasort(rng=rng)
    T = len(sched)
    V_hist = np.zeros((T + 1, N_ITEMS))
    V_hist[0] = m.values()
    probe_t, probe_P = [], []
    correct = np.zeros(T, bool)
    snapshot = None
    for t, (i, j) in enumerate(sched):
        if t % PROBE_EVERY == 0:
            probe_t.append(t)
            probe_P.append(choice_matrix(m))
        if t == int(np.sum(phase == 1)):                 # moment of transfer
            snapshot = dict(U=m.U.copy(), L=m.L.copy(), P=choice_matrix(m))
        ch = m.choose(i, j)
        rewarded = ch == min(i, j)                       # earlier item wins
        correct[t] = rewarded
        m.update(i, j, ch, rewarded)
        V_hist[t + 1] = m.values()
    return dict(V=V_hist, probe_t=np.array(probe_t), probe_P=np.array(probe_P),
                transfer=snapshot, phase=phase, correct=correct)


# ---------------------------------------------------------------------------
# Figures (all simulation output)
# ---------------------------------------------------------------------------
def caption(fig, text):
    fig.text(0.01, 0.005, "SIMULATION OUTPUT — " + text, fontsize=7.5,
             color="0.35", ha="left", va="bottom")


def shade_phases(ax, phase, y_text=None):
    edges = np.flatnonzero(np.diff(phase)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(phase)]
    labels = ["adjacent pairs", "all pairs", "massed FG"]
    for k, (s, e) in enumerate(zip(starts, ends)):
        if k % 2 == 1:
            ax.axvspan(s, e, color="0.94", zorder=0)
        if y_text is not None:
            ax.text((s + e) / 2, y_text, labels[k], ha="center", va="bottom",
                    fontsize=8.5, color="0.35")


def fig_positions(ex, path):
    """Beta position densities of all 7 items at the moment of transfer."""
    U, L = ex["transfer"]["U"], ex["transfer"]["L"]
    x = np.linspace(0.001, 0.999, 500)
    pdf = beta_pdf_grid(U + 1, L + 1, x)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k in range(N_ITEMS):
        ax.plot(x, pdf[k], color=ITEM_COLORS[k], lw=2)
        mode = x[np.argmax(pdf[k])]
        ax.text(mode, pdf[k].max() * 1.03, ITEMS[k], ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=ITEM_COLORS[k])
    ax.set_xlabel("position on the unit span  (0 = bottom, 1 = top)")
    ax.set_ylabel("density  Beta(U+1, L+1)")
    ax.set_title("Item positions after 240 adjacent-pair trials (one session)")
    ax.set_ylim(0, pdf.max() * 1.18)
    caption(fig, f"seed {EXAMPLE_SEED}, τ={TAU}, ξ={XI}. Only adjacent pairs "
                 "have been shown.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_value_trajectories(res, path):
    """Mean position V_i over the three phases (cf. Jensen et al. Fig 7A)."""
    V = np.mean([r["V"] for r in res], axis=0)
    phase = res[0]["phase"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    shade_phases(ax, phase, y_text=1.01)
    t = np.arange(V.shape[0])
    for k in range(N_ITEMS):
        ax.plot(t, V[:, k], color=ITEM_COLORS[k], lw=2)
        ax.text(t[-1] + 6, V[-1, k], ITEMS[k], va="center", fontsize=10,
                fontweight="bold", color=ITEM_COLORS[k])
    ax.set_xlim(0, t[-1] + 25)
    ax.set_ylim(0, 1.07)
    ax.set_xlabel("trial")
    ax.set_ylabel("position estimate  V = U / (U + L)")
    ax.set_title("Betasort spreads items apart during adjacent-pair training "
                 "and keeps the order under massed FG")
    caption(fig, f"mean over {len(res)} sessions, τ={TAU}, ξ={XI}.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_transfer_matrix(res, path):
    """P(choose earlier item) for all 21 pairs at the moment of transfer."""
    P = np.mean([r["transfer"]["P"] for r in res], axis=0)
    M = np.full((N_ITEMS, N_ITEMS), np.nan)
    for (i, j) in ALL_PAIRS:
        M[i, j] = P[i, j]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.5, vmax=1.0)
    for (i, j) in ALL_PAIRS:
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if M[i, j] > 0.85 else "black")
        if (i, j) in CRITICAL:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec="black", lw=1.6))
    ax.set_xticks(range(N_ITEMS), ITEMS)
    ax.set_yticks(range(N_ITEMS), ITEMS)
    ax.set_xlabel("later item")
    ax.set_ylabel("earlier item")
    ax.set_title("P(correct) at transfer\n(boxed = critical pairs, never trained)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    caption(fig, f"exact Eq. 9 probes, mean of {len(res)} sessions.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_symbolic_distance(res, path):
    """Accuracy at transfer by symbolic distance, critical vs terminal pairs."""
    P = np.array([r["transfer"]["P"] for r in res])           # sessions x 7 x 7
    groups = {
        "non-terminal pairs (B–F)": lambda i, j: i != 0 and j != N_ITEMS - 1,
        "pairs with A or G": lambda i, j: i == 0 or j == N_ITEMS - 1,
    }
    colors = [WARM[2], WARM[1]]
    label_xy = {"non-terminal pairs (B–F)": (4.6, 0.80),
                "pairs with A or G": (4.9, 0.90)}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (name, keep), c in zip(groups.items(), colors):
        ds, mu, lo, hi = [], [], [], []
        for d in range(1, N_ITEMS):
            pairs = [(i, j) for (i, j) in ALL_PAIRS if j - i == d and keep(i, j)]
            if not pairs:
                continue
            per_session = np.mean([P[:, i, j] for (i, j) in pairs], axis=0)
            ds.append(d)
            mu.append(per_session.mean())
            lo.append(np.percentile(per_session, 2.5))
            hi.append(np.percentile(per_session, 97.5))
        ax.fill_between(ds, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(ds, mu, "o-", color=c, lw=2)
        ax.annotate(name, (ds[-1], mu[-1]), xytext=label_xy[name],
                    textcoords="data", color=c, fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.text(1, 0.505, "chance", color="0.5", fontsize=8)
    ax.axvspan(1.5, 4.5, color="0.95", zorder=0)
    ax.text(3, 0.43, "critical pairs: d = 2–4, no A/G", ha="center",
            fontsize=8, color="0.4")
    ax.set_xlim(0.7, 6.4)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("symbolic distance  d = |rank_i − rank_j|")
    ax.set_ylabel("P(choose earlier item)")
    ax.set_title("Symbolic distance effect at the first moment of transfer")
    caption(fig, f"{len(res)} sessions; band = 2.5–97.5% across sessions.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_learning_curves(res, path):
    """Probe accuracy for non-terminal pairs by distance, over all phases."""
    t = res[0]["probe_t"]
    P = np.array([r["probe_P"] for r in res])          # sess x probes x 7 x 7
    phase = res[0]["phase"]
    colors = plt.cm.YlOrRd(np.linspace(0.45, 0.95, 4))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    shade_phases(ax, phase, y_text=1.005)
    for d, c in zip(range(1, 5), colors):
        pairs = [(i, j) for (i, j) in ALL_PAIRS
                 if j - i == d and i != 0 and j != N_ITEMS - 1]
        acc = np.mean([P[:, :, i, j] for (i, j) in pairs], axis=0).mean(axis=0)
        ax.plot(t, acc, color=c, lw=2)
        label = f"d={d}: " + ",".join(pair_name(i, j) for (i, j) in pairs)
        ax.text(t[-1] + 6, acc[-1], label, va="center", fontsize=8.5, color=c)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlim(0, t[-1] + 140)
    ax.set_ylim(0.4, 1.06)
    ax.set_xlabel("trial")
    ax.set_ylabel("P(correct), noiseless probe")
    ax.set_title("Non-terminal pairs: inference emerges before non-adjacent "
                 "pairs are ever shown")
    caption(fig, f"mean of {len(res)} sessions; probes every {PROBE_EVERY} "
                 "trials do not update memory.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def summarise(res):
    P = np.array([r["transfer"]["P"] for r in res]).mean(axis=0)
    phase = res[0]["phase"]
    corr = np.array([r["correct"] for r in res])
    print(f"\nBetasort, τ={TAU}, ξ={XI}, {len(res)} sessions")
    print("Accuracy at transfer (exact probe, mean over sessions):")
    for (i, j) in ALL_PAIRS:
        tag = "  critical" if (i, j) in CRITICAL else ""
        print(f"  {pair_name(i, j)}  d={j - i}  {P[i, j]:.3f}{tag}")
    crit = np.mean([P[i, j] for (i, j) in CRITICAL])
    print(f"Critical-pair mean at transfer: {crit:.3f}")
    for ph, name in [(1, "adjacent"), (2, "all pairs"), (3, "massed FG")]:
        print(f"Observed accuracy, phase {ph} ({name}): "
              f"{corr[:, phase == ph].mean():.3f}")
    Vend = np.mean([r["V"][-1] for r in res], axis=0)
    print("Mean V at end of massed FG:",
          " ".join(f"{ITEMS[k]}={Vend[k]:.2f}" for k in range(N_ITEMS)))
    order_kept = np.mean([np.all(np.diff(r["V"][-1]) < 0) for r in res])
    print(f"Sessions with full A>…>G order intact after massed FG: "
          f"{order_kept:.2%}")


def main():
    FIGDIR.mkdir(exist_ok=True)
    res = [run_session(s) for s in range(N_SESSIONS)]
    ex = res[EXAMPLE_SEED]
    fig_positions(ex, FIGDIR / "fig1_positions_at_transfer.png")
    fig_value_trajectories(res, FIGDIR / "fig2_position_trajectories.png")
    fig_transfer_matrix(res, FIGDIR / "fig3_transfer_matrix.png")
    fig_symbolic_distance(res, FIGDIR / "fig4_symbolic_distance.png")
    fig_learning_curves(res, FIGDIR / "fig5_learning_curves.png")
    summarise(res)
    print(f"\nFigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
