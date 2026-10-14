# betasort_sim — Betasort on the classic 7-item TI task

A self-contained teaching simulation of **betasort** (Jensen, Muñoz, Alkan, Ferrera & Terrace, 2015, *PLoS Comput Biol* 11(9): e1004523). It uses only numpy and matplotlib and does not import from the model package.

```
betasort_sim/
├── run_betasort.py        model + task + simulation figures (fig1–fig5)
├── concept_figures.py     schematic figures (concept1–concept2)
├── results_summary.txt    console output of run_betasort.py
└── figures/
```

```bash
python concept_figures.py
python run_betasort.py      # ~25 s; 200 sessions
```

## Task

The hidden order is **A > B > C > D > E > F > G**. Choosing the earlier item is rewarded. This follows the paper's monkey procedure, plus its Fig 7 massed-trial test:

| Phase | Pairs | Trials |
|---|---|---|
| 1. Adjacent training | AB, BC, CD, DE, EF, FG (each pair twice per block) | 20 blocks × 12 = 240 |
| 2. All pairs | all 21 pairs | 10 blocks × 21 = 210 |
| 3. Massed FG | FG only | 200 |

**Transfer** is the first trial of Phase 2. The **critical pairs** are BD, CE, DF, BE, CF and BF. They are non-adjacent, contain no terminal item, and are never trained before transfer.

## The model, built up from the update rules

**Representation.** Each item *i* carries two leaky evidence counts, U<sub>i</sub> ("upper") and L<sub>i</sub> ("lower"). Its position on the unit span is believed to be Beta(U<sub>i</sub>+1, L<sub>i</sub>+1). The best point estimate is V<sub>i</sub> = U<sub>i</sub>/(U<sub>i</sub>+L<sub>i</sub>), and V = 0.5 when both counts are 0. **V is a position, not P(reward | i).** Under adjacent training, B–F are each rewarded about 50% of the time. An expected-value learner (Q-learning, Rescorla–Wagner) therefore stays at chance on BD, while a position learner does not.

**Choice (Eqs 2–3).** With probability τ, both values are drawn from a uniform distribution. Otherwise, one value is drawn from each item's beta distribution. The larger draw wins. The exact choice probability (Eq 9) is

P(choose a | {a,b}) = τ/2 + (1−τ) ∫ f<sub>a</sub>(x) F<sub>b</sub>(x) dx.

All accuracy figures use this exact probability, computed on a grid. Probes therefore never change memory.

**Update, step 1: relaxation (Eq 4).** Every count leaks by ξ (the recall parameter). U and L also leak by an extra factor, ξ<sub>R</sub> = r/(r+1) + 0.5, where r = R/(R+N) is the item's recent reward rate. ξ<sub>R</sub> lies in [0.5, 1]. When the item is going well, it stays near 1, so memory is kept and the model exploits. When the item is going badly, it drops toward 0.5, so memory is forgotten and the model explores. U and L shrink by the same factor, so relaxation **widens the beliefs without moving them**.

**Update, step 2a: reward (Eq 6).** For *every* item, shown or not, U += V and L += 1−V. This adds one unit of evidence, split in proportion to the current position:

(U+V)/(U+L+1) = V

The mean is unchanged and the density narrows. This is **consolidation**.

**Update, step 2b: error, explicit part.** The chosen item gets L<sub>ch</sub> += 1 and the unchosen item gets U<sub>nc</sub> += 1. A whole unit goes on one side, so the mean **shifts**.

**Update, step 3: error, implicit part (Eq 7).** This step applies to the five items that were *not* shown:

| Where the item sits | Update | Effect |
|---|---|---|
| above the pair (V<sub>i</sub> > V<sub>nc</sub>) | U += 1 | pushed up |
| below the pair (V<sub>i</sub> < V<sub>ch</sub>) | L += 1 | pushed down |
| between the pair | U += V, L += 1−V | consolidated |

This step is what spreads B–F apart before any non-adjacent pair has been seen (`concept2_one_error_trial.png`). Without it, the model is the paper's *betaQ*, which shows much weaker transfer.

Parameters: τ = 0.05 and ξ = 0.95. These are the values the paper uses for its 200-trial adjacent-training simulation (Fig 6), similar to its best-performing human participants.

## Figures

| File | Type | What it shows |
|---|---|---|
| `concept1_position_code.png` | schematic | what (U, L) encode; consolidation vs shift |
| `concept2_one_error_trial.png` | schematic | one CE error, step by step: relax → explicit → implicit |
| `fig1_positions_at_transfer.png` | simulation | the seven beta densities after adjacent training (one session) |
| `fig2_position_trajectories.png` | simulation | mean V across all three phases (cf. paper Fig 7A) |
| `fig3_transfer_matrix.png` | simulation | P(correct) for all 21 pairs at transfer; critical pairs boxed |
| `fig4_symbolic_distance.png` | simulation | accuracy at transfer by distance, non-terminal vs A/G pairs |
| `fig5_learning_curves.png` | simulation | probe accuracy for non-terminal pairs by distance, all phases |

## Findings (200 sessions, τ = 0.05, ξ = 0.95)

- **Transfer is immediate.** At the first moment of transfer, the critical pairs average **0.92** correct, although none has ever been trained. By distance, BD/CE/DF are about 0.88–0.89, BE/CF about 0.95, and BF about 0.97. Fig 5 shows the distance ordering emerging steadily *during* adjacent-only training.
- **The symbolic distance effect is clear. The terminal-item effect is small.** At d = 1, AB and FG reach about 0.79, compared with about 0.73–0.77 for BC–EF. The paper reports the same overall picture: betasort tends to *over*-perform on critical pairs relative to monkeys.
- **Trained pairs are the hardest.** Adjacent pairs are the worst-performing pairs at transfer. Observed accuracy during Phase 1 is 0.68. This comes from the spatial code: neighbours overlap the most.
- **Massed FG keeps the order but squeezes the spacing.** After 200 FG-only trials, the full A > … > G order is intact in 91.5% of sessions. However, every item from B to F drifts upward (Fig 2). The reason is that each error on FG (choosing G) applies U += 1 to F *and* to every item above F, while A is already at the ceiling. Adjacent non-terminal accuracy falls from about 0.84 to 0.74. In other words, implicit updating protects the **order** rather than the **metric**.

## Implementation choices not fixed by the paper

1. **Items with R + N = 0 get r = 0.5** in ξ<sub>R</sub>. This only matters before an item's first appearance; with blocked training, that is at most the first block.
2. **Eq 7 is applied literally, parameter by parameter.** When the model's belief is inverted (V<sub>ch</sub> > V<sub>nc</sub>), an item lying between them meets both the "above" and "below" conditions. It gets U += 1 *and* L += 1, so it moves to neither side but becomes less certain. Check this against the S1 pseudocode if exact replication matters.
3. **Choice probabilities are computed by grid quadrature** with 1999 points; numerical error is far below the session-to-session variation.

## Where this sits relative to the ring project

Betasort is a scalar/value-ranking account. Its state is a set of positions on a bounded interval, and its implicit update relies on "above/below the pair", which only makes sense for a linear order. A ring has no above or below. Seam pairs are the clearest point where this representation and a ring diverge. How betasort *behaves* when given ring feedback has not yet been simulated here; that would be a natural next step rather than an assumed result.

## Possible extensions

- Lesion the implicit step (`Betasort(implicit=False)`) to get betaQ and reproduce the paper's contrast.
- Sweep τ and ξ, and look at the τ–ξ trade-off the paper notes for fitting.
- Feed the model a line-then-ring schedule (Map2-style) and look at seam pairs.
