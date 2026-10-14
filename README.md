# ti-models
Literature review and simulations of computational models of transitive inference: implementations and comparisons across value-based and relational learning accounts.

## Theoretical Overview
| Model | Family | Core Mechanism | When inference happens | Example in this repo |
|---|---|---|---|---|
| Bradley-Terry | Latent scalar | One strength per item, batch MLE over all comparisons | Encoding | `latent_scalar_sim/`, `tennis_sim/` |
| ELO | Latent scalar | One strength per item, online (SGD-like) update after each comparison | Encoding | `latent_scalar_sim/`, `tennis_sim/` |
| Betasort | Latent scalar position (with uncertainty) | Beta-distributed positions; *every* item updated on every trial | Encoding | `betasort_sim/` |
| REMERGE | Relational / episodic | Recurrent similarity over conjunctive (pair-level) memories | Retrieval | — |

**Phase 2 (planned): Successor Representation (SR) and the Tolman-Eichenbaum Machine (TEM).** These are not added as rows here because they do not fit a single cell of this table (see *Two axes* below).

## Literature Overview
This literature review traces how neuroscience has approached the transitive inference question. The models differ in many ways, but each tries to explain when, how, and why the brain makes accurate inferences in similar relational learning environments.

**Transitive inference (TI)** is the ability to deduce relationships between items that have never been explicitly paired together. In the classic task, items A–G are trained only as adjacent pairs (AB, BC, …, FG), and the critical test is a novel non-adjacent pair such as BD.

### Behavioral signatures
- **Symbolic distance effect:** accuracy and speed increase with the distance between items in the order. For example, B vs F is easier than B vs D, even though neither pair was trained.
- **Terminal item effect:** pairs containing an end item (A or G) are easier, because the end items are always or never rewarded.

These signatures are what every model below has to reproduce. The symbolic distance effect is especially informative, because an expected-value learner (e.g., Q-learning) stays at chance on B–F after adjacent-only training: each of those items is rewarded about 50% of the time (Jensen et al., 2015).

### Neurobiological Background
Four regions recur across the TI literature, each doing distinct work.

**Hippocampus.** Necessary for premise acquisition in rodent lesion studies. *(TODO: citations.)*

The symbolic distance effect reads as structure encoded during learning rather than integration at judgment time; several studies find no extra hippocampal signal on inference trials specifically. *(TODO: citations; this paragraph may belong under the Mechanistic Split.)*

**Prefrontal cortex.**

#### The Mechanistic Split
The models in this repo vary along two independent axes rather than one:

1. **Timing of the inference computation (encoding vs retrieval).**
   - *Encoding-based:* structure is built during learning and read out at test. This predicts the distance effects above and dominates the human fMRI evidence.
   - *Retrieval-based:* premises are chained together online at judgment time.
2. **Representational content (scalar vs structured).**
   - *Scalar:* each item is reduced to one number, or one distribution on a line.
   - *Structured:* the representation keeps relations between items, e.g., a map or graph.

Bradley-Terry, ELO, and Betasort sit in one corner (encoding, scalar). REMERGE sits in the opposite corner (retrieval, structured, in the sense of stored pairwise episodes). The off-diagonal cells are where Phase 2 lives:
- SR's discount parameter γ interpolates along the encoding–retrieval axis.
- TEM factorizes structure from content.

The two families also connect: for a chain, the first non-trivial eigenvector of the SR is approximately the scalar (ELO-like) value code.

## Bradley-Terry & ELO
The Bradley-Terry model was formally introduced in 1952 by the statisticians Ralph A. Bradley and Milton E. Terry [1]. The same model had been studied earlier by Zermelo (1929) for ranking chess players [2]. Bradley-Terry provides a statistical framework for paired (head-to-head) comparison data, turning wins and losses into rankings. Each item *i* has a latent strength *s<sub>i</sub>*, and

P(*i* beats *j*) = σ(*s<sub>i</sub>* − *s<sub>j</sub>*).

ELO [3] uses the same logistic link but estimates strengths differently.
- **Bradley-Terry** fits the entire match history at once, globally, with an optimizer (e.g., maximum likelihood).
- **ELO** updates ratings sequentially after each individual match, using an online rule that is equivalent to a stochastic gradient step on the same logistic loss.

So when the model is correctly specified, BT vs ELO is purely a **batch vs online** contrast.

Both models encode a latent scalar, so both assume a total order. On cyclic structure (a ring), neither can assign values that increase monotonically all the way around the loop, so both should fail there in the same way. Bertrand et al. (2023) show a related limitation: ELO can fail even on *transitive* data when win probabilities are not additive in rating differences [4].

### Example: `latent_scalar_sim/` (NFL season)
Setup:
- 16 teams with evenly spaced hidden strengths (listed order = true hierarchy).
- 10 weeks of binary win/loss outcomes on a random schedule.
- Week 11 is a TI test drawn only from pairs that never played.
- BT is fit by MAP with a Gaussian prior and refit weekly. ELO uses K = 20.

Findings (300 replicated seasons):
- **Both models show a clean symbolic distance effect on never-played pairs**, from about 55% correct at distance 1 to about 100% at distance 12 or more.
- **Neither ranks better than the win–loss record.** BT, ELO, and plain win–loss record recover the true order about equally well (Spearman ≈ 0.81 by week 10). With a balanced random schedule, BT adds little over the record for *ordering*.
- **The real difference is confidence.**
  - BT's predicted probabilities for never-played pairs are calibrated when binned by prediction. Binned by true distance, it looks underconfident, because the prior shrinks estimates toward the mean.
  - ELO is strongly underconfident, because K = 20 is too small for 10 games.

### Example: `tennis_sim/` (ATP players)
Six ATP players serve as a hidden skill ladder. This example introduces Bradley-Terry and ELO fundamentals at a scale small enough to follow by hand.
## Betasort
Betasort [5] is a scalar account like BT and ELO, but it was developed independently in the animal serial-learning tradition rather than as an extension of them. It differs in three ways:

1. **Position, not value.** Each item's position on the unit interval is represented by a beta distribution, Beta(U+1, L+1). Its mean V = U/(U+L) is where the item sits in the order, *not* the probability that choosing it is rewarded.
2. **Asymmetric feedback.**
   - A reward *consolidates*: it adds evidence in proportion to the current position, so the distribution narrows without moving.
   - An error *shifts* the positions of the two items shown.
3. **Implicit updating.** On every trial, *every* item is updated, including items not shown. After an error:
   - items believed above the pair are pushed up;
   - items believed below are pushed down;
   - items in between are consolidated.

   This is what spreads B–F apart before any non-adjacent pair has been seen.

At choice, Betasort draws a sample from each item's distribution and picks the larger. Uncertainty therefore shapes behavior directly; the model does not just read out the means. It is still a latent scalar account, because the state is one distribution per item on a single line. Inference happens at encoding.

### Example: `betasort_sim/` (7-item A–G)
Setup:
- Training: 240 adjacent-pair trials, then 210 all-pair trials, then 200 massed FG-only trials.
- Parameters: τ = 0.05, ξ = 0.95.
- 200 sessions. Accuracy is the exact choice probability (Eq. 9 of the paper), so probes never change memory.

Findings:
- **Transfer is immediate.** At transfer, the critical pairs (BD, CE, DF, BE, CF, BF) are about 92% correct, though none was ever trained. There is a clear symbolic distance effect (d = 2 ≈ 0.88, d = 4 ≈ 0.97) and only a small terminal item effect.
- **Trained adjacent pairs are the hardest pairs**, because neighboring positions overlap most.
- **Implicit updating does the work.** Removing it (the paper's betaQ variant) drops critical-pair accuracy at transfer to about 0.75.
- **Massed FG trials preserve the order but compress the spacing.** The full order survives in about 92% of sessions, but B–F drift upward, because each FG error pushes F *and everything above it* up. Implicit updating protects **order**, not **metric**.

**Relevance to the ring.** Betasort's implicit update depends on "above" and "below" the current pair, which presupposes a line. How it behaves under ring feedback has not been simulated yet.

## REMERGE
REMERGE [6] takes the opposite position from the scalar models: *inference occurs at retrieval instead of encoding.* This is a direct challenge to encoding-based accounts. (Note on dates: REMERGE (2012) predates Betasort (2015), so it is not a response to Betasort specifically.)

REMERGE has two layers:
- **Feature layer:** the stimuli.
- **Conjunctive layer:** pattern-separated episode codes, e.g., one unit for the AB pair and one for BC, assuming this is how the network is trained.

At retrieval, the presented items activate the conjunctive units that contain them. For example, B activates AB and BC, and D activates CD and DE. Through recurrent activity, activation spreads across shared items (C links BC and CD) to arrive at a graded response, without B > D ever having been explicitly encoded. No item values are needed: the stored pairwise memories, and which item in each pair was correct, drive the decision. A symbolic distance effect can emerge from how activation spreads through the chain of episodes.

## Sources
[1] Bradley, R. A., & Terry, M. E. (1952).
    Rank analysis of incomplete block designs: I. The method of paired comparisons.
    Biometrika, 39(3/4), 324–345.
    https://doi.org/10.2307/2334029

[2] Zermelo, E. (1929).
    Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der
    Wahrscheinlichkeitsrechnung. Mathematische Zeitschrift, 29, 436–460.

[3] Elo, A. E. (1978).
    The Rating of Chessplayers, Past and Present. New York: Arco.

[4] Bertrand, Q., Czarnecki, W. M., & Gidel, G. (2023).
    On the limitations of Elo: Real-world games are transitive, not additive.
    Proceedings of the 26th International Conference on Artificial Intelligence
    and Statistics (AISTATS), PMLR 206.
    https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf

[5] Jensen, G., Muñoz, F., Alkan, Y., Ferrera, V. P., & Terrace, H. S. (2015).
    Implicit value updating explains transitive inference performance: The Betasort model.
    PLOS Computational Biology, 11(9), e1004523.
    https://doi.org/10.1371/journal.pcbi.1004523

[6] Kumaran, D., & McClelland, J. L. (2012).
    Generalization through the recurrent interaction of episodic memories:
    A model of the hippocampal system.
    Psychological Review, 119(3), 573–616.
    https://doi.org/10.1037/a0028681
