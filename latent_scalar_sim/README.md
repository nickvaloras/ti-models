# latent_scalar_sim: Bradley-Terry & Elo (NFL example)

Folder layout: `latent_scalar_sim/` → `figures/`, `results/`, `simulation/` (package: teams, simulate, models/{base, bradley_terry, elo}, evaluate, plots) + `run_nfl.py`.

## Current design (v2, decided with Nicholas)
- 16 teams: NE, KC, PHI, GB, SEA, PIT, DAL, DEN, CHI, MIN, NO, NYG, HOU, ARI, NYJ, CLE (listed order = hidden true hierarchy).
- 10-week season, 8 games/week, no rematches, random schedule (tau=inf; adjustable). Binary W/L only, no margins, no conferences/divisions/playoffs.
- World: evenly spaced strengths, P(i beats j) = sigmoid(s_i − s_j), spread 3.0 logits (best team ~80% vs average; adjacent 55%). Models correctly specified → BT vs Elo = purely batch vs online.
- Week 11 = TI test week: 8 games drawn only from the 40 never-played pairs.
- Why 16/10: all per-team figures stay readable; 32/16 changes rating precision little and breaks matrix/trajectory figures.

## Models
- BT: MAP, Gaussian prior sd 1.0, Newton, Laplace SE, refit weekly.
- Elo: K=20, start 1500, no MOV, optional multiple passes.

## Figures (9)
results matrix, ranking bump chart, ratings on one Elo scale, week-by-week trajectories, probability matrices, symbolic distance (played vs never-played), week-11 test, learning curves, calibration.

## Findings (seed 7 + 300 replicated seasons)
- Both show a clean symbolic distance effect on never-played pairs (~55% at distance 1 → ~100% at 12+).
- Rank recovery: BT ≈ Elo ≈ plain win-loss record (Spearman ~0.81 at week 10). With a balanced random schedule, BT adds little over the record for ordering.
- Main difference is confidence: BT's never-played-pair probabilities are well calibrated when binned by prediction; Elo is strongly underconfident (K=20 too small for 10 games). |P−P_true| at week 10: BT 0.125, Elo 0.18.
- By true distance, BT still looks underconfident: the prior shrinks estimates toward the mean, but conditioned on its own prediction BT is calibrated.
- Seed 7 week 11: 4 upsets in 8 games; each model 5/8 correct.

## Candidate next steps
- Unbalanced schedules (tau small / clustered) where record ≠ BT, strength change mid-season, game-order effects, K and prior sweeps, Elo multiple passes, terminal-item analysis, non-additive models.