# ti-models
Literature Review & Model Aimulations of Computational Models of transitive inference: implementations and comparisons across value-based and relational learning accounts.

## Theoretical Overview
| Model | Family | Core Mechanism |
|---|---|---|
| Bradley-Terry | Value | Latent scalar, batch MLE |
| ELO | Value | Latent scalar, online SGD |
| Betasort | Value (w/ Uncertainty) | Beta distributions + implicit inference |
| REMERGE | Relational | Recurrent similarity over conjunctive memory |

## Literature Overview
The purpose of this literature review is to understand the fundamental theoretical development of how the Neuroscience field as whole has attempted to answer the transitive inference question. The models proposed differ in numerous ways, but aim to answer when, how, and why the brain can make accurate inferential learning choices in various, but quite similar environments. 

**Transitive inference** is defined as the ability to deduce relationships between items that never explicitly been paired together.

### Neurobiological Background
Four regions recur across the TI literature, each doing distinct work.
Hippocampus. Necessary for premise acquisition in rodent lesion studies. 

Symbolic distance: further-apart items (A vs. E) separate more than adjacent ones (A vs. B). Reads as structure encoded during learning, not integration at judgment time; several studies find no extra hippocampal signal on inference trials specifically.

Prefrontal cortex:

#### The Mechanistic Split
Encoding-based: structure built during learning, read out at test. Predicts the distance effects above, and dominates the human fMRI evidence. Retrieval-based: chained together online at judgment time. Maps onto the table: Bradley-Terry/ELO/Betasort are encoding-based (latent structure accrues, gets read out); REMERGE is built for online integration, retrieval-based.


## Bradley Terry
The Bradley-Terry model was formally introduced in 1952 by statisticians Ralph A. Bradley and Milton E. Terry, though its core mathematical concepts were independently studied decades earlier.  They created the statistical framework to analyze data from paired (head-to-head) comparisons and turn wins and losses into reliable rankings. [1], [2]


## ELO

## Betasort

## REMERGE


## Sources
[1] https://www.alphaxiv.org/abs/2601.14727
[2] https://mbrenndoerfer.com/writing/bradley-terry-model-pairwise-preferences-rankings
[3] https://youtu.be/dg11OwdL3qs?si=N2Fn-TzdqVh9Bfox
