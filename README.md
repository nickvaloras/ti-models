# ti-models
Literature Review & Model Aimulations of Computational Models of transitive inference: implementations and comparisons across value-based and relational learning accounts.

## Theoretical Overview
| Model | Family | Core Mechanism |
|---|---|---|
| Bradley-Terry | Value | Latent scalar, batch MLE |
| ELO | Value | Latent scalar, online SGD |
| Betasort | Value (w/ Uncertainty) | Implicit positional updating via beta distributions |
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
The Bradley-Terry model was formally introduced in 1952 by statisticians Ralph A. Bradley and Milton E. Terry, though its core mathematical concepts were independently studied decades earlier.  They created the statistical framework to analyze data from paired (head-to-head) comparisons and turn wins and losses into reliable rankings. [1]


## ELO
While Bradley-Terry processes the entire history of matches at once globally using optimization (like Maximum Likelihood Estimation), ELO updates ratings sequentially after each individual match using an online gradient-like rule. Both encode a latent scalar and implicitly assumes a total order, and thus, both will fail in the same specific way when the underlying structure is cyclic (a ring), because neither can assign monotonically increasing values around a loop. [2]

## Betasort
The Betasort model is an extension of the previous two scalar models. In all three of these cases *(Bradley-Terry, ELO, Betasort)*, a scalar value is encoded and updated after feedback is given on the decision made. This means that these models propose the idea that *inference occurs at encoding, not retrieval* Betasort uses the beta distribution to code uncertainty into the scalar value. So now, the value is a distribution, but ultimately still converges to the center of the distribution upon testing pairings, which is why we still refer to this as a latent scalar value. [3]

## REMERGE
The REMERGE model is in opposition to the first three models. It's ultimate claim is that the *inference occurs at retrieval instead of encoding.* This is a direct and unequivocal rejection of the previous studies belief.

REMERGE outlines to layers to the inferential system; the feature layer, or the stimuli, and the conjunctive layer, which stores pattern-separated episode codes (e.g., one unit for the AB pair, one for BC, assuming this is how the network is trained). At retrieval, the conjunctive layer is used to co-activate the stimuli's pairwise connections (being shown B and D activates AB/BC/CD in the conjunctive layer), and through recurrency, the network propagates activation across overlapping items (B is shared by AB and BC) to arrive at a graded response without ever having explicitly encoded B>D. In this model values are not necesary. Instead pairwise memories, given that they know that AB excites A and inhibits B, is the ultimate decision maker here. [4]

## Sources
[1]
[2] Bertrand, Q., Czarnecki, W. M., & Gidel, G. (2023).
    On the limitations of Elo: Real-world games are transitive, not additive.
    Proceedings of the 26th International Conference on Artificial Intelligence
    and Statistics (AISTATS), PMLR 206.
    https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf
    
[3] Jensen, G., Muñoz, F., Alkan, Y., Ferrera, V. P., & Terrace, H. S. (2015).
    Implicit value updating explains transitive inference performance: The Betasort model.
    PLOS Computational Biology, 11(9), e1004523.
    https://doi.org/10.1371/journal.pcbi.1004523

[3] Kumaran, D., & McClelland, J. L. (2012).
    Generalization through the recurrent interaction of episodic memories:
    A model of the hippocampal system.
    Psychological Review, 119(3), 573–616.
    https://doi.org/10.1037/a0028681
