# HW3 Experiment Summary: Goal-Conditioned Gated Residual Fusion

## Problem and hypothesis

My PointNav task predicts the next navigation action—`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, or `STOP`—from an 8 × 8 RGB observation plus relative goal distance and angle. The HW2 baseline immediately concatenates 192 image values with only two goal values, then passes the 194-dimensional vector through a 64–32 MLP. A limitation is that this early fusion treats two very different modalities identically: the high-dimensional image can dominate compact but decision-critical goal information.

I designed a **goal-conditioned gated residual fusion** model. It separately maps RGB to a 64-dimensional visual representation `v` and the goal to a 16-dimensional representation `g`. A joint branch predicts a bounded correction `c = tanh(Wc[v,g])`; a goal-dependent gate produces `a = sigmoid(Wg g)`; fusion is `z = v + a ⊙ c`. A 64–32 classifier maps `z` to four logits. My hypothesis was that the direct path would preserve scene context while the learned gate would apply only useful goal-conditioned corrections. A possible failure was sigmoid saturation, which could make the gate unresponsive and block useful gradients.

## Controlled experimental design

I compared: (1) the original HW2 early-fusion MLP, (2) the full gated residual design, and (3) a no-gate ablation using `z = v + c`. All models used the identical 2,880-sample dataset, episode-level 70/15/15 split, preprocessing, 90 epochs, batch size 128, Adam at learning rate 0.0025, and cross-entropy. I repeated each model with five seeds (7400–7404). Primary metrics were held-out accuracy and macro F1; I also recorded per-class accuracy, confusion matrices, parameter count, latency, and gate statistics.

| Model | Parameters | Test accuracy | Macro F1 | Test loss |
| --- | ---: | ---: | ---: | ---: |
| HW2 baseline | 14,692 | 95.60 ± 1.25% | 93.72 ± 2.01% | 0.1063 ± 0.0148 |
| Goal-gated residual | 20,884 | **98.70 ± 0.56%** | **98.27 ± 0.77%** | **0.0325 ± 0.0048** |
| No-gate ablation | 19,796 | 98.38 ± 0.33% | 97.85 ± 0.52% | 0.0406 ± 0.0109 |

## Observation and analysis

The full model improved mean test accuracy by 3.10 percentage points over HW2 and reduced seed variance. The clearest class-level change was `MOVE_FORWARD`: 76.3% for the baseline, 93.0% for the gated model, and 89.7% for the no-gate ablation. Since forward is assigned only within a narrow goal-angle region, this is consistent with the separate goal branch preserving goal information that early fusion sometimes underuses.

The ablation changes the interpretation. Removing the gate reduced mean accuracy by only 0.32 points, so most of the improvement came from modality-specific encoding plus the residual path—not from adaptive gating alone. The learned gates averaged 0.853, and 52.1% were near saturation (<0.1 or >0.9). Thus, the gate often became an almost-open switch. The design succeeded at the system level but only partly supported my mechanism-level hypothesis.

There are important trade-offs. The full model uses 20,884 parameters versus 14,692 (+42%) and measured 0.051 ms versus 0.018 ms per single CPU example. Because parameter count was not matched, extra capacity is a confound. The synthetic expert policy and reused image frames also limit generalization to real Habitat trajectories.

## Conclusion and next experiment

Explicitly separating visual and goal information was beneficial, and residual fusion made learning more consistent. The learned gate added a small further gain but saturated too often to claim that selective control was the main cause. Next, I would compare against a parameter-matched two-branch model and add a gate-entropy penalty or temperature schedule so the gate remains responsive early in training. I would then test on true Habitat rollouts with scene-disjoint splits.
