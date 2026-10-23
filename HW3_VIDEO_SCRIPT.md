# HW3 Demo Video Script

**Target length:** about 2 minutes 20 seconds

Hello, this is my Homework Three design experiment for Habitat next-action prediction. My task uses the current RGB image, goal distance, and relative goal angle to predict move forward, turn left, turn right, or stop.

My Homework Two baseline immediately concatenated 192 image values with only two goal values. I was concerned that the high-dimensional image could dominate the compact but important goal signal. The baseline also had no direct path that preserved visual information while goal context modified it.

My proposed design is goal-conditioned gated residual fusion. The image and goal are first encoded by separate branches. A joint branch predicts a bounded correction. The goal branch also predicts a sigmoid gate, and the fused feature is the visual representation plus the gate times the correction. I expected this to preserve visual context while making goal information harder to ignore. A possible failure was gate saturation.

I ran a controlled comparison among the original Homework Two baseline, the full gated model, and a no-gate residual ablation. Every model used the same 2,880 samples, episode split, preprocessing, 90 epochs, Adam learning rate, batch size, and cross-entropy loss. I repeated each experiment with five random seeds.

The baseline reached 95.60 percent mean test accuracy. The full design reached 98.70 percent, an improvement of 3.10 percentage points, and its seed-to-seed variation was lower. The largest class-level improvement was move forward, from 76.3 to 93.0 percent.

The validation curves show that both residual variants learned faster and converged above the baseline. The mean confusion matrices also show that most remaining mistakes are forward actions being classified as turns.

The ablation changed my interpretation. Without the gate, the model still reached 98.38 percent, only 0.32 points below the full design. So separate modality encoders and the residual path explain most of the gain, not gating alone.

The gate averaged 0.853, and 52.1 percent of activations were near saturation. It often became an almost-open switch, which matches my predicted failure mode. The full model also has 42 percent more parameters and about 2.8 times the measured single-sample latency, so extra capacity is a confounding factor.

I learned that explicit modality handling and residual fusion are useful for this task, but adding a learned mechanism does not prove that mechanism caused the improvement. My next experiment would use a parameter-matched control and gate-entropy regularization, then evaluate on true scene-disjoint Habitat rollouts. The code, per-seed results, visualizations, and report are all included in the repository. Thank you.

