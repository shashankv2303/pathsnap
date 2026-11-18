# PathSnap candidate-scoring models

PathSnap compared five classifiers to score nearby road segments for each GPS
observation. The team chose **HistGradientBoosting** for tuning and route
reconstruction. In the original benchmark it had the highest candidate recall
on student test and fitted much faster than the other nonlinear models. Random
Forest led on student-test Top-1 and cross-validation Macro F1.

Each classifier receives the same 18 features for a GPS observation and road
candidate, including distance, heading agreement, and road-network context. A
positive label means the candidate matches the aligned reference road. Arc
IDs are local to each trip and identify candidates throughout the pipeline.
The classifier scores individual candidates. Viterbi combines those scores
with estimates of how plausible it is to travel between consecutive candidates.

## Why we tried each model

| Model | Why we included it | What the benchmark showed |
| --- | --- | --- |
| [Logistic Regression](logistic_regression.py) | A scaled, regularized linear baseline. It tests how far an interpretable weighted combination of the features can get. The run used `C=1`, balanced class weights, and a 1,000-iteration cap. | Fastest full fit and highest scoring throughput. Its student-test Top-1 and Macro F1 were the lowest of the five, supporting a comparison with nonlinear models. |
| [Random Forest](random_forest.py) | An ensemble of trees trained on resampled data. It tests whether nonlinear feature interactions improve candidate scoring. The run used 100 trees, maximum depth 16, and balanced subsample weights. | Highest CV candidate Macro F1 and student-test Top-1. It also had the longest full fit, which made repeated search more expensive. |
| [Extra Trees](extra_trees.py) | A second tree ensemble that randomizes split thresholds. This tests whether more random splits give a useful speed or generalization tradeoff. The run used 100 trees and maximum depth 16. | Highest mean CV Top-1. It took 101.0 s to fit, between HistGradientBoosting and Random Forest, and scored 81.95% Top-1 on student test. |
| [HistGradientBoosting](hist_gradient_boosting.py) | Boosted trees learn feature interactions by adding trees in sequence to reduce errors. Binning numeric features keeps split finding fast. The baseline used learning rate 0.08, up to 150 iterations, 31 maximum leaves, and balanced class weights. | Highest student-test candidate recall at the 0.5 threshold. Its full fit took 22.5 s, close to Logistic Regression's 20.9 s and about 15 times faster than Random Forest's. |
| [Multi-Layer Perceptron](mlp.py) | A feed-forward neural network tests a different way to learn feature interactions. The run used scaled inputs, hidden layers of 64 and 32 units, and L2 penalty `alpha=0.001`. | Highest student-test candidate Macro F1 and lowest log loss in the original benchmark. It had the lowest recall at 0.5 and a substantially longer full fit than HistGradientBoosting. |

## Original five-model benchmark

The [comparison CSV](../reports/full_model_comparison.csv) records five grouped
validation folds across 80 training trips. Each model was then refitted on
the training partition and scored on the ten student-test trips. **Candidate
Macro F1** gives equal weight to the match and non-match classes. **Top-1**
asks whether each GPS observation's highest-scoring candidate matches the
reference. Recall below is candidate recall at a probability threshold of 0.5.

| Model | CV candidate Macro F1 | Student Top-1 | Student candidate Macro F1 | Student recall | Student log loss ↓ | Full fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7889 | 80.75% | 0.7937 | 92.75% | 0.2650 | 20.9 s |
| Random Forest | **0.8587** | **82.68%** | 0.8779 | 92.68% | 0.1583 | 335.9 s |
| Extra Trees | 0.8496 | 81.95% | 0.8641 | 93.28% | 0.2010 | 101.0 s |
| HistGradientBoosting | 0.8441 | 81.98% | 0.8606 | **94.26%** | 0.1828 | 22.5 s |
| Multi-Layer Perceptron | 0.8497 | 82.33% | **0.8908** | 79.60% | **0.1264** | 291.3 s |

## Why HistGradientBoosting went forward

HistGradientBoosting's baseline Top-1 was within 0.70 percentage points of
Random Forest's on student test. Its much shorter fit made repeated grouped
cross-validation practical. Its strong thresholded recall was another useful
baseline diagnostic. Viterbi uses continuous match scores for every candidate,
so changing the classifier threshold has no effect on which states it considers.

## After Bayesian optimization

Optuna's TPE sampler ran eight trials on five grouped folds across all 80
training trips. It selected the settings with the highest mean validation
candidate Macro F1. The winning model used a learning rate of 0.04415,
80 boosting iterations, at most 27 leaves per tree, at least 35 samples per
leaf, L2 regularization of 0.17342, and no class weighting.

Both configurations were refitted on all 80 training trips, then scored on
the same ten student-test trips. Internal early stopping was disabled for both.

| Metric | Baseline | Tuned |
| --- | ---: | ---: |
| CV candidate Macro F1 | 0.8436 | **0.8662** |
| Student-test candidate Macro F1 | 0.8599 | **0.8935** |
| Student-test pooled observation Top-1 | 81.71% | **82.13%** |
| Student-test binary log loss ↓ | 0.1827 | **0.1254** |
| Student-test candidate recall at 0.5 | **94.29%** | 80.75% |

The tuned model improved ranking and log loss while sacrificing recall at the
0.5 classification threshold. The decoder receives continuous scores from the
tuned model for every candidate. [Trial history](../../results/bayesian-full/optuna_trials.csv) ·
[Tuning comparison](../../results/bayesian-full/comparison.csv)

## After HMM and Viterbi

The hidden state is the candidate road segment at each GPS observation. The
HMM uses the classifier score as emission evidence and penalizes transitions
where the directed road distance differs from the observed GPS displacement.
Viterbi finds the sequence with the highest combined score and backtracks to
recover the selected roads. The decoder inserts connecting arcs where a
plausible directed path exists and records breaks across GPS gaps, missing
candidates, or impossible transitions.

| Mean per student-test trip | Tuned classifier only | With Viterbi |
| --- | ---: | ---: |
| Pointwise accuracy | 83.88% | **89.95%** |
| Route continuity | 91.40% | **99.81%** |
| Normalized route edit distance ↓ | 0.529 | **0.181** |

Viterbi improved student-test pointwise accuracy by **6.07 percentage points**
with the same candidate sets and classifier. [Decoder comparison](../../results/hmm-viterbi/comparison.csv) ·
[Decoder settings](../../results/hmm-viterbi/decoder_config.json)

### Post-Viterbi accuracy by split

| Split | Trips | Mean per-trip accuracy | Pooled GPS-point accuracy |
| --- | ---: | ---: | ---: |
| Training | 80 | 87.68% | 87.82% |
| Validation, five held-out folds | 80 | 87.18% | 86.88% |
| Student test | 10 | 89.95% | 87.83% |
| Evaluator test | 10 | 90.11% | 84.83% |

Both accuracy columns use observations that could be aligned to the reference.
Mean per-trip accuracy gives each trip equal weight. Pooled accuracy gives
each scorable GPS point equal weight. The training score uses the final model
fitted on all 80 trips. Validation refits the classifier per fold and scores
its held-out trips with settings already selected during development, so it
is a post-tuning estimate. The evaluator results come from a separate final
run with the frozen model and decoder.

## Reproduce the comparison

From the repository root, with the candidate datasets and dependencies available:

```bash
python -m pathsnap.experiments.cross_validation
```

See the [root README](../../README.md) for candidate data, tuning, and decoder
commands and the [phase summary](../../results/phase3-phase4-summary.md) for
the measured tuning and route-decoding results.
