# LightGBM and small neural net (#79, built on #78)

From `backend`, install `requirements.txt`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.evaluate --models top90,rel90,november,xgb,lightgbm,lightgbm-tuned,small-nn,small-nn-tuned
```

Use `--data path/to/Prices.csv` to change the source. Data loading and future
Keepa-format support belong to R1/T1; neither adapter loads data itself.

Both adapters consume exactly `ml.features.FEATURES`: 18 inputs with the shared
8% drop / next-14-days label. They implement R1's fit/score protocol and return
probabilities, enabling Brier and calibration error alongside ranking metrics.
Nothing is saved over the served model.

## Defaults and tuning

LightGBM uses library defaults except seed 42, one worker and quiet logging.
The small neural net is scikit-learn's MLPClassifier: ReLU hidden layers 32/16,
Adam, maximum 100 epochs, seed 42. Other optimizer settings use sklearn defaults.
This is a small-model baseline, not the library's default 100-unit architecture.
Median imputation (all-empty columns retained as zero) and standard scaling are
fitted on training rows only. LightGBM handles missing values natively.
Nonfinite inputs become missing values in both adapters.

Each tuned variant fits eight candidates, including its default. LightGBM varies
leaves (15/31), learning rate (.03/.1), and L2 penalty (0/2); the neural net varies
layers (32 or 32/16), L2 penalty (.0001/.01), and learning rate (.001/.003).
The budget means candidate fits, not equal CPU time. Coordinate this eight-fit
budget with the owners of the other four R2 models before the final comparison.

Select the highest validation average precision (PR-AUC); first candidate wins
ties. This targets rare-drop ranking without selecting a favorable wait rate or
letting expensive products dominate tuning. Dollar metrics remain reported at
all three wait rates. Validation never fits preprocessing or weights. No random
internal validation split or early stopping is enabled for the neural net;
100 epochs is a cap and sklearn may stop earlier on training-loss convergence.
Convergence warnings indicate that the cap was reached and must be reviewed.

The fitted adapter exposes `trials_`, `best_params_`, and `validation_pr_auc_`
for the eventual R1/T1 results log. Current R1 CLI still evaluates only its legacy
chronological split. Multiple windows, product-cluster error bars, product
breakdowns, and shared-log integration remain framework dependencies; do not
interpret this initial leaderboard as completing those requirements.

## Prior work

Reviewed #46 and #47 (already closed) and `lightgbm-model`'s training script.
The older branch provides a binary LightGBM precedent but uses separate feature,
label, split and threshold choices. The new adapter retains the binary classifier
approach and uses the shared current pipeline for comparable results. No old
saved weights or threshold rules are reused.

## Checks

`python -m pytest tests/test_tabular.py tests/test_bench.py tests/test_clean_model.py -q`
checks probability outputs, reproducibility, equal tuning budgets, validation
selection, train-only preprocessing, and the existing benchmark leakage guards.

## Initial legacy-split run

See `R2_INITIAL_RESULTS.txt` for the complete initial leaderboard (all three wait
rates). On 41,289 test rows / 218 products, tuned LightGBM PR-AUC was .533 and
tuned neural net .447. This is an initial single-split comparison, without error
bars or a claim of superiority to served XGBoost. Several neural-net fits reached
the 100-epoch cap. The existing quantile threshold scorer can slightly exceed a
target wait rate when scores tie; the report includes actual wait rates.

Run environment: Python 3.12, sklearn 1.8.0, LightGBM 4.7.0, numpy 2.3.5.
Focused checks: 23 passed. After installing backend requirements, the full backend
suite: 49 passed, two Keepa tests failed because local MySQL was unavailable
(and the failed startup import left `main.keepa_fetch` unset).

## Training-cap check and XGBoost comparison

`check_nn_training.py` is a validation-only diagnostic. Reproduce it from backend:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.bench.check_nn_training --epochs 100 400 --output nn_training_check.json
```

For the fixed baseline architecture and seed, increasing the cap from 100 to 400
reduced training loss from .2513 to .2257, but validation PR-AUC fell from .3893
to .3580. Both runs reached the cap. More optimization did not improve validation
ranking; keep the existing 100-epoch recipe. This checks the baseline, not all
eight configurations, and does not prove convergence or an optimal epoch count.
No test scores were computed by this diagnostic. The record is in
`R2_NN_TRAINING_CHECK.json`. A future validation-based stopping strategy could
address overfitting, but is a separate protocol change.

`R2_XGB_COMPARISON.txt` compares current LightGBM variants with the served model
and XGBoost retrained on the same legacy split. Served XGBoost PR-AUC is .540,
retrained XGBoost .541, and tuned LightGBM .533. At 20% target wait, dollars caught
per decision are $5.316, $5.243 and $5.328 respectively. These differences do not
establish a winner without R1 uncertainty estimates. XGBoost uses its existing
recipe, not an equal eight-trial search, so this is a reference comparison, not
the final equal-budget six-model study. Frozen served scores are raw classifier
probabilities, as in R1; this is not the app's calibrated operating threshold.

The earlier legacy test results have already been inspected. These are development
comparisons on that same holdout, not new independent final evidence. Any future
model decisions should use validation; confirm findings on R1's additional
held-out periods. The served model and its artifacts have not been replaced.
