# FoG-STAR Final Engineering

## Research position
The engineering package deliberately separates two questions:
1. **Published-comparable baseline**: single ankle gyroscope, 120x3 input, 2 s window, 75% overlap, official-style 1D-CNN.
2. **Final research model (MSCoordNet)**: all four IMUs (24 signal channels) with missing-aware multi-scale temporal encoding, dynamic cross-body coordination attention, auxiliary activity prediction, and a transition regularizer.

The final research model does **not consume ground-truth activity at inference**. Activity is auxiliary supervision only, preventing label/context leakage.

## Why this is the converged model
- Keeps the strong published temporal CNN inductive bias instead of replacing it with a large Transformer.
- Uses all 24 channels only where they add a specific hypothesis: cross-body coordination.
- Explicitly models sensor missingness, which is non-trivial in FoG-STAR.
- Multi-scale temporal filters target shuffling/trembling/akinetic patterns with different temporal signatures.
- Dynamic node attention yields interpretable ankle↔ankle, ankle↔back, back↔wrist coordination weights.
- Activity is predicted as an auxiliary task, intended to reduce turning/FoG confusion without using oracle labels.
- Keeps event/severity/long-context extensions outside the core classifier until they demonstrate stable gains in ablation.

## Data
Put official Zenodo files at:
```
data/sensor_data.csv
data/clinical_data.csv
```
FoG-STAR schema: 24 IMU channels + activity/fog/fog_severity/subjectID/sessionID/taskID, 60 Hz.

## Run
```bash
python -m pip install -r requirements.txt
bash scripts/run_baseline.sh
bash scripts/run_final.sh
```
To reproduce an exact published split, provide a JSON file with `train`, `val`, `test` subject ID arrays and append:
```
--split-json configs/official_split.json
```
The verified publication describes clinically matched 11/5/6 groups but the exact subject-ID list was not present in the sources retrieved during this build, so the default code creates a deterministic clinically balanced 11/5/6 approximation. Do not call that approximation an exact paper reproduction.

## Outputs
Each run saves checkpoint, epoch history, raw test probabilities, selected validation threshold, split IDs, and final metrics.

## Primary metrics
AUROC (paper comparability), AUPRC, F1, sensitivity, specificity, balanced accuracy, MCC. Threshold is selected on validation only.

## Required experiment order
1. Official-like left-ankle gyro baseline.
2. Right-ankle gyro baseline and best single ankle.
3. Both ankles.
4. All 4 IMUs without coordination attention (ablation to add).
5. MSCoordNet full.
6. Activity-aux off/on.
7. Missing-mask off/on.
8. Dynamic graph off/on.
9. Patient-level bootstrap / paired test.
10. Only after core gains are stable: dense event decoding, severity ordinal head, long-context transition analysis.

## Scientific guardrail
Do not report a gain until the baseline reaches the expected published neighborhood under a genuinely comparable split/preprocessing protocol. If baseline is materially below ~0.91 AUROC, diagnose protocol mismatch before interpreting the proposed model.

## 2026-09-17 converged experimental model: ACTE-FoG
Exploration on the uploaded FoG-STAR v3.0 data converged to an ankle-centric hybrid rather than the original heavy MSCoordNet.

ACTE-FoG uses bilateral ankle acc+gyro window descriptors (STD, RMS, mean absolute derivative, 3-8 Hz / 0.5-3 Hz freeze-index, missingness), compact whole-body magnitude-correlation descriptors, a balanced ExtraTrees ensemble, and a 7-window temporal probability decoder. The centered decoder is for offline clinical analysis; `--causal` provides an online-compatible decoder.

Generate features:
```bash
python -m src.context_probe --csv data/sensor_data.csv --out results/context_probe
```
Run final model:
```bash
python -m src.final_model --features results/context_probe/context_features.npz --out results/final_acte
```
See `results/FINAL_RESULTS.md` for the current experimental summary and confirmatory-evaluation caveat.
