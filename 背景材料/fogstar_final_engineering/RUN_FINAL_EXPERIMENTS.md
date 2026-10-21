# Final reproducible experiment commands

Assume raw `sensor_data.csv` is available locally.

## 1. Build window-level compact features

```bash
python -m src.context_probe \
  --csv /path/to/sensor_data.csv \
  --out results/context_probe_full
```

## 2. Fixed 11/5/6 ACTE-FoG experiment

```bash
python -m src.final_model \
  --features results/context_probe_full/context_features.npz \
  --out results/acte_seed2 \
  --seed 2
```

Use seeds 1-5 for the reported stability experiment.

## 3. Patient-level LOSO confirmatory analysis

```bash
python -m src.confirmatory_loso \
  --features results/context_probe_full/context_features.npz \
  --out results/confirmatory_loso \
  --seeds 2 \
  --trees 300
```

The script writes pooled, subject-macro, activity-stratified and per-subject results. It also runs the optional training-only hard-negative weighting comparison. Ground-truth activity is not consumed at inference.

## 4. Online deployment

Pass `--causal` to `src.final_model` for a causal temporal decoder. Do not report the centered offline decoder as a real-time algorithm.
