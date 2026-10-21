#!/usr/bin/env bash
set -e
PYTHONPATH=. python -m src.train --config configs/final_model.yaml --out results/final
