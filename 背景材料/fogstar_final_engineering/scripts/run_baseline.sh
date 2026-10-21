#!/usr/bin/env bash
set -e
PYTHONPATH=. python -m src.train --config configs/baseline.yaml --out results/baseline
