"""fair_ctrl_v2: controlled CNN-vs-ExtraTrees comparison for FoG-STAR.

Every experiment in this package reads one shared processed tensor (T), one
observation mask (M) and one window table (W). Model branches never filter,
demean, scale, fill or re-label on their own.

Layout (relative to the engineering root):
    src/fair_v2/        this package
    results/fair_ctrl_v2/   outputs (ignored by git)
"""

__all__ = ["schema", "base", "views", "features", "models", "evaluate", "run"]
