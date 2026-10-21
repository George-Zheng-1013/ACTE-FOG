from __future__ import annotations
import numpy as np, pandas as pd

CLIN=['age','h_y','updrs_iii','fog_q']

def matched_11_5_6(sensor_df, clinical_df, seed=20260916):
    """Deterministic subject-independent 11/5/6 split, clinically balanced approximately.
    The publication states matching by age/H&Y/UPDRS-III/FoG-Q but does not expose subject IDs
    in the materials verified here. This approximates that rule; exact official IDs can be passed
    with --split-json when known.
    """
    rng=np.random.default_rng(seed)
    c=clinical_df.copy().set_index('subjectID')
    fog_by_sub=sensor_df.groupby('subjectID')['fog'].max().rename('freezer')
    c=c.join(fog_by_sub).fillna({'freezer':0})
    subjects=np.array(sorted(c.index.astype(int)))
    X=c.reindex(subjects)[[x for x in CLIN if x in c]].apply(pd.to_numeric,errors='coerce')
    X=X.fillna(X.median()).to_numpy(float)
    X=(X-X.mean(0))/(X.std(0)+1e-6)
    freezer=c.reindex(subjects)['freezer'].to_numpy(int)
    target=[11,5,6]
    # multi-start greedy minimization of standardized centroid and freezer-rate mismatch
    best=None
    for _ in range(4000):
        perm=rng.permutation(len(subjects)); groups=[perm[:11],perm[11:16],perm[16:]]
        if any(freezer[g].sum()==0 or (1-freezer[g]).sum()==0 for g in groups): continue
        score=sum(np.square(X[g].mean(0)-X.mean(0)).mean() for g in groups)
        score+=sum((freezer[g].mean()-freezer.mean())**2 for g in groups)
        if best is None or score<best[0]: best=(score,groups)
    if best is None:
        p=rng.permutation(len(subjects)); groups=[p[:11],p[11:16],p[16:]]
    else: groups=best[1]
    return {'train':subjects[groups[0]].tolist(),'val':subjects[groups[1]].tolist(),'test':subjects[groups[2]].tolist()}
