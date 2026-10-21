from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed
from .final_model import feature_indices, smooth

ACTIVITY_NAMES={1:'Walk',2:'Sit',3:'Stand',4:'Sit-to-stand',5:'Stand-to-sit',6:'Turn-right',7:'Turn-left'}

def safe_auc(y,p):
    if len(np.unique(y))<2: return None
    return float(roc_auc_score(y,p))

def safe_ap(y,p):
    if np.sum(y)==0: return None
    return float(average_precision_score(y,p))

def _one_fold(sid,X,y,s,act,task,idx,seed,hard,n_estimators,k):
    tr=s!=sid; te=s==sid
    m=ExtraTreesClassifier(n_estimators=n_estimators,min_samples_leaf=3,max_features=.5,
        class_weight='balanced',n_jobs=1,random_state=seed)
    sw=None
    if hard:
        sw=np.ones(tr.sum(),float); ya=y[tr]; aa=act[tr]; neg=ya==0
        sw[neg & (aa==3)] = 2.0
        sw[neg & np.isin(aa,[6,7])] = 1.5
    m.fit(X[tr][:,idx],y[tr],sample_weight=sw)
    q=m.predict_proba(X[te][:,idx])[:,1]
    temp=np.zeros(len(y)); temp[te]=q; temp=smooth(temp,s,task,te,k=k,causal=False)
    yy=y[te]; pp=temp[te]
    return int(sid), np.flatnonzero(te), pp, {'subject':int(sid),'n':int(te.sum()),'pos':int(yy.sum()),'auroc':safe_auc(yy,pp),'auprc':safe_ap(yy,pp)}

def run(X,y,s,act,task,seed=2,hard=False,n_estimators=200,k=7):
    idx=feature_indices(); pred=np.full(len(y),np.nan,float)
    jobs=Parallel(n_jobs=5,prefer='processes')(delayed(_one_fold)(sid,X,y,s,act,task,idx,seed,hard,n_estimators,k) for sid in sorted(np.unique(s)))
    fold=[]
    for sid,ii,pp,rr in jobs:
        pred[ii]=pp; fold.append(rr)
    valid=np.isfinite(pred)
    subject_aucs=[r['auroc'] for r in fold if r['auroc'] is not None]
    subject_aps=[r['auprc'] for r in fold if r['auprc'] is not None]
    activity={}
    for a,name in ACTIVITY_NAMES.items():
        q=valid & (act==a)
        if q.sum()==0: continue
        activity[name]={'n':int(q.sum()),'pos':int(y[q].sum()),'auroc':safe_auc(y[q],pred[q]),'auprc':safe_ap(y[q],pred[q])}
    return {
      'seed':seed,'hard_negative':hard,'temporal_k':k,
      'pooled':{'n':int(valid.sum()),'pos':int(y[valid].sum()),'auroc':safe_auc(y[valid],pred[valid]),'auprc':safe_ap(y[valid],pred[valid])},
      'subject_macro':{'n_valid_auc_subjects':len(subject_aucs),'auroc_mean':float(np.mean(subject_aucs)),'auroc_std':float(np.std(subject_aucs)),
                       'n_valid_ap_subjects':len(subject_aps),'auprc_mean':float(np.mean(subject_aps)),'auprc_std':float(np.std(subject_aps))},
      'activity':activity,'folds':fold,'pred':pred}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='1,2,3,4,5'); ap.add_argument('--trees',type=int,default=400); a=ap.parse_args()
    z=np.load(a.features); X,y,s,act,task=z['X'],z['y'],z['s'],z['activity'],z['task']
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    allres=[]
    for seed in [int(x) for x in a.seeds.split(',') if x.strip()]:
        for hard in [False,True]:
            r=run(X,y,s,act,task,seed=seed,hard=hard,n_estimators=a.trees,k=7)
            pred=r.pop('pred'); allres.append(r)
            np.savez_compressed(out/f"pred_seed{seed}_{'hard' if hard else 'base'}.npz",y=y,p=pred,s=s,activity=act,task=task)
            print(seed,hard,r['pooled'],r['subject_macro'])
    (out/'loso_results.json').write_text(json.dumps(allres,indent=2),encoding='utf-8')
    rows=[]
    for r in allres:
        rows.append({'seed':r['seed'],'hard_negative':r['hard_negative'],**{f"pooled_{k}":v for k,v in r['pooled'].items()},**{f"macro_{k}":v for k,v in r['subject_macro'].items()}})
    pd.DataFrame(rows).to_csv(out/'loso_summary.csv',index=False)
if __name__=='__main__': main()
