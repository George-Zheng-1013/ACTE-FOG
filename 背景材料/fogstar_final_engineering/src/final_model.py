"""Final converged FoG-STAR model: Ankle-Coordination Temporal Ensemble (ACTE-FoG).

Representation:
- bilateral ankle accelerometer+gyroscope multi-scale statistics and spectral freeze features
- compact whole-body acc/gyro magnitude coordination correlations
- explicit missingness features
Classifier: balanced ExtraTrees ensemble
Temporal decoder: centered 7-window moving probability average (offline, ~5 s effective context)
For causal/online use, replace centered smoothing with the causal decoder documented in README.
"""
from pathlib import Path
import argparse, json, numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score


def feature_indices():
    # context_probe feature layout: [std24,rms24,mad24,freeze_index24,missing24,cross12]
    idx=[]
    for off in [0,24,48,72,96]: idx += [off+i for i in range(12)]  # bilateral ankles
    idx += list(range(120,132))  # whole-body coordination summaries
    return idx

def smooth(p,s,task,mask,k=7,causal=False):
    out=p.copy(); inds=np.flatnonzero(mask); st=0
    while st<len(inds):
        en=st+1
        while en<len(inds) and inds[en]==inds[en-1]+1 and s[inds[en]]==s[inds[st]] and task[inds[en]]==task[inds[st]]: en+=1
        ix=inds[st:en]; q=p[ix]
        if causal:
            cs=np.cumsum(np.r_[0.,q]); r=np.empty_like(q)
            for i in range(len(q)):
                a=max(0,i-k+1); r[i]=(cs[i+1]-cs[a])/(i-a+1)
        else:
            r=np.convolve(q,np.ones(k)/k,mode='same'); den=np.convolve(np.ones_like(q),np.ones(k),mode='same'); r=r*k/den
        out[ix]=r; st=en
    return out

def eer_threshold(y,p):
    best=(9,.5)
    for t in np.linspace(0,1,1001):
        z=p>=t; tp=((z)&(y==1)).sum(); fn=((~z)&(y==1)).sum(); tn=((~z)&(y==0)).sum(); fp=((z)&(y==0)).sum()
        d=abs(tp/max(1,tp+fn)-tn/max(1,tn+fp))
        if d<best[0]: best=(d,float(t))
    return best[1]

def metrics(y,p,t):
    z=p>=t; tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'f1':float(f1_score(y,z)),'sensitivity':float(tp/max(1,tp+fn)),
            'specificity':float(tn/max(1,tn+fp)),'accuracy':float((tp+tn)/len(y)),'threshold':float(t)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=2); ap.add_argument('--causal',action='store_true')
    a=ap.parse_args(); z=np.load(a.features); X,y,s,task=z['X'],z['y'],z['s'],z['task']; idx=feature_indices()
    tr=np.isin(s,range(1,12)); va=np.isin(s,range(12,17)); te=np.isin(s,range(17,23))
    model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.5,class_weight='balanced',n_jobs=-1,random_state=a.seed)
    model.fit(X[tr][:,idx],y[tr]); p=np.zeros(len(y)); p[va]=model.predict_proba(X[va][:,idx])[:,1]; p[te]=model.predict_proba(X[te][:,idx])[:,1]
    p=smooth(p,s,task,va,7,a.causal); p=smooth(p,s,task,te,7,a.causal); thr=eer_threshold(y[va],p[va])
    res={'model':'ACTE-FoG','seed':a.seed,'causal':a.causal,'split':{'train':list(range(1,12)),'val':list(range(12,17)),'test':list(range(17,23))},'val':metrics(y[va],p[va],thr),'test':metrics(y[te],p[te],thr)}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps(res,indent=2)); np.savez_compressed(out/'predictions.npz',y=y,p=p,s=s,task=task); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
