from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

POS=['ankleL','ankleR','back','wrist']; TYPES=['acc','gyro']; AX=list('xyz')
CH=[f'{p}_{t}_{a}' for p in POS for t in TYPES for a in AX]

def build(csv):
    use=CH+['fog','activity','subjectID','sessionID','taskID']
    df=pd.read_csv(csv,usecols=use)
    sos=butter(4,[.5,20],btype='bandpass',fs=60,output='sos')
    freqs=np.fft.rfftfreq(120,d=1/60)
    loco=(freqs>=.5)&(freqs<3); freeze=(freqs>=3)&(freqs<8)
    feats=[]; y=[]; sid=[]; act=[]; task=[]
    for _,g in df.groupby(['subjectID','sessionID','taskID'],sort=False,dropna=False):
        arr=g[CH].to_numpy(np.float64); out=arr.copy()
        for j in range(len(CH)):
            finite=np.isfinite(arr[:,j]); idx=np.flatnonzero(finite)
            cuts=np.r_[0,np.where(np.diff(idx)>1)[0]+1,len(idx)]
            for aa,bb in zip(cuts[:-1],cuts[1:]):
                run=idx[aa:bb]
                if len(run)>30: out[run,j]=sosfiltfilt(sos,arr[run,j])
        fog=g.fog.to_numpy(np.int64); av=g.activity.to_numpy(np.int64)
        for st in range(0,max(0,len(g)-120+1),30):
            z=out[st:st+120].copy(); m=np.isfinite(z)
            mu=np.nanmean(z,axis=0); z=z-mu; z=np.nan_to_num(z)
            sd=np.std(z,axis=0); rms=np.sqrt(np.mean(z*z,axis=0)); mad=np.mean(np.abs(np.diff(z,axis=0)),axis=0)
            sp=np.abs(np.fft.rfft(z,axis=0))**2
            pl=sp[loco].sum(0); pf=sp[freeze].sum(0); fi=np.log1p(pf/(pl+1e-6));
            miss=1-m.mean(0)
            # Cross-body coordination: magnitude correlations for acc/gyro between four nodes
            cross=[]
            for typ_i,typ in enumerate(TYPES):
                mags=[]
                for pi,p in enumerate(POS):
                    base=pi*6+typ_i*3; mags.append(np.sqrt((z[:,base:base+3]**2).sum(1)))
                C=np.corrcoef(np.stack(mags)); C=np.nan_to_num(C)
                cross.extend(C[np.triu_indices(4,1)])
            f=np.r_[sd,rms,mad,fi,miss,np.asarray(cross)]
            feats.append(f.astype(np.float32)); y.append(int(np.bincount(fog[st:st+120]).argmax())); sid.append(int(g.subjectID.iloc[0]));
            vv=av[st:st+120]; vv=vv[vv>0]; act.append(int(np.bincount(vv).argmax()) if len(vv) else 0); task.append(int(g.taskID.iloc[0]))
    return np.asarray(feats),np.asarray(y),np.asarray(sid),np.asarray(act),np.asarray(task)

def met(y,p): return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    X,y,s,act,task=build(a.csv);print('features',X.shape,'pos',y.sum(),flush=True)
    tr=np.isin(s,range(1,12));va=np.isin(s,range(12,17));te=np.isin(s,range(17,23))
    models={
      'extra_trees':ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.7,class_weight='balanced',n_jobs=-1,random_state=1),
      'hist_gb':HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=1)
    }
    res={}
    for name,m in models.items():
      m.fit(X[tr],y[tr]); pv=m.predict_proba(X[va])[:,1];pt=m.predict_proba(X[te])[:,1]
      res[name]={'val':met(y[va],pv),'test':met(y[te],pt)}; print(name,res[name],flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'context_features.npz',X=X,y=y,s=s,activity=act,task=task);(out/'results.json').write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
