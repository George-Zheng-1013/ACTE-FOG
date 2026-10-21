from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .models import OfficialCNN

CH=['ankleL_gyro_x','ankleL_gyro_y','ankleL_gyro_z']

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def build_windows(csv, group_level='task', drop_missing=True):
    use=CH+['fog','activity','subjectID','sessionID','taskID']
    df=pd.read_csv(csv,usecols=use)
    sos=butter(4,[0.5,20.0],btype='bandpass',fs=60,output='sos')
    groups=['subjectID','sessionID','taskID'] if group_level=='task' else (['subjectID','sessionID'] if group_level=='session' else ['subjectID'])
    X=[];Y=[];S=[]; A=[]; nskip=0
    for keys,g in df.groupby(groups,sort=False,dropna=False):
        arr=g[CH].to_numpy(np.float64); fog=g.fog.to_numpy(np.int64); act=g.activity.to_numpy(np.int64)
        # zero-phase filtering separately inside each recording and contiguous finite run
        out=arr.copy()
        for j in range(3):
            finite=np.isfinite(arr[:,j]); idx=np.flatnonzero(finite)
            cuts=np.r_[0,np.where(np.diff(idx)>1)[0]+1,len(idx)]
            for a,b in zip(cuts[:-1],cuts[1:]):
                run=idx[a:b]
                if len(run)>30:
                    out[run,j]=sosfiltfilt(sos,arr[run,j])
        sid=int(g.subjectID.iloc[0]); w=120; stride=30
        for st in range(0,max(0,len(g)-w+1),stride):
            z=out[st:st+w].copy()
            if drop_missing and not np.isfinite(z).all():
                nskip+=1; continue
            mu=np.nanmean(z,axis=0,keepdims=True); z=z-mu
            z=np.nan_to_num(z,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
            y=int(np.bincount(fog[st:st+w]).argmax())
            av=act[st:st+w]; av=av[av>0]; aa=int(np.bincount(av).argmax()) if len(av) else 0
            X.append(z);Y.append(y);S.append(sid);A.append(aa)
    return np.asarray(X,np.float32),np.asarray(Y,np.int64),np.asarray(S,np.int64),np.asarray(A,np.int64),nskip

def equal_error_threshold(y,p):
    best=(1e9,.5)
    for t in np.linspace(0,1,1001):
        z=p>=t; tp=((z==1)&(y==1)).sum(); fn=((z==0)&(y==1)).sum(); tn=((z==0)&(y==0)).sum(); fp=((z==1)&(y==0)).sum()
        sen=tp/max(1,tp+fn); spe=tn/max(1,tn+fp)
        d=abs(sen-spe)
        if d<best[0]: best=(d,float(t))
    return best[1]

def metrics(y,p,thr):
    z=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),threshold=float(thr),
        sensitivity=float(tp/max(1,tp+fn)),specificity=float(tn/max(1,tn+fp)),accuracy=float((tp+tn)/len(y)),
        f1=float(f1_score(y,z,zero_division=0)),precision=float(precision_score(y,z,zero_division=0)))

def pred(model,X,batch,dev):
    model.eval(); out=[]
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=batch):
            lg,_=model(x.to(dev));out.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(out)

def l2_penalty(model):
    s=0.
    # Keras model kernel_regularizer L2=0.001 on conv1-4 and fc1, not final output
    for name,m in model.named_modules():
        if isinstance(m,(nn.Conv1d,nn.Linear)) and name!='fc2': s=s+(m.weight*m.weight).sum()
    return s

def run(X,Y,S,seed=1,drop_l2=False,internal_eval=True,early_stop=True,max_epochs=120):
    seed_all(seed); dev=torch.device('cpu')
    qtr=np.isin(S,np.arange(1,12)); qv=np.isin(S,np.arange(12,17)); qt=np.isin(S,np.arange(17,23))
    X11,y11=X[qtr],Y[qtr]
    if internal_eval:
        idx=np.arange(len(y11)); i_tr,i_ev=train_test_split(idx,test_size=.4,stratify=y11,random_state=seed)
        xtr,ytr=X11[i_tr],y11[i_tr]; xev,yev=X11[i_ev],y11[i_ev]
    else:
        xtr,ytr=X11,y11; xev,yev=X[qv],Y[qv]
    model=OfficialCNN(3).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.0004)
    bce=nn.BCEWithLogitsLoss()
    ld=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=64,shuffle=True)
    best_loss=1e99; best=None; bad=0; hist=[]
    for ep in range(1,max_epochs+1):
        model.train(); ls=[]
        for x,y in ld:
            opt.zero_grad(); lg,_=model(x.to(dev)); loss=bce(lg,y.float().to(dev))
            if not drop_l2: loss=loss+0.001*l2_penalty(model)
            loss.backward(); opt.step(); ls.append(loss.item())
        # evaluation loss (including regularizer, Keras val_loss includes it)
        model.eval(); evloss=[]
        with torch.no_grad():
            evld=DataLoader(TensorDataset(torch.from_numpy(xev),torch.from_numpy(yev)),batch_size=256)
            for x,y in evld:
                lg,_=model(x.to(dev)); loss=bce(lg,y.float().to(dev))
                if not drop_l2: loss=loss+0.001*l2_penalty(model)
                evloss.append(loss.item())
        vl=float(np.mean(evloss)); hist.append((ep,float(np.mean(ls)),vl))
        if vl < best_loss-1e-3:
            best_loss=vl; bad=0; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        if early_stop and bad>=10: break
    if best is not None: model.load_state_dict(best)
    pv=pred(model,X[qv],256,dev); pt=pred(model,X[qt],256,dev)
    # paper computes EER threshold on each evaluated distribution; for honest test also report val-fixed threshold
    tv=equal_error_threshold(Y[qv],pv)
    return {'seed':seed,'epochs':len(hist),'val':metrics(Y[qv],pv,tv),'test_valthr':metrics(Y[qt],pt,tv),'test_eer':metrics(Y[qt],pt,equal_error_threshold(Y[qt],pt))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--out',required=True); ap.add_argument('--group-level',default='task');ap.add_argument('--keep-missing',action='store_true');ap.add_argument('--seeds',default='1,2,3,4,5')
    a=ap.parse_args(); X,Y,S,A,nskip=build_windows(a.csv,a.group_level,not a.keep_missing)
    print('pack',X.shape,'positive',Y.sum(),'skipped_missing',nskip)
    rs=[]
    for seed in map(int,a.seeds.split(',')):
        r=run(X,Y,S,seed); rs.append(r);print(json.dumps(r))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'windows_leftgyro.npz',X=X,Y=Y,S=S,A=A)
    (out/'runs.json').write_text(json.dumps(rs,indent=2))
    for split,mask in [('train',np.isin(S,np.arange(1,12))),('val',np.isin(S,np.arange(12,17))),('test',np.isin(S,np.arange(17,23)))]:
        print(split,'n',mask.sum(),'pos',Y[mask].sum(),'pct',Y[mask].mean())
    for key in ['val','test_valthr','test_eer']:
        for m in ['auroc','f1','sensitivity','specificity']:
            z=np.array([r[key][m] for r in rs]); print(key,m,z.mean(),z.std(ddof=1) if len(z)>1 else 0)
if __name__=='__main__':main()
