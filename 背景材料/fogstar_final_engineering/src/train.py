from __future__ import annotations
import argparse, json, os, random, yaml, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from .data import channel_names, preprocess, make_windows
from .split import matched_11_5_6
from .models import OfficialCNN, MSCoordNet
from .metrics import binary_metrics,best_threshold

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def subset(pack,subjects):
    q=np.isin(pack.subject,subjects); return q

def loader(pack,q,batch,shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(pack.x[q]),torch.from_numpy(pack.mask[q]),torch.from_numpy(pack.y[q]),torch.from_numpy(pack.activity[q])),batch_size=batch,shuffle=shuffle)

def predict(model,ld,dev):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,m,y,a in ld:
            logit,_=model(x.to(dev),m.to(dev)); ys.extend(y.numpy()); ps.extend(torch.sigmoid(logit).cpu().numpy())
    return np.asarray(ys),np.asarray(ps)

def run(cfg,outdir,split_override=None):
    seed_all(cfg['seed']); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d=cfg['data']; raw=pd.read_csv(d['sensor_csv']); clinical=pd.read_csv(d['clinical_csv'])
    ch=channel_names(d['sensor_set']); missing=[c for c in ch if c not in raw.columns]
    if missing: raise ValueError(f'Missing channels: {missing}')
    raw=preprocess(raw,ch,d['fs'],d['filter_low'],d['filter_high'])
    pack=make_windows(raw,ch,d['fs'],d['window_seconds'],d['overlap'],d.get('label_rule','majority'))
    split=split_override or matched_11_5_6(raw,clinical,cfg['seed'])
    qt,qv,qe=[subset(pack,split[k]) for k in ['train','val','test']]
    # Final model expects exactly 24 signal channels. Baseline 3.
    name=cfg['model']['name']
    model=OfficialCNN(len(ch)) if name=='official_cnn' else MSCoordNet(cfg['model'].get('hidden',48),cfg['model'].get('graph_heads',4),cfg['model'].get('dropout',.25))
    model.to(dev); tr=cfg['train']
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay'])
    ytr=pack.y[qt]; pos=max(1,int(ytr.sum())); neg=max(1,int((1-ytr).sum())); pw=torch.tensor([neg/pos],device=dev,dtype=torch.float32)
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    best_auc=-1; bad=0; best_state=None
    train_ld=loader(pack,qt,tr['batch_size'],True); val_ld=loader(pack,qv,tr['batch_size'],False)
    hist=[]
    for ep in range(1,tr['epochs']+1):
        model.train(); losses=[]
        for x,m,y,a in train_ld:
            x,m,y,a=x.to(dev),m.to(dev),y.float().to(dev),a.to(dev)
            opt.zero_grad(); logit,aux=model(x,m); loss=bce(logit,y)
            if name!='official_cnn':
                valid=a>0
                if valid.any(): loss=loss+cfg['model'].get('aux_activity_weight',.2)*torch.nn.functional.cross_entropy(aux['activity_logits'][valid],a[valid])
                # transition supervision: windows containing both labels are boundary-rich; approximate from raw majority confidence unavailable here,
                # therefore regularize transition logit toward FoG label only weakly.
                loss=loss+cfg['model'].get('transition_weight',.1)*bce(aux['transition_logit'],y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); losses.append(loss.item())
        yv,pv=predict(model,val_ld,dev); met=binary_metrics(yv,pv); hist.append({'epoch':ep,'loss':float(np.mean(losses)),**met})
        auc=met['auroc'] if np.isfinite(met['auroc']) else -1
        if auc>best_auc+1e-4: best_auc=auc; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        if bad>=tr.get('patience',15): break
    model.load_state_dict(best_state); yv,pv=predict(model,val_ld,dev); thr=best_threshold(yv,pv)
    test_ld=loader(pack,qe,tr['batch_size'],False); yt,pt=predict(model,test_ld,dev); tm=binary_metrics(yt,pt,thr)
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'best.pt'); pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
    pd.DataFrame({'y':yt,'p':pt}).to_csv(out/'test_predictions.csv',index=False)
    summary={'device':str(dev),'channels':ch,'n_windows':len(pack.y),'split':split,'threshold':thr,'val':binary_metrics(yv,pv,thr),'test':tm}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); ap.add_argument('--split-json')
    a=ap.parse_args(); cfg=yaml.safe_load(open(a.config,encoding='utf-8')); sp=json.load(open(a.split_json)) if a.split_json else None
    print(json.dumps(run(cfg,a.out,sp),indent=2))
if __name__=='__main__': main()
