import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix

def binary_metrics(y,p,thr=.5):
    y=np.asarray(y); p=np.asarray(p); z=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    spec=tn/max(1,tn+fp)
    return {'auroc':roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            'auprc':average_precision_score(y,p) if len(np.unique(y))>1 else np.nan,
            'f1':f1_score(y,z,zero_division=0),'precision':precision_score(y,z,zero_division=0),
            'sensitivity':recall_score(y,z,zero_division=0),'specificity':spec,
            'balanced_accuracy':balanced_accuracy_score(y,z),'mcc':matthews_corrcoef(y,z)}

def best_threshold(y,p):
    best=(0.5,-1)
    for t in np.linspace(.05,.95,181):
        s=f1_score(y,np.asarray(p)>=t,zero_division=0)
        if s>best[1]: best=(float(t),float(s))
    return best[0]
