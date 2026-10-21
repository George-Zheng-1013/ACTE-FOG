from __future__ import annotations
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from dataclasses import dataclass

POSITIONS = ["ankleL","ankleR","back","wrist"]
SIX = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

@dataclass
class WindowPack:
    x: np.ndarray          # N,T,C
    mask: np.ndarray       # N,T,C, 1 observed
    y: np.ndarray
    activity: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    task: np.ndarray
    start: np.ndarray


def channel_names(sensor_set):
    out=[]
    for s in sensor_set:
        if s.endswith('_gyro'):
            p=s[:-5]; out += [f"{p}_gyro_{a}" for a in ['x','y','z']]
        elif s.endswith('_acc'):
            p=s[:-4]; out += [f"{p}_acc_{a}" for a in ['x','y','z']]
        else:
            out += [f"{s}_{m}" for m in SIX]
    return out


def _safe_filter(a, fs, low, high):
    # Filtering each contiguous finite run avoids spreading NaNs across a recording.
    out=a.astype(np.float64, copy=True)
    finite=np.isfinite(out)
    if finite.sum()<32: return out
    idx=np.flatnonzero(finite)
    cuts=np.r_[0, np.where(np.diff(idx)>1)[0]+1, len(idx)]
    sos=butter(4,[low,high],btype='bandpass',fs=fs,output='sos')
    for i,j in zip(cuts[:-1],cuts[1:]):
        run=idx[i:j]
        if len(run)>30:
            try: out[run]=sosfiltfilt(sos,out[run])
            except ValueError: pass
    return out


def preprocess(df, channels, fs=60, low=.5, high=20):
    df=df.copy()
    for c in channels:
        df[c]=_safe_filter(df[c].to_numpy(),fs,low,high)
    return df


def make_windows(df, channels, fs=60, window_seconds=2., overlap=.75, label_rule='majority'):
    w=int(round(fs*window_seconds)); stride=max(1,int(round(w*(1-overlap))))
    xs=[]; ms=[]; ys=[]; acts=[]; subs=[]; sess=[]; tasks=[]; starts=[]
    # never make a window across subject/session/task boundaries
    for (sid,se,tid),g in df.groupby(['subjectID','sessionID','taskID'],sort=False,dropna=False):
        g=g.reset_index(drop=True)
        arr=g[channels].to_numpy(np.float32)
        fog=g['fog'].to_numpy(np.int64)
        activity=g['activity'].fillna(0).to_numpy(np.int64)
        for st in range(0,max(0,len(g)-w+1),stride):
            en=st+w; z=arr[st:en].copy(); mask=np.isfinite(z)
            # official-style per-window de-meaning, but NaN-aware and train-safe
            mu=np.nanmean(z,axis=0,keepdims=True); mu=np.nan_to_num(mu,nan=0.0)
            z=z-mu; z=np.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0)
            lab=int(np.mean(fog[st:en])>=.5) if label_rule=='majority' else int(fog[st:en].max())
            av=activity[st:en]; av=av[av>0]; act=int(np.bincount(av).argmax()) if len(av) else 0
            xs.append(z); ms.append(mask.astype(np.float32)); ys.append(lab); acts.append(act)
            subs.append(int(sid)); sess.append(int(se) if pd.notna(se) else -1); tasks.append(int(tid) if pd.notna(tid) else -1); starts.append(st)
    return WindowPack(np.asarray(xs,np.float32),np.asarray(ms,np.float32),np.asarray(ys,np.int64),
                      np.asarray(acts,np.int64),np.asarray(subs),np.asarray(sess),np.asarray(tasks),np.asarray(starts))
