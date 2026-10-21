from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class OfficialCNN(nn.Module):
    """PyTorch equivalent of the public 120x3 Keras reference architecture."""
    def __init__(self,in_ch=3):
        super().__init__()
        self.c1=nn.Conv1d(in_ch,16,9,padding=8)
        self.c2=nn.Conv1d(16,16,7,padding=6)
        self.c3=nn.Conv1d(16,24,5,padding=4)
        self.c4=nn.Conv1d(24,24,5,padding=4)
        self.drop=nn.Dropout(.4); self.fc1=nn.Linear(24,48); self.fc2=nn.Linear(48,1)
    @staticmethod
    def causal(x,k): return x[..., :-(k-1)] if k>1 else x
    def forward(self,x,mask=None):
        x=x.transpose(1,2)
        x=F.relu(self.causal(self.c1(x),9)); x=F.relu(self.causal(self.c2(x),7)); x=F.max_pool1d(x,4,4); x=self.drop(x)
        x=F.relu(self.causal(self.c3(x),5)); x=F.max_pool1d(x,3,3); x=self.drop(x)
        x=F.relu(self.causal(self.c4(x),5)); x=x.mean(-1)
        x=self.drop(F.relu(self.fc1(x))); return self.fc2(x).squeeze(-1), {}

class DepthSep(nn.Module):
    def __init__(self,c,k,d=1):
        super().__init__(); p=d*(k-1)//2
        self.dw=nn.Conv1d(c,c,k,padding=p,dilation=d,groups=c,bias=False); self.pw=nn.Conv1d(c,c,1,bias=False)
        self.bn=nn.BatchNorm1d(c)
    def forward(self,x): return F.gelu(self.bn(self.pw(self.dw(x))))

class NodeEncoder(nn.Module):
    def __init__(self,in_ch=12,h=48):
        super().__init__(); self.proj=nn.Conv1d(in_ch,h,1)
        self.b1=DepthSep(h,5); self.b2=DepthSep(h,11); self.b3=DepthSep(h,7,2)
        self.mix=nn.Conv1d(h*3,h,1); self.norm=nn.BatchNorm1d(h)
    def forward(self,x):
        x=self.proj(x); y=self.mix(torch.cat([self.b1(x),self.b2(x),self.b3(x)],1)); return F.gelu(self.norm(x+y))

class MSCoordNet(nn.Module):
    """Missing-aware multi-scale dynamic coordination model.
    Input channel order is 4 nodes x 6 signals; mask is concatenated node-wise (6+6).
    Dynamic cross-node attention models coordination, while a transition head regularizes
    rapid changes in inter-node state. Activity is learned only as an auxiliary target,
    avoiding use of ground-truth activity at inference.
    """
    def __init__(self,h=48,heads=4,drop=.25,n_activity=8):
        super().__init__(); self.h=h
        self.enc=nn.ModuleList([NodeEncoder(12,h) for _ in range(4)])
        self.attn=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True)
        self.norm=nn.LayerNorm(h); self.drop=nn.Dropout(drop)
        self.fog=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Dropout(drop),nn.Linear(h,1))
        self.activity=nn.Linear(h,n_activity)
        self.transition=nn.Linear(h,1)
    def forward(self,x,mask=None):
        if mask is None: mask=torch.ones_like(x)
        # x: B,T,24 => 4 x 6. each encoder sees value + missingness.
        nodes=[]; temporal=[]
        for i in range(4):
            z=x[:,:,i*6:(i+1)*6]; m=mask[:,:,i*6:(i+1)*6]
            e=self.enc[i](torch.cat([z,m],-1).transpose(1,2)) # B,H,T
            temporal.append(e); nodes.append(e.mean(-1))
        n=torch.stack(nodes,1) # B,4,H
        a,w=self.attn(n,n,n,need_weights=True,average_attn_weights=False)
        n=self.norm(n+self.drop(a)); pooled=torch.cat([n.mean(1),n.max(1).values],-1)
        fog=self.fog(pooled).squeeze(-1)
        act=self.activity(n.mean(1))
        # transition magnitude proxy from pooled first differences in latent signals
        tt=torch.stack(temporal,1).mean(1); d=(tt[:,:,1:]-tt[:,:,:-1]).abs().mean(-1)
        tr=self.transition(n.mean(1)).squeeze(-1)
        return fog, {'activity_logits':act,'transition_logit':tr,'graph_attention':w,'latent_delta':d}
