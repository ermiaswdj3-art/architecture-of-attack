"""Pitch control vs distance-to-nearest-opponent: paired bootstrap of the AUC difference, per game (untuned constants 5.0 / 0.45 / 4.3)."""
import numpy as np
from scipy.stats import rankdata
import pc_oos as P
def auc(s,y): r=rankdata(s);n1=y.sum();n0=len(y)-n1;return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
rng=np.random.default_rng(0)
for g in (1,2):
    ctrl=[];opp=[];y=[]
    for d in P.build(g):
        o=d['ap'] if d['home'] else d['hp']
        ctrl.append(P.control(d,5.0,0.45,4.3)); opp.append(np.linalg.norm(o-d['tgt'],axis=1).min()); y.append(d['ok'])
    ctrl,opp,y=np.array(ctrl),np.array(opp),np.array(y)
    diff=auc(ctrl,y)-auc(opp,y); bs=[]
    for _ in range(500):
        i=rng.integers(0,len(y),len(y)); bs.append(auc(ctrl[i],y[i])-auc(opp[i],y[i]))
    print(f'Game {g}: n={len(y)}  AUC control {auc(ctrl,y):.3f}  nearest-opponent {auc(opp,y):.3f}  diff {diff:+.3f}  95% CI {np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f}')
