"""Kickoff symmetry numbers, trivial-baseline test for pitch control, and print-safe figures for Part IV."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import pc_oos as P
L,W=105.0,68.0
def auc(s,y): r=rankdata(s);n1=y.sum();n0=len(y)-n1;return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
# ---- 1. kickoff symmetry --------------------------------------------------------------------------
for g,f in ((1,1),(2,51)):
    fh,H,bh=P.load(g,'Home'); fa,A,ba=P.load(g,'Away'); i=int(np.where(fh==f)[0][0])
    hx=np.array([v[0][i] for v in H.values() if not np.isnan(v[0][i])]); ax=np.array([v[0][i] for v in A.values() if not np.isnan(v[0][i])])
    hy=np.array([v[1][i] for v in H.values() if not np.isnan(v[0][i])]); ay=np.array([v[1][i] for v in A.values() if not np.isnan(v[0][i])])
    own_h=(hx<L/2).sum() if hx.mean()<L/2 else (hx>L/2).sum(); own_a=(ax<L/2).sum() if ax.mean()<L/2 else (ax>L/2).sum()
    print(f'Game {g} kickoff frame {f}: players H {len(hx)} A {len(ax)}; centroid x H {hx.mean():.1f} m, A {ax.mean():.1f} m; mirror gap |H-(105-A)| {abs(hx.mean()-(L-ax.mean())):.1f} m; centroid y H {hy.mean():.1f} A {ay.mean():.1f}; players in own half H {own_h}/11 A {own_a}/11; ball ({bh[0][i]:.1f},{bh[1][i]:.1f}) m')
# ---- 2. trivial baseline: distance to nearest opponent at the pass target -------------------------
def feats(data):
    X=[];y=[]
    for d in data:
        opp=d['ap'] if d['home'] else d['hp']; mate=d['hp'] if d['home'] else d['ap']
        no=np.linalg.norm(opp-d['tgt'],axis=1).min(); nm=np.linalg.norm(mate-d['tgt'],axis=1).min()
        X.append([P.control(d,4.5,0.45,8.0) if False else P.control(d,5.0,0.45,4.3),no,nm,d['len']]); y.append(d['ok'])
    return np.array(X),np.array(y)
for g in (1,2):
    X,y=feats(P.build(g)); print(f'\nGame {g}: n={len(y)}')
    for nm,s in (('pitch control',X[:,0]),('distance to nearest opponent',X[:,1]),('distance to nearest teammate (neg)',-X[:,2]),('pass length (neg)',-X[:,3])): print(f'  AUC {nm:36s} {auc(s,y):.3f}')
    for nm,cols in (('nearest opponent + length',[1,3]),('control + length',[0,3]),('nearest opponent + control + length',[0,1,3])):
        Z=(X[:,cols]-X[:,cols].mean(0))/X[:,cols].std(0); r=[auc(cross_val_predict(LogisticRegression(),Z,y,cv=StratifiedKFold(5,shuffle=True,random_state=s),method='predict_proba')[:,1],y) for s in range(5)]
        print(f'  CV AUC {nm:38s} {np.mean(r):.3f}')
