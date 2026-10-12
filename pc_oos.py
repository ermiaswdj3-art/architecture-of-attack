"""Out-of-sample pitch-control test. Tune (speed, sigma, lambda) on Game 1 only, then evaluate on Game 2 untouched.
Same pass-success label as pc_validate.py. Vectorised re-implementation of pitch_control_full (same equations)."""
import numpy as np, pandas as pd, math, itertools
from scipy.stats import rankdata
L, W = 105.0, 68.0
def load(game, side):
    p = f'tracking/Sample_Game_{game}_RawTrackingData_{side}_Team.csv'
    df = pd.read_csv(p, header=None, skiprows=3); ids = [str(v) for v in pd.read_csv(p, header=None, nrows=3).iloc[2]]
    cols = {n:(df[j].values*L, df[j+1].values*W) for j,n in enumerate(ids) if n.startswith('Player')}
    return df[1].values.astype(int), cols, (df[len(ids)-2].values*L, df[len(ids)-1].values*W)
def build(game):
    fh, H, bh = load(game,'Home'); fa, A, ba = load(game,'Away'); fidx={f:i for i,f in enumerate(fh)}
    ev = pd.read_csv(f'tracking/Sample_Game_{game}_RawEventsData.csv'); out=[]
    def st(cols,i):
        P=[];V=[]
        for x,y in cols.values():
            if np.isnan(x[i]) or np.isnan(y[i]): continue
            j0,j1=max(i-1,0),min(i+1,len(x)-1); dx,dy=x[j1]-x[j0],y[j1]-y[j0]
            P.append((x[i],y[i])); V.append((0,0) if np.isnan(dx) or np.isnan(dy) else (dx/.08,dy/.08))
        return np.array(P),np.array(V)
    for k in range(len(ev)-1):
        e=ev.iloc[k]
        if e['Type']!='PASS' or np.isnan(e['End X']) or int(e['Start Frame']) not in fidx: continue
        i=fidx[int(e['Start Frame'])]; n=ev.iloc[k+1]
        ok=int(n['Team']==e['Team'] and n['Type']!='BALL LOST')
        hp,hv=st(H,i); ap,av=st(A,i); b=(bh[0][i],bh[1][i])
        if len(hp)<10 or len(ap)<10 or np.isnan(b[0]): continue
        tgt=np.array([e['End X']*L,e['End Y']*W]); stt=np.array([e['Start X']*L,e['Start Y']*W])
        out.append(dict(period=int(e['Period']),hp=hp,hv=hv,ap=ap,av=av,ball=np.array(b),tgt=tgt,len=np.linalg.norm(tgt-stt),home=e['Team']=='Home',ok=ok))
    return out
def control(d,speed,sigma,lam,react=0.7,bs=15.0,dt=0.04):
    def T(p,v): r=p+v*react; return react+np.linalg.norm(d['tgt']-r,axis=1)/speed
    Th,Ta=T(d['hp'],d['hv']),T(d['ap'],d['av']); Tall=np.concatenate([Th,Ta]); nh=len(Th)
    t=np.linalg.norm(d['tgt']-d['ball'])/bs; tot=0.0; ppcf=np.zeros(len(Tall)); k=np.pi/(math.sqrt(3)*sigma)
    for _ in range(250):
        if tot>=0.99: break
        ppcf+=(1-tot)/(1+np.exp(-k*(t-Tall)))*lam*dt; tot=ppcf.sum(); t+=dt
    ph=ppcf[:nh].sum()/max(ppcf.sum(),1e-9); return ph if d['home'] else 1-ph
def auc(s,y):
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def evalset(data,params):
    s=np.array([control(d,*params) for d in data]); y=np.array([d['ok'] for d in data]); return s,y
if __name__=='__main__':
    g1,g2=build(1),build(2); print('passes G1',len(g1),'G2',len(g2))
    y1=np.array([d['ok'] for d in g1]); y2=np.array([d['ok'] for d in g2])
    print(f'success rate G1 {y1.mean():.3f} G2 {y2.mean():.3f}')
    default=(5.0,0.45,4.3); s,_=evalset(g2,default); print(f'G2 default constants: AUC {auc(s,y2):.3f}')
    best=None
    for sp,sg,lm in itertools.product([4.5,5.0,6.0],[0.3,0.45,0.7],[2.0,4.3,8.0]):
        s,_=evalset(g1,(sp,sg,lm)); a=auc(s,y1)
        if best is None or a>best[0]: best=(a,(sp,sg,lm))
    print('G1 best params',best[1],f'G1 AUC {best[0]:.3f}')
    s,_=evalset(g2,best[1]); a=auc(s,y2); rng=np.random.default_rng(0)
    bs=[auc(s[i],y2[i]) for i in (rng.integers(0,len(y2),len(y2)) for _ in range(500))]
    print(f'G2 OUT-OF-SAMPLE with G1-tuned params: AUC {a:.3f} (95% CI {np.percentile(bs,2.5):.3f}-{np.percentile(bs,97.5):.3f})')
    ln=np.array([d['len'] for d in g2]); print(f'G2 pass-length baseline AUC {auc(-ln,y2):.3f}')
