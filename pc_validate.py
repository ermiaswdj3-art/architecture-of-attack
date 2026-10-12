"""Pre-stated test: does pitch control at a pass's destination (at kick time) predict whether the pass succeeds?
Success = the next event is by the same team and is not BALL LOST. Baseline: pass length alone.
Pitch control = pitch_control_full.integrated_pitch_control (literature-default constants, not fit)."""
import sys, numpy as np, pandas as pd, math
sys.path.insert(0, 'pc/pitch_control_release')
import pitch_control_full as pcf
from scipy.stats import rankdata
L, W = 105.0, 68.0
def load(side):
    df = pd.read_csv(f'tracking/Sample_Game_1_RawTrackingData_{side}_Team.csv', header=None, skiprows=3)
    names = pd.read_csv(f'tracking/Sample_Game_1_RawTrackingData_{side}_Team.csv', header=None, nrows=3)
    ids = [str(v) for v in names.iloc[2].tolist()]
    t = df[2].values; cols = {}
    for j, n in enumerate(ids):
        if n.startswith('Player'): cols[n] = (df[j].values * L, df[j+1].values * W)
    return df[1].values.astype(int), t, cols, (df[len(ids)-2].values*L, df[len(ids)-1].values*W)
fh, th, H, bh = load('Home'); fa, ta, A, ba = load('Away')
fidx = {f: i for i, f in enumerate(fh)}
ev = pd.read_csv('tracking/Sample_Game_1_RawEventsData.csv')
def state(cols, i):
    pos, vel = {}, {}
    for p, (x, y) in cols.items():
        if np.isnan(x[i]) or np.isnan(y[i]): continue
        pos[p] = (x[i], y[i])
        j0, j1 = max(i-1, 0), min(i+1, len(x)-1)
        dx, dy = x[j1]-x[j0], y[j1]-y[j0]
        vel[p] = (0.0, 0.0) if np.isnan(dx) or np.isnan(dy) else (dx/0.08, dy/0.08)
    return pos, vel
rows = []
for k in range(len(ev)-1):
    e = ev.iloc[k]
    if e['Type'] != 'PASS' or np.isnan(e['End X']) or e['Start Frame'] not in fidx: continue
    i = fidx[int(e['Start Frame'])]
    nxt = ev.iloc[k+1]; ok = int(nxt['Team'] == e['Team'] and nxt['Type'] != 'BALL LOST')
    hp, hv = state(H, i); ap, av = state(A, i)
    if len(hp) < 10 or len(ap) < 10: continue
    ball = (bh[0][i], bh[1][i])
    if np.isnan(ball[0]): continue
    tgt = (e['End X']*L, e['End Y']*W)
    ph, _ = pcf.integrated_pitch_control(hp, hv, ap, av, ball, tgt)
    pc = ph if e['Team'] == 'Home' else 1 - ph
    rows.append((pc, math.hypot(tgt[0]-e['Start X']*L, tgt[1]-e['Start Y']*W), ok))
R = np.array(rows); y = R[:, 2].astype(int)
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
rng = np.random.default_rng(0)
def ci(s):
    b = [auc(s[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]; return np.percentile(b, [2.5, 97.5])
print(f'passes evaluated: {len(y)}, success rate {y.mean():.3f}')
for nm, s in [('pitch control at target', R[:,0]), ('baseline: -pass length', -R[:,1])]:
    lo, hi = ci(s); print(f'  AUC {nm:26s} {auc(s,y):.3f} (95% CI {lo:.3f}-{hi:.3f})')
# does control add beyond length? logistic fit with both, in-sample vs 5-fold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
for nm, X in [('length only', R[:,[1]]), ('length + control', R[:,[0,1]])]:
    p = cross_val_predict(LogisticRegression(), X, y, cv=5, method='predict_proba')[:,1]
    print(f'  5-fold CV AUC {nm:18s} {auc(p,y):.3f}')
