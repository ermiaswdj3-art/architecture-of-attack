"""Grid comparison: (A) engine mapper as coded, (B) true hexagons side 3 m (23.38 m2),
(C) squares of the same 23.38 m2. Same possessions, same outcome (possession contains a shot)."""
import numpy as np, collections
from load import load_matches, load_events, possessions, ts
from ami_multi_ceilings import MAX_VT
SBL, SBW = 120.0, 80.0; sx, sy = 105/SBL, 68/SBW
S = 3.0; AREA = 1.5*np.sqrt(3)*S*S; SQ = np.sqrt(AREA)

def cell_A(x, y):   # engine's mapper, verbatim
    return (int(x/(1.5*S)), int(y/(np.sqrt(3)/2*S)))
def cell_B(x, y):   # true pointy->flat-top hex (side S) via cube rounding
    q = (2/3*x)/S; r = (-1/3*x + np.sqrt(3)/3*y)/S
    X, Z = q, r; Y = -X-Z
    rx, ry, rz = round(X), round(Y), round(Z)
    dx, dy, dz = abs(rx-X), abs(ry-Y), abs(rz-Z)
    if dx > dy and dx > dz: rx = -ry-rz
    elif dy > dz: ry = -rx-rz
    else: rz = -rx-ry
    return (rx, rz)
def cell_C(x, y): return (int(x//SQ), int(y//SQ))
def centre(grid, c):  # centre in metres, then in cell-length units for compactness
    if grid == 'A': return (c[0], c[1])  # engine uses raw index coordinates
    if grid == 'B': return (S*1.5*c[0], S*np.sqrt(3)*(c[1] + c[0]/2))
    return (c[0]*SQ, c[1]*SQ)
NOMINAL = {'A': 105*68/(1.5*S*np.sqrt(3)/2*S), 'B': 105*68/AREA, 'C': 105*68/AREA}
FN = {'A': cell_A, 'B': cell_B, 'C': cell_C}
UNIT = {'A': 1.0, 'B': S*np.sqrt(3), 'C': SQ}   # length unit for compactness (metres)

def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    order = score.argsort(kind='mergesort'); ranks = np.empty(len(score)); s = score[order]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        ranks[order[i:j+1]] = (i+j)/2+1; i = j+1
    n1 = y.sum(); n0 = len(y)-n1
    return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

rows = []  # (y, vt, {grid: (t, c)}, nactions)
for mid in load_matches():
    ev = load_events(mid); sh = {}
    for p in possessions(ev): sh[(p['pid'], p['team'])] = p['shots']
    g = collections.defaultdict(list)
    for e in ev:
        t = e['type']['name']
        if t not in ('Pass','Carry') or e['period'] > 4: continue
        loc = e.get('location'); end = (e.get('pass') or {}).get('end_location') if t=='Pass' else (e.get('carry') or {}).get('end_location')
        if not loc or not end or e['team']['name'] != e['possession_team']['name']: continue
        dur = max(e.get('duration') or 0.5, 0.1)
        g[(e['possession'], e['possession_team']['name'])].append((loc[0]*sx, loc[1]*sy, end[0]*sx, end[1]*sy, dur, t.lower()))
    for k, evs in g.items():
        if len(evs) < 2 or k not in sh: continue
        v = [max(a[2]-a[0], 0)/a[4] for a in evs]
        v = [x for x, a in zip(v, evs) if x <= MAX_VT[a[5]]]
        if not v: continue
        d = {}
        for gr in 'ABC':
            cells = [FN[gr](a[2], a[3]) for a in evs]
            pts = np.array([centre(gr, c) for c in set(cells)] if False else [centre(gr, c) for c in cells])/UNIT[gr]
            d[gr] = (len(set(cells))/NOMINAL[gr], 1/(1+pts[:,0].var()+pts[:,1].var()))
        rows.append((1 if sh[k] > 0 else 0, float(np.mean(v)), d, len(evs)))
y = np.array([r[0] for r in rows]); vt = np.array([r[1] for r in rows])
print('possessions', len(y), 'with shot', y.sum())
def mm(a): a = np.asarray(a); return (a-a.min())/(a.max()-a.min())
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
res = {}
for gr in 'ABC':
    t = np.array([r[2][gr][0] for r in rows]); c = np.array([r[2][gr][1] for r in rows])
    ami = 0.4*mm(t)+0.4*mm(vt)+0.2*mm(c); res[gr] = ami
    print(f'grid {gr}: AMI AUC {auc(ami,y):.3f} | T_occ {auc(t,y):.3f} | compact {auc(c,y):.3f} | mean T_occ {t.mean():.3f}')
for a_, b_ in [('B','C'), ('B','A'), ('A','C')]:
    d = [auc(res[a_][i], y[i]) - auc(res[b_][i], y[i]) for i in B]
    print(f'AMI AUC diff {a_}-{b_}: {auc(res[a_],y)-auc(res[b_],y):+.4f}  95% CI {np.percentile(d,2.5):+.4f} to {np.percentile(d,97.5):+.4f}')
print('baseline n actions', f'{auc([r[3] for r in rows], y):.3f}', '| V_t only', f'{auc(vt,y):.3f}')
