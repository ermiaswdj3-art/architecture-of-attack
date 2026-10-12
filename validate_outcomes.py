"""Task 4: does AMI relate to shots/goals? Possession level (n~10.7k) and team-match level (n=128).
Baselines matter: if a trivial feature does as well, AMI adds nothing."""
import json, numpy as np, collections
from load import load_matches, load_events, possessions
rows = json.load(open('ami_rows.json'))
ms = load_matches()
outcome = {}; match_team = collections.defaultdict(lambda: dict(xg=0.0, shots=0))
for mid in ms:
    for p in possessions(load_events(mid)):
        k = (mid, p['pid'], p['team']); o = outcome.setdefault(k, dict(shots=0, xg=0.0, goals=0))
        o['shots'] += p['shots']; o['xg'] += p['xg']; o['goals'] += p['goals']
        match_team[(mid, p['team'])]['xg'] += p['xg']; match_team[(mid, p['team'])]['shots'] += p['shots']

def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    order = score.argsort(kind='mergesort'); ranks = np.empty(len(score)); 
    # average ranks for ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        ranks[order[i:j+1]] = (i + j)/2 + 1; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)

# extra baseline features from the raw possession (actions count, forward progress)
feat = {}
for mid in ms:
    ev = load_events(mid)
    g = collections.defaultdict(list)
    for e in ev:
        if e['type']['name'] in ('Pass','Carry') and e.get('location') and e['period'] <= 4 and e['team']['name']==e['possession_team']['name']:
            g[(e['possession'], e['team']['name'])].append(e['location'][0])
    for (pid, team), xs in g.items(): feat[(mid, pid, team)] = (len(xs), max(xs) - xs[0])

y, S = [], collections.defaultdict(list)
for r in rows:
    k = (r['match'], r['poss'], r['team'])
    if k not in outcome or k not in feat: continue
    y.append(1 if outcome[k]['shots'] > 0 else 0)
    S['AMI'].append(r['ami']); S['V_t only'].append(r['v']); S['T_occ only'].append(r['t']); S['compactness only'].append(r['c'])
    S['n actions (baseline)'].append(feat[k][0]); S['forward progress (baseline)'].append(feat[k][1])
y = np.array(y); print(f'possessions: {len(y)}, ending in/containing a shot: {y.sum()} ({y.mean()*100:.1f}%)')
print('AUC for "possession contains a shot" (0.5 = coin flip):')
rng = np.random.default_rng(0)
for name, sc in S.items():
    sc = np.array(sc); a = auc(sc, y)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y)); boots.append(auc(sc[idx], y[idx]))
    print(f'  {name:30s} {a:.3f}  (95% CI {np.percentile(boots,2.5):.3f}-{np.percentile(boots,97.5):.3f})')

# team-match level: mean AMI vs match xG
tm = collections.defaultdict(list)
for r in rows: tm[(r['match'], r['team'])].append(r['ami'])
keys = [k for k in tm if k in match_team]
a = np.array([np.mean(tm[k]) for k in keys]); xg = np.array([match_team[k]['xg'] for k in keys])
def spearman(a, b): return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0,1]
rho = spearman(a, xg); bs = [spearman(a[i], xg[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(500))]
print(f'\nteam-match level (n={len(keys)}): Spearman(mean AMI, match xG) = {rho:.3f}  (95% CI {np.percentile(bs,2.5):.3f} to {np.percentile(bs,97.5):.3f})')
print(f'std of team-match mean AMI = {a.std():.4f}; France 0-0 vs Final gap: see ami_multi.py (0.004, about 0.46 SD of the team-match means)')
