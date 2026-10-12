"""Table 12.1: AUC for 'possession contains a shot' within strata of possession length (number of pass/carry actions).
Features: forward progress, progress per second, start x, furthest x. 0.5 = chance. Open-play and set-piece possessions with >=2 actions by the possessing team."""
import json, numpy as np, collections
from load import load_matches, load_events, ts
from scipy.stats import rankdata
rows = {(r['match'], r['poss'], r['team']): r for r in json.load(open('ami_rows.json'))}
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
D = []
for mid in load_matches():
    g = collections.defaultdict(list); shot = collections.defaultdict(int)
    for e in load_events(mid):
        if e['period'] > 4: continue
        k = (mid, e['possession'], e['possession_team']['name'])
        if e['type']['name'] == 'Shot' and e['team']['name'] == e['possession_team']['name']: shot[k] += 1
        if e['type']['name'] in ('Pass', 'Carry') and e.get('location') and e['team']['name'] == e['possession_team']['name']:
            end = (e.get('pass') or e.get('carry') or {}).get('end_location')
            if end: g[k].append((e['location'][0], end[0], ts(e['timestamp']), ts(e['timestamp']) + (e.get('duration') or 0.5)))
    for k, a in g.items():
        if k in rows and len(a) >= 2:
            start = a[0][0]; far = max(x[1] for x in a); dur = max(a[-1][3] - a[0][2], 0.5)
            D.append((len(a), far - start, (far - start) / dur, start, far, 1 if shot[k] else 0))
D = np.array(D); y = D[:, 5].astype(int)
bins = [(2, 3), (4, 5), (6, 8), (9, 14), (15, 10**6)]
print('stratum        n   progress  per-sec  start-x  furthest-x')
for lo, hi in bins:
    m = (D[:, 0] >= lo) & (D[:, 0] <= hi); yy = y[m]
    print(f'{lo}-{hi if hi < 10**6 else "+":>3} {m.sum():6d}  ' + '  '.join(f'{auc(D[m, c], yy):.2f}   ' for c in (1, 2, 3, 4)))
