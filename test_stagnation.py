"""Intent of the 30s rule: long AND stagnant (little net forward progress).
StatsBomb normalises every team to attack toward x=120, so forward = +x."""
from load import *
ms = load_matches(); rows = []
for mid in ms: rows += possessions(load_events(mid))
def progress(r):
    xs = [e['location'][0] for e in r['events'] if e.get('location') and e['team']['name']==r['team']]
    return (max(xs) - xs[0]) * 105/120 if len(xs) >= 2 else None   # metres, on a 105m pitch
L = []
for r in rows:
    if r['dur'] > 30 and r['pattern']=='Regular Play':
        p = progress(r)
        if p is not None: r['prog'] = p; L.append(r)
print('long (>30s) open-play possessions:', len(L))
for lo, hi, lab in [(-999,10,'<10m progress (stagnant)'),(10,30,'10-30m'),(30,999,'30m+ (progressive)')]:
    b = [r for r in L if lo <= r['prog'] < hi]; n = len(b)
    print(f"{lab:>26}: n={n:4d}  shot {sum(r['shots']>0 for r in b)/n*100:5.1f}%  goal {sum(r['goals']>0 for r in b)/n*100:4.1f}%  xG/poss {sum(r['xg'] for r in b)/n:.3f}")
