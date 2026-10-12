import math
from load import *
from test_30s import wilson
ms = load_matches(); rows = []
for mid in ms: rows += possessions(load_events(mid))
rows = [r for r in rows if r['n'] >= 2]
op = [r for r in rows if r['pattern'] == 'Regular Play']
print('open-play possessions only:', len(op), 'of', len(rows))
print(f"{'bucket':>9} {'n':>6} {'shot%':>7} {'goal%':>7} {'xG/poss':>8}")
for lo, hi in [(0,10),(10,20),(20,30),(30,45),(45,9999)]:
    b = [r for r in op if lo <= r['dur'] < hi]; n = len(b)
    print(f"{(str(lo)+'-'+str(hi)+'s') if hi<9999 else str(lo)+'s+':>9} {n:6d} {sum(r['shots']>0 for r in b)/n*100:6.1f}% {sum(r['goals']>0 for r in b)/n*100:6.1f}% {sum(r['xg'] for r in b)/n:8.3f}")
# Share-of-goals vs share-of-possessions for >30s
L = [r for r in op if r['dur'] > 30]
print(f"\n>30s open-play: {len(L)/len(op)*100:.1f}% of possessions, {sum(r['goals'] for r in L)/sum(r['goals'] for r in op)*100:.1f}% of goals, {sum(r['xg'] for r in L)/sum(r['xg'] for r in op)*100:.1f}% of xG")
