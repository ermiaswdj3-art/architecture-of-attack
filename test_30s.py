"""Test the Manifesto's Ch.7 claim on all 64 WC2018 matches.
Claim: 'Possessions exceeding 30 seconds without a shot almost never produce goals.'
Note: that sentence is true by definition (a goal needs a shot, own goals aside).
The informative questions are: how often do long possessions happen, and do they
produce shots/goals at a lower rate than shorter ones?"""
import math, collections
from load import *

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)

def main():
    ms = load_matches(); rows = []
    for mid in ms:
        rows += possessions(load_events(mid))
    rows = [r for r in rows if r['n'] >= 2]          # drop 1-event stubs
    print('team-possessions analysed:', len(rows))

    buckets = [(0,10),(10,20),(20,30),(30,45),(45,9999)]
    print(f"{'bucket':>9} {'n':>6} {'share':>6} {'shot%':>14} {'goal%':>14} {'xG/poss':>8}")
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r['dur'] < hi]
        n = len(b); s = sum(1 for r in b if r['shots']>0); g = sum(1 for r in b if r['goals']>0)
        ws, wg = wilson(s,n), wilson(g,n)
        lab = f'{lo}-{hi}s' if hi<9999 else f'{lo}s+'
        print(f"{lab:>9} {n:6d} {n/len(rows)*100:5.1f}% {s/n*100:5.1f}% [{ws[0]*100:.1f}-{ws[1]*100:.1f}] {g/n*100:4.1f}% [{wg[0]*100:.1f}-{wg[1]*100:.1f}] {sum(r['xg'] for r in b)/n:8.3f}")

    long_ = [r for r in rows if r['dur'] > 30]
    print('\npossessions >30s:', len(long_), '| with a shot:', sum(1 for r in long_ if r['shots']>0),
          '| goals from them:', sum(r['goals'] for r in long_), 'of', sum(r['goals'] for r in rows), 'total shot goals')
    nos = [r for r in long_ if r['shots']==0]
    print('>30s WITHOUT a shot:', len(nos), '| goals from those (excluding own goals):', sum(r['goals'] for r in nos))


if __name__ == "__main__":
    main()
