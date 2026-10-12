"""AMI Phase 2: run the validated v2 engine on all 64 WC2018 matches.
Components are computed per (match, team, possession); normalisation is GLOBAL across
the tournament so scores are comparable between matches (per-match min-max would not be)."""
import json, numpy as np, collections
from engine import PitchConfig, MetricFootballEngine, Event
from load import load_matches, load_events, ts

SBL, SBW = 120.0, 80.0
from ami_multi_ceilings import MAX_VT
cfg = PitchConfig(); eng = MetricFootballEngine(cfg)
sx, sy = cfg.length/SBL, cfg.width/SBW

def components(mid):
    out = {}
    groups = collections.defaultdict(list)
    for i, e in enumerate(load_events(mid)):
        t = e['type']['name']
        if t not in ('Pass','Carry') or e['period'] > 4: continue
        loc = e.get('location'); end = (e.get('pass') or {}).get('end_location') if t=='Pass' else (e.get('carry') or {}).get('end_location')
        if not loc or not end: continue
        if e['team']['name'] != e['possession_team']['name']: continue  # only the possessing team's own actions
        dur = e.get('duration') or 0.5; t0 = ts(e['timestamp'])
        groups[(e['possession'], e['possession_team']['name'])].append(Event(i, t0, t0+dur, '?', e['team']['name'],
            (loc[0]*sx, loc[1]*sy), (end[0]*sx, end[1]*sy), t.lower()))
    for key, evs in groups.items():
        if len(evs) < 2: continue
        v = [eng.calculate_territorial_velocity(e) for e in evs]
        v = [x for x, e in zip(v, evs) if x <= MAX_VT[e.action_type]]
        if not v: continue
        cells = [eng.mapper.map_to_cell(e.end_pos) for e in evs]
        out[key] = (len(set(cells))/cfg.num_cells, float(np.mean(v)), eng.compute_compactness(cells))
    return out

ms = load_matches(); rows = []
for mid, m in ms.items():
    for (pid, team), (t, v, c) in components(mid).items():
        rows.append((mid, team, pid, t, v, c))
print('possession sequences (>=2 actions):', len(rows))
A = np.array([[r[3], r[4], r[5]] for r in rows])
N = np.column_stack([eng.minmax_normalize(A[:, j].tolist()) for j in range(3)])
ami = N @ np.array([eng.w1, eng.w2, eng.w3])
json.dump([dict(match=r[0], team=r[1], poss=r[2], t=r[3], v=r[4], c=r[5], ami=float(a)) for r, a in zip(rows, ami)], open('ami_rows.json','w'))

def summary(mid):
    sel = [i for i, r in enumerate(rows) if r[0] == mid]
    by = collections.defaultdict(list)
    for i in sel: by[rows[i][1]].append(i)
    for team, idx in by.items():
        print(f"  {team:10s} n={len(idx):3d}  mean Vt={A[idx,1].mean():5.2f} m/s  mean T_occ={A[idx,0].mean():.4f}  mean compact={A[idx,2].mean():.3f}  mean AMI={ami[idx].mean():.3f}")
for mid, name in [(8658, 'France-Croatia (Final)'), (7563, 'Denmark-France (0-0)')]:
    print(name); summary(mid)
print('\nTournament-wide: mean AMI %.3f, mean Vt %.2f m/s' % (ami.mean(), A[:,1].mean()))
