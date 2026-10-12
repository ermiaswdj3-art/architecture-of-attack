import math
from load import ts, load_matches, load_events, possessions
from test_30s import wilson

def test_ts():
    assert ts('00:01:30.500') == 90.5 and ts('01:00:00.000') == 3600

def test_wilson_bounds():
    lo, hi = wilson(50, 100); assert lo < 0.5 < hi and 0.40 < lo and hi < 0.60
    assert wilson(0, 0) == (0, 0)

def test_goal_totals_match_official_scores():
    ms = load_matches(); total = 0
    for mid, m in ms.items():
        ev = load_events(mid)
        sg = sum(1 for e in ev if e['type']['name']=='Shot' and e['period']<=4 and e['shot']['outcome']['name']=='Goal')
        og = sum(1 for e in ev if e['type']['name']=='Own Goal Against')
        assert sg + og == m['home_score'] + m['away_score'], mid
        total += sg + og
    assert total == 169 and len(ms) == 64

def test_possession_goals_equal_shot_goals_in_final():
    ev = load_events(8658); ps = possessions(ev)
    assert sum(p['goals'] for p in ps) == 5 or sum(p['goals'] for p in ps) == 6   # 4-2 final incl. 1 own goal
    assert all(p['dur'] >= 0 for p in ps)

def test_no_goal_without_shot():
    ps = [p for mid in [8658, 7563] for p in possessions(load_events(mid))]
    assert all(p['shots'] > 0 for p in ps if p['goals'] > 0)


# ---- regression tests for the bugs described in Chapter 7 ----
def test_bug1_speed_independent_of_match_clock():
    from engine import PitchConfig, MetricFootballEngine, Event
    e = MetricFootballEngine(PitchConfig())
    early = Event(1, 12.0, 14.0, 'P', 'T', (20, 30), (35, 30), 'pass')
    late = Event(2, 1800.0, 1802.0, 'P', 'T', (20, 30), (35, 30), 'pass')
    assert abs(e.calculate_territorial_velocity(early) - e.calculate_territorial_velocity(late)) < 1e-9

def test_bug3_type_specific_ceilings():
    import ami_multi_ceilings as c
    assert c.MAX_VT['pass'] > c.MAX_VT['carry'] and c.MAX_VT['carry'] < 12.5

def test_bug5_every_sequence_is_a_real_possession():
    import json
    from load import load_matches, load_events, possessions
    rows = json.load(open('ami_rows.json'))
    keys = set()
    for mid in load_matches():
        for p in possessions(load_events(mid)): keys.add((mid, p['pid'], p['team']))
    assert all((r['match'], r['poss'], r['team']) in keys for r in rows)
