"""Shared loading helpers for StatsBomb open-data (World Cup 2018)."""
import json, collections

def ts(s):
    h, m, x = s.split(':'); return int(h)*3600 + int(m)*60 + float(x)

def load_matches():
    return {m['match_id']: m for m in json.load(open('data/matches_43_3.json'))}

def load_events(mid):
    return json.load(open(f'data/{mid}.json'))

def possessions(events):
    """Group events by (period, possession id). Returns list of dicts."""
    groups = collections.OrderedDict()
    for e in events:
        if e.get('period', 0) > 4: continue            # exclude penalty shootout
        groups.setdefault((e['period'], e['possession']), []).append(e)
    out = []
    for (per, pid), evs in groups.items():
        team = evs[0]['possession_team']['name']
        t0 = min(ts(e['timestamp']) for e in evs)
        t1 = max(ts(e['timestamp']) + (e.get('duration') or 0) for e in evs)
        shots = [e for e in evs if e['type']['name'] == 'Shot' and e['team']['name'] == team]
        goals = [s for s in shots if s['shot']['outcome']['name'] == 'Goal']
        out.append(dict(period=per, pid=pid, team=team, dur=t1 - t0, n=len(evs),
                        shots=len(shots), goals=len(goals),
                        xg=sum(s['shot'].get('statsbomb_xg', 0) or 0 for s in shots),
                        pattern=evs[0]['play_pattern']['name'], events=evs))
    return out
