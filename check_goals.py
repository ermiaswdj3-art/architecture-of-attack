from load import *
ms = load_matches()
shot_goals = own_goals = 0; mism = []
for mid, m in ms.items():
    ev = load_events(mid)
    sg = sum(1 for e in ev if e['type']['name']=='Shot' and e.get('period',0)<=4 and e['shot']['outcome']['name']=='Goal')
    og = sum(1 for e in ev if e['type']['name']=='Own Goal Against')
    shot_goals += sg; own_goals += og
    # official final score after extra time, excluding shootouts, equals goals scored
    official = m['home_score'] + m['away_score']
    if sg + og != official: mism.append((mid, m['home_team']['home_team_name'], m['away_team']['away_team_name'], sg, og, official))
print('shot goals', shot_goals, '| own goals', own_goals, '| total', shot_goals+own_goals)
print('official goals (sum of match scores):', sum(m['home_score']+m['away_score'] for m in ms.values()))
print('matches where counts disagree:', mism)
