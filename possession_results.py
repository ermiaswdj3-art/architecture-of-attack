"""Steelman test: in WC2018, did the team with more of the ball win more? Possession share = share of total possession time (open play+set pieces, periods 1-4)."""
import numpy as np, collections
from load import load_matches, load_events, possessions
from scipy.stats import spearmanr
rows=[]
for mid,m in load_matches().items():
    h,a=m['home_team']['home_team_name'],m['away_team']['away_team_name']; t=collections.Counter()
    for p in possessions(load_events(mid)): t[p['team']]+=p['dur']
    tot=t[h]+t[a]
    if tot==0: continue
    sh=t[h]/tot; gd=m['home_score']-m['away_score']
    rows.append((mid,h,a,sh,gd,m['home_score'],m['away_score']))
sh=np.array([r[3] for r in rows]); gd=np.array([r[4] for r in rows])
print('matches',len(rows))
# decisive matches only: did the team with more possession win?
dec=[r for r in rows if r[4]!=0]; more_wins=sum(1 for r in dec if (r[3]>0.5)==(r[4]>0)); 
print(f'decisive (after ET): {len(dec)}; team with MORE possession won {more_wins} ({more_wins/len(dec)*100:.1f}%)')
rho,p=spearmanr(sh,gd); rng=np.random.default_rng(0)
bs=[spearmanr(sh[i],gd[i])[0] for i in (rng.integers(0,len(sh),len(sh)) for _ in range(1000))]
print(f'Spearman(possession share, goal diff) = {rho:.3f} (95% CI {np.percentile(bs,2.5):.3f} to {np.percentile(bs,97.5):.3f}), p={p:.3f}')
# Wilson CI for more-possession win rate
n=len(dec);k=more_wins;z=1.96;ph=k/n;den=1+z*z/n;c=(ph+z*z/(2*n))/den;hw=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
print(f'win-rate Wilson 95% CI: {c-hw:.3f}-{c+hw:.3f}')
for r in rows:
    if r[0] in (8658,7563): print(r[1],'v',r[2],'home possession share',round(r[3],3),'score',r[5],r[6])
# top-possession teams: results
srt=sorted(rows,key=lambda r:-abs(r[3]-0.5))[:10]
print('10 most one-sided possession matches (share, result for the possession-heavy side):')
for r in srt:
    heavy=r[3]>0.5; gdh=r[4] if heavy else -r[4]; print(f'  {r[1] if heavy else r[2]} {max(r[3],1-r[3]):.2f} -> goal diff {gdh:+d}')
