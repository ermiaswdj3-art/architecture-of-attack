"""Do the index parts add information beyond WHERE a possession is and HOW LONG it is? 5-fold CV AUC for 'possession contains a shot'."""
import json, numpy as np, collections
from load import load_matches, load_events
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
rows={ (r['match'],r['poss'],r['team']):r for r in json.load(open('ami_rows.json'))}
X=[];y=[]
for mid in load_matches():
    ev=load_events(mid); g=collections.defaultdict(list); shot=collections.defaultdict(int)
    for e in ev:
        if e['period']>4: continue
        k=(mid,e['possession'],e['possession_team']['name'])
        if e['type']['name']=='Shot' and e['team']['name']==e['possession_team']['name']: shot[k]+=1
        if e['type']['name'] in ('Pass','Carry') and e.get('location') and e['team']['name']==e['possession_team']['name']:
            end=(e.get('pass') or e.get('carry') or {}).get('end_location')
            if end: g[k].append((e['location'][0],end[0]))
    for k,a in g.items():
        if k in rows and len(a)>=2:
            r=rows[k]; X.append([len(a),a[0][0],max(x[1] for x in a),r['t'],r['v'],r['c'],r['ami']]); y.append(1 if shot[k] else 0)
X=np.array(X);y=np.array(y);print('n',len(y),'shot rate',y.mean().round(3))
def auc(s,y): r=rankdata(s);n1=y.sum();n0=len(y)-n1;return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def cv(cols):
    Z=(X[:,cols]-X[:,cols].mean(0))/X[:,cols].std(0); res=[]
    for seed in range(5):
        p=cross_val_predict(LogisticRegression(max_iter=500),Z,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method='predict_proba')[:,1]; res.append(auc(p,y))
    return np.mean(res),np.std(res)
names=['n actions','start x','furthest x','T_occ','V_t','compact','AMI']
for label,cols in [('length + place (n, start x, furthest x)',[0,1,2]),('... + V_t',[0,1,2,4]),('... + T_occ',[0,1,2,3]),('... + compactness',[0,1,2,5]),('... + all three parts',[0,1,2,3,4,5]),('... + AMI',[0,1,2,6]),('AMI alone',[6])]:
    m,s=cv(cols); print(f'{label:42s} CV AUC {m:.3f} (+/-{s:.3f} over 5 reshuffles)')
# Wilson CIs for stagnation groups
def wil(k,n):
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
print('stagnant 5/63',[round(v,3) for v in wil(5,63)],' progressive',[round(v,3) for v in wil(round(.208*1164),1164)])
