"""Weight sensitivity of AMI on all 9,155 corrected possessions (needs ami_rows.json from ami_multi.py)."""
import json, numpy as np
from scipy.stats import spearmanr
rows=json.load(open('ami_rows.json')); T=np.array([r['t'] for r in rows]);V=np.array([r['v'] for r in rows]);Cc=np.array([r['c'] for r in rows])
mm=lambda a:(a-a.min())/(a.max()-a.min()); N=[mm(T),mm(V),mm(Cc)]; B=(0.4,0.4,0.2)
base=sum(w*n for w,n in zip(B,N)); k=200; top=set(np.argsort(-base)[:k])
print('n possessions',len(rows),'; top-k =',k)
for i,name in enumerate(['T_occ','V_t','Compactness']):
    o=[j for j in range(3) if j!=i]; ratio=B[o[0]]/(B[o[0]]+B[o[1]])
    for w in (0.05,0.2,0.4,0.6,0.85):
        ws=[0,0,0]; ws[i]=w; ws[o[0]]=(1-w)*ratio; ws[o[1]]=(1-w)*(1-ratio)
        s=sum(a*n for a,n in zip(ws,N)); print(f'{name} w={w:.2f}: rho {spearmanr(base,s)[0]:.2f}, top-{k} overlap {len(top&set(np.argsort(-s)[:k]))}/{k}')
for (a,na),(b,nb) in [((0,'T_occ'),(1,'V_t')),((0,'T_occ'),(2,'Compactness')),((1,'V_t'),(2,'Compactness'))]: print(f'{na} vs {nb}: {spearmanr(N[a],N[b])[0]:.2f}')
print('Spearman(AMI, component):',[round(spearmanr(base,n)[0],2) for n in N])
