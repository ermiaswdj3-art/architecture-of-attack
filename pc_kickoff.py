"""Kickoff symmetry at the model level: with the ball at the centre spot, the two teams should control about equal areas, each mostly its own half."""
import numpy as np, pc_oos as P
L,W=105.0,68.0
def state(cols,i):
    pos=[];vel=[]
    for x,y in cols.values():
        if np.isnan(x[i]) or np.isnan(y[i]): continue
        j0,j1=max(i-1,0),min(i+1,len(x)-1); dx,dy=x[j1]-x[j0],y[j1]-y[j0]
        pos.append((x[i],y[i])); vel.append((0,0) if np.isnan(dx) or np.isnan(dy) else (dx/.08,dy/.08))
    return np.array(pos),np.array(vel)
for g,f in ((1,1),(2,51)):
    fh,H,bh=P.load(g,'Home'); fa,A,ba=P.load(g,'Away'); i=int(np.where(fh==f)[0][0]); hp,hv=state(H,i); ap,av=state(A,i)
    ball=np.array([bh[0][i],bh[1][i]]); xs=np.arange(1.5,L,3.0); ys=np.arange(1.5,W,3.0); M=np.zeros((len(ys),len(xs)))
    for a,x in enumerate(xs):
        for b,y in enumerate(ys): M[b,a]=P.control(dict(hp=hp,hv=hv,ap=ap,av=av,ball=ball,tgt=np.array([x,y]),home=True),5.0,0.45,4.3)
    home_left=hp[:,0].mean()<L/2; left=M[:,xs<L/2].mean(); right=M[:,xs>=L/2].mean()
    own_home=left if home_left else right; own_away=1-(right if home_left else left)
    print(f'Game {g}: mean P(home controls) over whole pitch {M.mean():.3f}; home controls {own_home:.3f} of its own half; away controls {own_away:.3f} of its own half; share of grid cells >0.5 for home {np.mean(M>0.5):.3f}')
