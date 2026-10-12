"""Print-safe figures (greyscale-readable; shape and fill carry identity, not colour). 300 dpi PNGs into ../book_build/figures."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
import pc_oos as P
L,W=105.0,68.0; INK='#222222'; MUTED='#666666'
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.edgecolor':INK,'text.color':INK})
def pitch(ax):
    ax.add_patch(Rectangle((0,0),L,W,fill=False,ec=INK,lw=0.8)); ax.plot([L/2]*2,[0,W],color=INK,lw=0.6)
    ax.add_patch(Circle((L/2,W/2),9.15,fill=False,ec=INK,lw=0.6))
    for x0,s in ((0,1),(L,-1)):
        ax.add_patch(Rectangle((x0 if s==1 else L-16.5,W/2-20.15),16.5,40.3,fill=False,ec=INK,lw=0.6))
        ax.add_patch(Rectangle((x0 if s==1 else L-5.5,W/2-9.16),5.5,18.32,fill=False,ec=INK,lw=0.6))
    ax.set_xlim(-1,L+1); ax.set_ylim(W+1,-1); ax.set_aspect('equal'); ax.axis('off')
# ---- Figure A: pitch-control map in open play (Game 2), flipped so the attack runs left to right ------
from matplotlib.colors import LinearSegmentedColormap
CM=LinearSegmentedColormap.from_list('g',['#ffffff','#8c8c8c'])
data=P.build(2); pick=None
for d in data[200:]:
    if not d['home']: continue
    attack_left = (d['period']==1)          # Game 2: Home attacks left in period 1 (kickoff centroid x=65.8 m), right in period 2
    sgn=-1 if attack_left else 1
    prog=sgn*(d['tgt'][0]-d['ball'][0])
    bx=d['ball'][0] if not attack_left else L-d['ball'][0]
    if d['ok'] and prog>14 and 55<bx<80 and len(d['hp'])==11 and len(d['ap'])==11: pick=(d,attack_left); break
d,flip=pick
def fx(a): a=np.array(a,float).copy(); (a.__setitem__((...,0),L-a[...,0]) if flip else None); return a
hp,ap,ball,tgt=fx(d['hp']),fx(d['ap']),fx(d['ball']),fx(d['tgt'])
xs=np.arange(1.5,L,3.0); ys=np.arange(1.5,W,3.0); M=np.zeros((len(ys),len(xs)))
for a,x in enumerate(xs):
    for b,y in enumerate(ys):
        dd=dict(d); dd['tgt']=fx([x,y]); M[b,a]=P.control(dd,5.0,0.45,4.3)
fig,ax=plt.subplots(figsize=(6.0,4.1),dpi=300)
im=ax.imshow(M,extent=[0,L,W,0],cmap=CM,vmin=0,vmax=1,aspect='equal',interpolation='bilinear'); pitch(ax)
ax.scatter(hp[:,0],hp[:,1],s=34,c='white',ec=INK,lw=1.1,zorder=3,label="Attacking team (passer's side)")
ax.scatter(ap[:,0],ap[:,1],s=34,c=INK,ec='white',lw=0.6,zorder=3,label='Defending team')
ax.scatter([ball[0]],[ball[1]],marker='*',s=80,c='black',ec='white',lw=0.6,zorder=4)
ax.annotate('',xy=tgt,xytext=ball,arrowprops=dict(arrowstyle='->',color=INK,lw=1.2,ls='--'),zorder=4)
ax.annotate('Attack direction',xy=(L-6,W-3),xytext=(L-30,W-3),arrowprops=dict(arrowstyle='->',color=INK,lw=0.8),fontsize=7,va='center',zorder=5)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.01); cb.set_label('Probability the attacking team controls the ball here',fontsize=7); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=7)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.01),ncol=2,frameon=False,fontsize=7,handletextpad=0.3)
fig.savefig('../book_build/figures/fig_pitch_control_map.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('flipped',flip,'ball',ball.round(1),'target',tgt.round(1),'ok',d['ok'])
# ---- Figure B: out-of-sample AUC dot plot ---------------------------------------------------
rows=[('Game 2: pitch control (tuned on Game 1)',0.658,0.607,0.705),('Game 2: pitch control (default constants)',0.657,None,None),('Game 2: nearest-opponent distance',0.592,None,None),('Game 2: pass length',0.512,None,None),('Game 1: pitch control',0.594,0.543,0.646),('Game 1: nearest-opponent distance',0.555,None,None),('Game 1: pass length',0.485,0.437,0.536)]
fig,ax=plt.subplots(figsize=(6.0,2.8),dpi=300)
for i,(lab,v,lo,hi) in enumerate(rows):
    y=len(rows)-1-i; ax.plot([0.5,v],[y,y],color='#bbbbbb',lw=0.8,zorder=1)
    if lo: ax.plot([lo,hi],[y,y],color=INK,lw=2.0,zorder=2)
    ax.scatter([v],[y],s=26,c=INK if 'pitch' in lab else 'white',ec=INK,lw=1.0,zorder=3); ax.text(0.725,y,f'{v:.3f}',va='center',fontsize=7,color=MUTED)
ax.axvline(0.5,color=INK,lw=0.6,ls=':'); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=7.5)
ax.set_xlim(0.42,0.78); ax.set_xlabel('AUC for “the pass succeeds” (0.5 = chance)',fontsize=8)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.savefig('../book_build/figures/fig_auc_out_of_sample.png',dpi=300,bbox_inches='tight'); plt.close(fig)
