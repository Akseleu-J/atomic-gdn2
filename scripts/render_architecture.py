"""Editable vector computation diagram, matching the released hybrid forward path."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,Rectangle

OUT=ROOT/'paper/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.1,8.7))
ax.set_xlim(0,16);ax.set_ylim(0,14);ax.axis('off')
palette={'gdn':'#d9e7f5','mla':'#f4d6d7','ffn':'#f7e9bc','dar':'#d8ead9','neutral':'#eeeeee'}
def box(x,y,w,h,text,color='neutral',fs=8.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.04,rounding_size=0.09',
        facecolor=palette[color],edgecolor='#404040',lw=.8,zorder=3))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,zorder=4)
def arrow(x1,y1,x2,y2,style='-',color='#333333'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=.8,color=color,linestyle=style),zorder=2)
def plus(x,y):
    ax.add_patch(Circle((x,y),.14,facecolor='white',edgecolor='#444444',lw=.8,zorder=4))
    ax.text(x,y,'+',ha='center',va='center',fontsize=10,zorder=5)
ax.text(2.5,13.5,'(a) hybrid model',ha='center',fontweight='bold',fontsize=10)
ax.text(10.5,13.5,'(b) Expanded hybrid block',ha='center',fontweight='bold',fontsize=10)
box(2.5,.7,3.5,.55,'Input tokens')
box(2.5,1.7,3.5,.65,'Token embedding')
arrow(2.5,.98,2.5,1.34)
ax.add_patch(FancyBboxPatch((.45,3.0),4.1,7.65,boxstyle='round,pad=0.05,rounding_size=0.15',
    facecolor='#fafafa',edgecolor='#666666',lw=1))
ax.text(.05,6.8,'7 ×',ha='right',va='center',fontsize=9)
box(2.5,4.0,3.35,.9,'GDN-2\nwith DAR',color='gdn')
box(2.5,5.55,3.35,.9,'GDN-2\nwith DAR',color='gdn')
box(2.5,7.10,3.35,.9,'GDN-2\nwith DAR',color='gdn')
box(2.5,8.65,3.35,.9,'MLA\nwith DAR',color='mla')
box(2.5,10.05,3.35,.8,'SwiGLU FFN\nwith DAR',color='ffn')
for a,b in [(2.05,3.54),(4.47,5.08),(6.02,6.63),(7.57,8.18),(9.12,9.61)]:arrow(2.5,a,2.5,b)
box(2.5,11.45,3.5,.6,'Final LayerNorm')
arrow(2.5,10.48,2.5,11.1)
box(2.5,12.45,3.5,.6,'Tied projection → logits')
arrow(2.5,11.79,2.5,12.12)
ax.text(2.5,13.0,'21 GDN-2 + 7 MLA + 7 FFN',ha='center',fontsize=7.7)

x=8.8
box(8.8,.7,2.5,.5,'Block input $x_{b,0}$')
box(14.35,6.9,2.15,2.2,'Depth sources\n\nPrevious blocks\n+ earlier deltas',color='dar',fs=7.8)
ax.plot([12.95,12.95],[1.45,11],color='#628565',lw=.8)
arrow(13.25,6.9,12.95,6.9,color='#628565')
positions=[2.0,4.0,6.0,8.0,10.0]
for i,y in enumerate(positions):
    norm='LayerNorm' if i==4 else 'RMSNorm'
    module='SwiGLU FFN' if i==4 else ('MLA' if i==3 else 'GDN-2')
    color='ffn' if i==4 else ('mla' if i==3 else 'gdn')
    in_y=y-.8; route_y=y-.48; out_y=y+.83
    if i==0:arrow(x,.98,x,in_y)
    else:arrow(x,positions[i-1]+.98,x,in_y)
    arrow(x,in_y,x,route_y-.14)
    plus(x,route_y)
    box(x,y+.13,3.4,.72,norm+'\n'+module,color=color,fs=8)
    arrow(x,route_y+.14,x,y-.25)
    arrow(x,y+.5,x,out_y-.14)
    plus(x,out_y)
    ax.text(x+.27,y+.61,f'$\\delta_{{b,{i+1}}}$',fontsize=8,va='center')
    # Ordinary residual skips both routing and sublayer, preserving pre-norm ordering.
    ax.plot([x,6.65,6.65],[in_y,in_y,out_y],color='#555555',lw=.8,zorder=1)
    arrow(6.65,out_y,x-.15,out_y)
    box(11.55,route_y,1.05,.48,'DAR',color='dar',fs=7.5)
    arrow(12.95,route_y,12.12,route_y,color='#628565')
    arrow(10.98,route_y,x+.16,route_y,color='#628565')
    ax.text(10.05,route_y+.12,f'$r_{{b,{i+1}}}$',ha='center',fontsize=6.8,color='#42674c')
box(x,12.0,3.5,.6,'Block output $x_{b,5}$')
arrow(x,10.98,x,11.66)
ax.text(11.1,12.8,'Append summary: $s_b=x_{b,5}-x_{b,0}$',ha='center',fontsize=8.5)
ax.text(14.2,11.8,'Only previously\ncomputed sources',ha='center',fontsize=7)
ax.text(8,.07,'DAR: normalized keys, learned query, original delta values. Routing is along depth at each token position.',ha='center',fontsize=7)
fig.subplots_adjust(left=.04,right=.99,top=.98,bottom=.03)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
print(OUT/'architecture.pdf')
