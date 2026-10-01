"""Generate the fixed teaching figures for lecture 4 (no slide-time Python)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons

OUT = Path(__file__).resolve().parents[1] / 'website/slides/img'
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'axes.labelcolor': '#17324d', 'text.color': '#17324d'})
COLORS = ['#0072b2', '#d55e00']

def save(fig, name):
    fig.savefig(OUT / f'svm-{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def points(ax, X, y):
    for c, marker in [(0, 'o'), (1, 's')]:
        ax.scatter(*X[y == c].T, c=COLORS[c], marker=marker, s=55, edgecolors='white', linewidths=.6, zorder=4)

def boundary(ax, model, X, y, support=False):
    xx, yy = np.meshgrid(np.linspace(-1.7,2.7,220), np.linspace(-1.2,1.8,170))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=[-1e6,0,1e6], colors=COLORS, alpha=.10)
    ax.contour(xx, yy, z, levels=[0], colors=['#17324d'], linewidths=1.8)
    points(ax,X,y)
    if support:
        ax.scatter(*model.support_vectors_.T, s=130, facecolors='none', edgecolors='#17324d', linewidths=1.3, zorder=5)
    ax.set(xlim=(-1.7,2.7), ylim=(-1.2,1.8), xlabel='Feature 1', ylabel='Feature 2', xticks=[], yticks=[])
    ax.set_aspect('equal')

# A fitted linear SVM: introduce only the separating boundary.
X = np.array([[-2,-.7],[-1.7,.7],[-1.2,-.2],[-.9,1], [1,-.8],[1.3,.2],[1.8,1],[2,-.1]])
y = np.array([0]*4+[1]*4)
model = SVC(kernel='linear',C=10).fit(X,y)
fig, ax = plt.subplots(figsize=(9,3.5), layout='constrained')
xx,yy=np.meshgrid(np.linspace(-2.6,2.6,150),np.linspace(-1.4,1.6,100))
z=model.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,z,levels=[-1e6,0,1e6],colors=COLORS,alpha=.10)
ax.contour(xx,yy,z,levels=[0],colors=['#17324d'],linewidths=2)
points(ax,X,y)
ax.set(xlabel='Feature 1',ylabel='Feature 2',xticks=[],yticks=[],title='A straight boundary separates the classes')
save(fig,'linear')

x=np.array([-2.3,-1.9,-1.6,-.6,-.2,.3,.7,1.7,2,2.4]); y=(np.abs(x)<1).astype(int)
fig,ax=plt.subplots(figsize=(10,3),layout='constrained')
points(ax,np.c_[x,np.zeros_like(x)],y)
ax.axhline(0,color='#94a3b8',zorder=0)
ax.set(xlabel='Original feature x',yticks=[],ylim=(-.7,.7),xlim=(-2.7,2.7))
save(fig,'original')
fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
points(axes[0],np.c_[x,np.zeros_like(x)],y)
axes[0].axhline(0,color='#94a3b8',zorder=0)
axes[0].set(xlabel='Original feature x',yticks=[],ylim=(-1,1),title='One threshold is not enough')
points(axes[1],np.c_[x,x*x],y)
axes[1].set(xlabel='Original feature x',ylabel='New feature x²',title='Add x² as a feature')
for ax in axes: ax.set_xlim(-2.7,2.7)
save(fig,'mapping')

fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
r=np.linspace(0,4,400)
for g,c in [(.1,'#0072b2'),(1,'#009e73'),(10,'#d55e00')]:
    ax.plot(r,np.exp(-g*r*r),label=f'γ = {g}',color=c,lw=3)
ax.set(xlabel='Distance between two points',ylabel='RBF similarity',ylim=(0,1.05),xlim=(0,4))
ax.legend(); ax.grid(alpha=.15)
save(fig,'similarity')

X,y=make_moons(n_samples=65,noise=.19,random_state=12)
model=SVC(C=2,gamma=1).fit(X,y)
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
boundary(ax,model,X,y,True)
ax.set_title('Outlined training points are support vectors')
save(fig,'support')

fig,axes=plt.subplots(2,3,figsize=(11,5.4),layout='constrained')
for row,C in enumerate([.1,100]):
    for col,gamma in enumerate([.1,1,10]):
        model=SVC(C=C,gamma=gamma).fit(X,y)
        boundary(axes[row,col],model,X,y)
        axes[row,col].set(xlabel='',ylabel=f'C = {C}' if col==0 else '',title=f'γ = {gamma}' if row==0 else '')
save(fig,'controls')
