#%%
import os
from sympy import sympify
from core import EGraph
from visual import EGraphVisualizer

expr = sympify("2*(3*x)*(x**2 + 3*x + 2)+x**2")

eg = EGraph()
eg.build(expr)

#%%
visualizer = EGraphVisualizer(eg)
egraph_viz = visualizer.visualize()
egraph_viz.render('egraph', format='pdf')
# it produces an auxiliary file egraph which
# can be removed after pdf is generated
os.system('rm egraph; open egraph.pdf')

# %%
