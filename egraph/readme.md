# egraph in python
Time for some fun with sympy: let's build an egraph.

## 240113
Had some good results from a good one-hour hack.

Given a sympy expression like
```python
expr = sympify("2*(3*x)*(x**2 + 3*x + 2)+x**2")
```
we can build an EGraph for this expression using
```python
eg = EGraph()
eg.build(expr)
```
To visualize how it looks:
```python
visualizer = EGraphVisualizer(eg)
egraph_viz = visualizer.visualize()
egraph_viz.render('egraph', format='pdf')
```
It will produce a graph that looks like

<img width="457" alt="image" src="https://github.com/guanyilun/scratch/assets/1038228/a1d22688-f8d9-4ec3-a2d4-0924a16f9b3e">

Next steps
- add rewrite rules
- equation saturation

