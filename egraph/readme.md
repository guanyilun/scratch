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

<img width="620" alt="image" src="https://github.com/guanyilun/scratch/assets/1038228/7b91cb47-0155-459f-9ed4-725b2357f6f6">

Next steps
- add rewrite rules
- equation saturation

