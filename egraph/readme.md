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

I have also implemented a simple rewriting rule system using `sympy.unify.usympy.unify`,
which is a powerful but poorly documented functionality of `sympy`.

Here is how I designed the term-rewriting system. We define a rewrite rule using
a simple string
```python
from rewrite import Rule

rule = Rule.parse("?a + ?b + c -> ?b * ?a + c")
```
Each variable with a `?` in front will be considered a **slot** variable, which
matches any subexpression that match the pattern. `Rule` can be applied to
an expression to apply it. Currently it only rewrite based on the first match,
which is probably the most likely case. To apply the rule
```python
expr = sympify("a*x**2 + x + c")
rule(expr)
```
The output is
```python
a*x**3 + c
```
in which we can see that `?a` has successfully matched to `a*x**2` and `?b` has
matched to `x`.

## 240114
Added two rewriting routines: `Prewalk` and `Postwalk` which traverses through each expression tree differently. `Prewalk` traverses in a top-down left-right approach, whereas `Postwalk` traverses using bottom-up left-right approach. Subexpressions that do not match are automatically passed through.

For example:
```python
rule = Rule.parse("?a + ?b + c -> ?b * ?a + c")
expr = sympify("x + y + c")
print(rule(expr))

rule_pre = Prewalk(rule)
print(rule_pre(expr))

rule_post = Postwalk(rule)
print(rule_post(expr))
```
The outputs are
```
c + x*y
c + x*y
c + x*y
```
They all give the same results but the way to traverse through expression is different.

## TODO list
- [X] build egraph from sympy expression
- [X] add rewriters based on pattern
- [X] add post- and pre-walk routines for rule application
- [ ] support a chain of rules
- [ ] allow predicates on rewriters
- [ ] implement equation saturation
