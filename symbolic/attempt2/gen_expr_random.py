"""In this script, I will try to write a function that generates random expressions
from a given set of operaters"""

#%%
import random
from dataclasses import dataclass, field
from gen_expr_custom import Symbol, Op, EGraph, ENode

@dataclass
class ExprGen:
    symbols: list[Symbol] = field(default_factory=list)
    operators: list[Op] = field(default_factory=list)
    graph: EGraph | None = None

    def __post_init__(self):
        graph = EGraph()
        for s in self.symbols:
            graph.digest(s)
        self.graph = graph

    def gen_expr(self, niter: int, seed: int | None = None):
        if seed: random.seed(seed)
        for _ in range(niter):
            op = random.choice(self.operators)
            eclass_ids = list(self.graph.cid_to_class.keys())
            left = random.choice(eclass_ids)
            right = random.choice(eclass_ids)
            left_expr = list(self.graph.cid_to_class[left].nodes)[0]
            right_expr = list(self.graph.cid_to_class[right].nodes)[0]
            node = ENode(val=op, children=[left, right], expr=Op(op, left_expr, right_expr))
            cid = self.graph.add_node(node)
        return self.graph.cid_to_class[cid]

if __name__ == "__main__":
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    gen = ExprGen(symbols=[x, y, z], operators=["+", "-", "*", "/"])
    expr = gen.gen_expr(10, 42)
    print(list(expr.nodes)[0].expr)

    from gen_expr_custom import EGraphVisualizer
    # Visualize the EGraph
    visualizer = EGraphVisualizer(gen.graph)
    egraph_viz = visualizer.visualize()
    egraph_viz.render('egraph', format='pdf')
    import os
    os.system('rm egraph; open egraph.pdf')
            
            

# %%
