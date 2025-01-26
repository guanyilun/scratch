#%%
from dataclasses import dataclass, field

class Expr: pass

@dataclass(frozen=True)
class Symbol(Expr):
    name: str

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class Number(Expr):
    value: int

    def __repr__(self):
        return str(self.value)

@dataclass(frozen=True)
class Op(Expr):
    op: str
    left: Expr
    right: Expr

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


@dataclass
class ENode:
    val: Expr
    children: list[int]
    expr: Expr

    def __hash__(self):
        return hash(self.expr)

    def __repr__(self):
        return self.expr.__repr__()

@dataclass
class EClass:
    nodes: set[ENode] = field(default_factory=set)
    
@dataclass
class EGraph:
    node_to_cid: dict[ENode, int] = field(default_factory=dict)
    cid_to_class: dict[int, EClass] = field(default_factory=dict)

    def digest(self, expr: Expr) -> int:
        # if expr is a symbol or number, it is a leaf node
        if isinstance(expr, (Symbol, Number)):
            node = ENode(val=expr, children=[], expr=expr)
            cid = self.add_node(node)
        else:
            node = ENode(val=expr.op, children=[], expr=expr)
            cid = self.add_node(node)
            children_cids = [self.digest(arg) for arg in [expr.left, expr.right]]
            node.children = children_cids
        return cid
        
    def add_node(self, node) -> int:
        # no need to add new node if it already exists
        k = hash(node)
        if k in self.node_to_cid:
            return self.node_to_cid[k]
        else:
            mapping = self.cid_to_class
            cid = max(mapping.keys())+1 if len(mapping) > 0 else 0
            self.node_to_cid[k] = cid
            self.cid_to_class[cid] = EClass(nodes=set([node]))
        return cid

    def merge_class(self, from_cid: int, to_cid: int) -> None:
        if from_cid == to_cid: return

        # Move all nodes from to_class into from_class
        to_class = self.cid_to_class.pop(to_cid)
        self.cid_to_class[from_cid].nodes.update(to_class.nodes)
        
        # Update the node_to_cid mappings
        for node in to_class.nodes:
            self.node_to_cid[hash(node)] = from_cid

        # Find all nodes that point to to_cid and update them to point to from_cid
        for eclass in self.cid_to_class.values():
            for node in eclass.nodes:
                if to_cid in node.children:
                    node.children = [from_cid if cid == to_cid else cid for cid in node.children]

    def apply_rule(self, from_expr: Expr, to_expr: Expr) -> None:
        """Apply a rewrite rule to the e-graph."""
        from_cid = self.digest(from_expr)
        to_cid = self.digest(to_expr)
        
        if from_cid != to_cid:
            self.merge_class(from_cid, to_cid)

from graphviz import Digraph

class EGraphVisualizer:
    def __init__(self, egraph):
        self.egraph = egraph
        self.graph = Digraph(comment='The EGraph')

    def visualize(self):
        for cid, eclass in self.egraph.cid_to_class.items():
            with self.graph.subgraph(name=f'cluster_{cid}') as c:
                c.attr(style='filled', color='lightgrey')
                c.node_attr.update(style='filled', color='white')
                c.attr(label=f'EClass {cid}')
                
                # Iterate over each node in the eclass
                for enode in eclass.nodes:
                    # Create a unique node name based on cid and enode's expression
                    node_name = f'{cid}_{str(enode.expr)}'
                    
                    # Label the node with the expression it represents
                    # c.node(node_name, label=str(enode.expr))
                    c.node(node_name, label=str(enode.val))
                    
                    # If the node has children, create edges to them
                    if enode.children:
                        for child_cid in enode.children:
                            # Get the first node in the child eclass to represent the child
                            child_node = list(self.egraph.cid_to_class[child_cid].nodes)[0]
                            child_name = f'{child_cid}_{str(child_node.expr)}'
                            
                            # Create an edge from the parent to the child
                            self.graph.edge(node_name, child_name)
        
        return self.graph


#%%
import os

x = Symbol('x')
expr = Op('+', 
    Op('*', 
        Op('*', 
            Number(2), 
            Op('*', Number(3), x)), 
        Op('+', 
            Op('+', 
                Op('^', x, Number(2)), 
                Op('*', Number(3), x)), 
            Number(2))), 
    Op('^', x, Number(2)))

# Create the EGraph and digest the expression
eg = EGraph()
eg.digest(expr)

# Visualize the EGraph
visualizer = EGraphVisualizer(eg)
egraph_viz = visualizer.visualize()
egraph_viz.render('egraph', format='pdf')

# Clean up and open the PDF
import os
os.system('rm egraph; open egraph.pdf')

# %%
