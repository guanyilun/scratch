#%%
from dataclasses import dataclass
from sympy import srepr

@dataclass
class ENode:
    val: "Expr"
    children: list["EClass"] | None
    expr: "Expr"

    def __hash__(self):
        return hash(self.expr)

    def __repr__(self):
        return self.expr.__repr__()

@dataclass
class EClass:
    nodes: list[ENode]
    
class EGraph:
    def __init__(self):
        self.node_to_cid = {} 
        self.cid_to_class = {}

    def build(self, expr):
        # if expr is a symbol or number, it is a leaf node
        if expr.is_symbol or expr.is_number:
            node = ENode(val=expr, children=[], expr=expr)
            cid = self.add_node(node)
        else:
            node = ENode(val=expr.func, children=[], expr=expr)
            cid = self.add_node(node)
            children_cids = [self.build(arg) for arg in expr.args]
            node.children = children_cids
        return cid
        
    def add_node(self, node):
        # no need to add new node if it already exists
        k = hash(node)
        if k in self.node_to_cid:
            return self.node_to_cid[k]
        else:
            mapping = self.cid_to_class
            cid = max(mapping.keys())+1 if len(mapping) > 0 else 0
            self.node_to_cid[k] = cid
            self.cid_to_class[cid] = EClass(nodes=[node])
        return cid
