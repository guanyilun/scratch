import unittest
from gen_expr_sympy import EGraph, ENode, EClass
from sympy import Symbol, Add, Mul

class TestEGraph(unittest.TestCase):
    def test_basic_construction(self):
        """Test creating an empty EGraph"""
        egraph = EGraph(node_to_cid={}, cid_to_class={})
        self.assertEqual(len(egraph.node_to_cid), 0)
        self.assertEqual(len(egraph.cid_to_class), 0)

    def test_digest_symbol(self):
        """Test digesting a simple symbol"""
        egraph = EGraph(node_to_cid={}, cid_to_class={})
        x = Symbol('x')
        cid = egraph.digest(x)
        
        # Verify the node was created correctly
        self.assertEqual(len(egraph.node_to_cid), 1)
        self.assertEqual(len(egraph.cid_to_class), 1)
        node = egraph.cid_to_class[cid].nodes[0]
        self.assertEqual(node.val, x)
        self.assertEqual(node.children, [])

    def test_digest_expression(self):
        """Test digesting a simple expression"""
        egraph = EGraph(node_to_cid={}, cid_to_class={})
        x = Symbol('x')
        y = Symbol('y')
        expr = Add(x, y)
        cid = egraph.digest(expr)
        
        # Verify the expression was processed correctly
        self.assertEqual(len(egraph.node_to_cid), 3)  # x, y, and Add
        self.assertEqual(len(egraph.cid_to_class), 3)
        
        # Verify the Add node has correct children
        add_node = egraph.cid_to_class[cid].nodes[0]
        self.assertEqual(len(add_node.children), 2)
        self.assertEqual(add_node.val, Add)

    def test_merge_classes(self):
        """Test merging two classes"""
        egraph = EGraph(node_to_cid={}, cid_to_class={})
        x = Symbol('x')
        y = Symbol('y')
        
        x_cid = egraph.digest(x)
        y_cid = egraph.digest(y)
        
        # Merge x and y classes
        egraph.merge_class(x_cid, y_cid)
        
        # Verify merge
        self.assertEqual(len(egraph.cid_to_class), 1)
        self.assertEqual(len(egraph.cid_to_class[x_cid].nodes), 2)

    def test_apply_rule(self):
        """Test applying a rewrite rule"""
        egraph = EGraph(node_to_cid={}, cid_to_class={})
        x = Symbol('x')
        y = Symbol('y')
        
        # Create initial expression x + y
        expr1 = Add(x, y, evaluate=False)
        cid1 = egraph.digest(expr1)
        
        # Apply rule x + y -> y + x
        expr2 = Add(y, x, evaluate=False)
        egraph.apply_rule(expr1, expr2)
        
        # Verify the classes were merged
        self.assertEqual(len(egraph.cid_to_class), 3)  # x, y, and Add
        self.assertEqual(len(egraph.cid_to_class[cid1].nodes), 2)

if __name__ == '__main__':
    unittest.main()
