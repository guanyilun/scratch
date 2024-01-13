from graphviz import Digraph

# credit: gpt4
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
                for enode in eclass.nodes:
                    node_name = f'{cid}_{hash(enode)}'
                    c.node(node_name, label=str(enode))
                    if enode.children:
                        for child_cid in enode.children:
                            child_name = f'{child_cid}_{hash(self.egraph.cid_to_class[child_cid].nodes[0])}'
                            self.graph.edge(node_name, child_name)
        return self.graph


# some old codes that I don't want to delete yet

# import matplotlib.pyplot as plt
# import networkx as nx

# # convert EGraph to NetworkX graph
# G = nx.DiGraph()
# for cid, eclass in eg.cid_to_class.items():
#     for node in eclass.nodes:
#         G.add_node(node)
#         for child_cid in node.children:
#             child_node = eg.cid_to_class[child_cid].nodes[0]
#             G.add_edge(node, child_node)

# # visualize the graph with less compact layout
# pos = nx.spring_layout(G, k=1)  # Adjust the value of k
# nx.draw(G, pos, with_labels=True)
# plt.show()