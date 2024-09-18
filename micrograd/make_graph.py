from graphviz import Digraph
import uuid

def get_nodes_in_graph(result_node):
    nodes = []
    to_visit = [result_node]
    explored = set()
    while to_visit:
        node = to_visit.pop(0)
        if id(node) not in explored:
            nodes.append(node)
            explored.add(id(node))
            if node._children:
                for child in node._children:
                    to_visit.append(child)
    return nodes        

def make_graph(nodes_list):
    edges = []
    node_ids = {}
    dot = Digraph(graph_attr={'rankdir': 'LR'})
    for n in nodes_list:
        # create a unique identifier for each node
        node_id = str(uuid.uuid4())
        node_ids[id(n)] = node_id
    for n in nodes_list:
        node_id = node_ids.get(id(n))
        dot.node(node_id, "{" + n.label + " | " + "data: " + str(round(n.data, 3)) + "|" + "grad: " + str(round(n.grad, 3)) + "}", shape='record')
        if n._op:
            op_id = node_id + n._op
            dot.node(op_id, n._op)
            dot.edge(op_id, node_id)
        if n._children:
            for child in n._children:
                child_id = node_ids.get(id(child))
                if child_id:
                    if n._op:
                        dot.edge(child_id, op_id)
                    else:
                        dot.edge(child_id, node_id)
    return dot 