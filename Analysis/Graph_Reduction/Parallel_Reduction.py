import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def find_nearest_common_join(G, fork, succ):
    """
    Trova il 'join' più vicino (lowest common descendant) dei rami successivi.
    """
    common = set.intersection(*(set(nx.descendants(G, s)) for s in succ))
    if not common:
        return None

    best = None
    best_dist = float("inf")
    for j in common:
        dists = []
        ok = True
        for s in succ:
            try:
                d = nx.shortest_path_length(G, s, j)
                dists.append(d)
            except nx.NetworkXNoPath:
                ok = False
                break
        if ok:
            max_d = max(dists)
            if max_d < best_dist:
                best_dist = max_d
                best = j
    return best


def reduce_parallel_branches(G: nx.DiGraph):
    """
    Riduce solo un livello di fork-join (no ricorsione).
    """
    G = G.copy()
    changed = True

    while changed:
        changed = False
        for node in list(G.nodes()):
            print("Node", node)
            succ = list(G.successors(node))
            if len(succ) > 1:
                join = find_nearest_common_join(G, node, succ)
                if join is None:
                    continue
                print("\t Join", join)

                middle_nodes = set()
                empty_branches = 0
                for s in succ:
                    # nodi sul cammino tra s e join = discendenti di s ∩ antenati di join
                    reachable = set(nx.descendants(G, s)) & set(nx.ancestors(G, join))
                    reachable.add(s)  # includo il figlio stesso
                    if reachable == {s}:  # nessun nodo in mezzo
                        empty_branches += 1
                    else:
                        middle_nodes |= (reachable - {s, join})
                print("\t Middle Nodes", middle_nodes)
                print("\t Empty Branches", empty_branches)

                members = set(middle_nodes)
                if empty_branches > 0:
                    members.add(f"<empty_branch_x{empty_branches}>")

                supernode = f"super_{node}_{join}"
                G.add_node(supernode, type="supernode", members=members)

                G.add_edge(node, supernode)
                G.add_edge(supernode, join)

                G.remove_nodes_from(middle_nodes)

                changed = True
                break
    
    print(len(G.nodes))
    return G


def draw_sorted_dag(G, title="DAG"):
    """
    Draw a DAG in hierarchical top-down layout using matplotlib.
    Supernodes are highlighted and show their members.
    """
    plt.figure(figsize=(10, 6))
    
    # Use Graphviz 'dot' layout for hierarchical/topological drawing
    pos = graphviz_layout(G, prog="dot")
    
    # Node labels
    labels = {}
    node_colors = []
        
    
    # Draw nodes and edges
    nx.draw(G, pos, labels=labels, with_labels=False,
            node_size=100, node_color="lightblue",
            font_size=10, edgecolors="black")
    
    plt.title(title)
    plt.show()







# ========================
# ESEMPIO
# ========================
# G = nx.DiGraph()
# G.add_edges_from([
#     ("in", "A"),
#     ("A", "B1"), ("A", "B2"),
#     ("B1", "C1"), ("B1", "C2"),
#     ("B2", "C3"),
#     ("C1", "D"), ("C2", "D"), ("C3", "D"),
#     ("D", "out")
# ])

# print("Originale:", G.edges())
# R = reduce_parallel_branches(G)
# print("Ridotto:", R.edges())

# for n, data in R.nodes(data=True):
#     if data.get("type") == "supernode":
#         print(f"{n} contiene {data['members']}")
