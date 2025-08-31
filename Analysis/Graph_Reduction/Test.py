import networkx as nx


import json
from networkx.readwrite import json_graph
from Parallel_Reduction import reduce_parallel_branches, draw_sorted_dag


def main() :

    network_graph = nx.DiGraph()
    network_graph.add_edge(0, 0, bandwidth = 0)
    network_graph.add_edge(0, 1, bandwidth = 100)
    network_graph.add_edge(1, 0, bandwidth = 100)
    network_graph.add_edge(1, 1, bandwidth = 0)
    # network_graph.add_edge(0, 2, bandwidth = 100)
    # network_graph.add_edge(2, 0, bandwidth = 100)
    # network_graph.add_edge(2, 1, bandwidth = 100)
    # network_graph.add_edge(1, 2, bandwidth = 100)
    # network_graph.add_edge(2, 2, bandwidth = 0)

    with open("yolo11x-seg.json") as f:
        data = json.load(f)["graph"]

    model_graph = json_graph.node_link_graph(data, directed=True)  # ensure DiGraph

    reduced_graph = reduce_parallel_branches(model_graph)
    draw_sorted_dag(reduced_graph)
    

    # GeneticSolver.solve_problem(model_graph, network_graph, exec_profile_dict)
    # CMA.solve_cma(model_graph, network_graph, exec_profile_dict)
    # BayesianOptimization.optimize(model_graph, network_graph, exec_profile_dict)

    pass


if __name__ == "__main__" :
    main()