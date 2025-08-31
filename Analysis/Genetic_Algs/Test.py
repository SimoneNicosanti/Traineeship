import networkx as nx
import numpy as np
import onnx


from ModelProfiler import OnnxModelProfiler
import GeneticSolver
import json
from networkx.readwrite import json_graph
import Utils
import difflib

MODEL_PATH = "../onnx_model/yolo11x-seg/yolo11x-seg.onnx"


def test(model_graph : nx.DiGraph) :

    Utils.kahn_topo_sort_with_dfs(model_graph)
    # generator = np.random.default_rng(seed = 42)

    # topo_sort = list(nx.topological_sort(model_graph))
    # layers_assignments = {}
    # for node in topo_sort :
    #     layers_assignments[node] = int(generator.integers(0, 3, size = 1)[0])

    # comp_graph = nx.DiGraph()
    # for node in topo_sort :
        
    #     pass

    # print(len(layers_assignments))
    # comp_graph = Utils.ComponentGraphBuilder(model_graph).compute_components_graph(layers_assignments, model_graph)
    # print(len(comp_graph.edges))
    return None



def main() :
    onnx_model = onnx.load_model(MODEL_PATH)    
    model_graph : nx.DiGraph = OnnxModelProfiler().profile_model(onnx_model, MODEL_PATH, {})

    test(model_graph)


    network_graph = nx.DiGraph()
    network_graph.add_edge(0, 0, bandwidth = 0)
    network_graph.add_edge(0, 1, bandwidth = 100)
    network_graph.add_edge(1, 0, bandwidth = 100)
    network_graph.add_edge(1, 1, bandwidth = 0)
    network_graph.add_edge(0, 2, bandwidth = 100)
    network_graph.add_edge(2, 0, bandwidth = 100)
    network_graph.add_edge(2, 1, bandwidth = 100)
    network_graph.add_edge(1, 2, bandwidth = 100)
    network_graph.add_edge(2, 2, bandwidth = 0)

    server_profiles = {}
    for net_node in network_graph.nodes :
        server_profiles[net_node] = {}
        for mod_node in model_graph.nodes :
            if net_node == 0 :
                server_profiles[net_node][mod_node] = model_graph.nodes[mod_node]["flops"] / 1e10
            elif net_node == 1 :
                server_profiles[net_node][mod_node] = model_graph.nodes[mod_node]["flops"] / 1e12
            elif net_node == 2 :
                server_profiles[net_node][mod_node] = model_graph.nodes[mod_node]["flops"] / 1e14


    GeneticSolver.solve_problem_deap(model_graph, network_graph, server_profiles)

    pass


if __name__ == "__main__" :
    main()