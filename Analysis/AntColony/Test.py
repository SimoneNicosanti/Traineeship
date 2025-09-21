import json

import networkx as nx
import onnx
from AntColony import Colony, ColonyParams, Problem
from ModelProfiler import OnnxModelProfiler

MODEL_PATH = "../onnx_model/yolo11x-seg/yolo11x-seg.onnx"


def main():
    with open("graph.json", "r") as f:
        data = json.load(f)
        model_graph = nx.node_link_graph(data, directed=True)

    network_graph = nx.DiGraph()
    network_graph.add_edge(0, 0, bandwidth=0)
    network_graph.add_edge(0, 1, bandwidth=100)
    network_graph.add_edge(1, 0, bandwidth=100)
    network_graph.add_edge(1, 1, bandwidth=0)
    network_graph.add_edge(0, 2, bandwidth=100)
    network_graph.add_edge(2, 0, bandwidth=100)
    network_graph.add_edge(2, 1, bandwidth=100)
    network_graph.add_edge(1, 2, bandwidth=100)
    network_graph.add_edge(2, 2, bandwidth=0)

    server_profiles = {}
    for net_node in network_graph.nodes:
        server_profiles[net_node] = {}
        for mod_node in model_graph.nodes:
            if net_node == 0:
                server_profiles[net_node][mod_node] = (
                    model_graph.nodes[mod_node]["flops"] / 1e10
                )
            elif net_node == 1:
                server_profiles[net_node][mod_node] = (
                    model_graph.nodes[mod_node]["flops"] / 1e12
                )
            elif net_node == 2:
                server_profiles[net_node][mod_node] = (
                    model_graph.nodes[mod_node]["flops"] / 1e14
                )

    problem = Problem(model_graph, network_graph, server_profiles)
    colony_params = ColonyParams(100, 0.1, 0.1, 0.1)
    colony = Colony(problem=problem, colony_params=colony_params)
    colony.run_colony(iterations=50)


if __name__ == "__main__":
    main()
