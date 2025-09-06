import json

import networkx as nx
import onnx
import PulpSolver
from ModelProfiler import OnnxModelProfiler

MODEL_PATH = "../onnx_model/yolo11x-seg/yolo11x-seg.onnx"


def main():

    onnx_model = onnx.load_model(MODEL_PATH)
    model_graph = OnnxModelProfiler().profile_model(onnx_model, "yolo11x-seg", {})

    data = nx.node_link_data(model_graph)  # Converts graph to node-link format

    # Write to JSON file
    with open("graph.json", "w") as f:
        json.dump(data, f, indent=2)

    network_graph = nx.DiGraph()

    network_graph.add_node(0, tot_comps=3)
    network_graph.add_node(1, tot_comps=2)
    # network_graph.add_node(2, tot_comps=1)

    network_graph.add_edge(0, 0, bandwidth=0)
    network_graph.add_edge(0, 1, bandwidth=100)
    network_graph.add_edge(1, 0, bandwidth=100)
    network_graph.add_edge(1, 1, bandwidth=0)
    # network_graph.add_edge(0, 2, bandwidth=100)
    # network_graph.add_edge(2, 0, bandwidth=100)
    # network_graph.add_edge(2, 1, bandwidth=100)
    # network_graph.add_edge(1, 2, bandwidth=100)
    # network_graph.add_edge(2, 2, bandwidth=0)

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

    PulpSolver.solve_problem(model_graph, network_graph, server_profiles)

    pass


if __name__ == "__main__":
    main()
