
import onnx
import PyomoSolver
from ModelProfiler import OnnxModelProfiler
import onnx
import networkx as nx


MODEL_PATH = "../onnx_model/yolo11x-seg/yolo11x-seg.onnx"


def main() :

    onnx_model = onnx.load_model(MODEL_PATH)
    model_graph = OnnxModelProfiler().profile_model(onnx_model, "yolo11x-seg", {})

    network_graph = nx.DiGraph()

    network_graph.add_node(0, tot_comps = 5)
    network_graph.add_node(1, tot_comps = 5)

    network_graph.add_edge(0, 0, bandwidth = 0)
    network_graph.add_edge(0, 1, bandwidth = 100)
    network_graph.add_edge(1, 0, bandwidth = 100)
    network_graph.add_edge(1, 1, bandwidth = 0)

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


    PyomoSolver.solve_problem(model_graph, network_graph, server_profiles)





    pass





if __name__ == "__main__" :
    main()