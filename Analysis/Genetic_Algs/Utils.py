
import time
from matplotlib import pyplot as plt
import networkx as nx


class ComponentGraphBuilder :
    def __init__(self, model_graph : nx.DiGraph) :
        self.model_graph = model_graph

        self.topo_sort = list(nx.topological_sort(model_graph))

        self.descendants_map = {}
        for node_id in model_graph.nodes:
            self.descendants_map[node_id] = set(nx.descendants(model_graph, node_id))

        self.successors_map = {}
        for node_id in model_graph.nodes:
            self.successors_map[node_id] = set(model_graph.successors(node_id))

        self.ancestors_map = {}
        for node_id in model_graph.nodes:
            self.ancestors_map[node_id] = set(nx.ancestors(model_graph, node_id))

        self.previous_map = {}
        for node_id in model_graph.nodes:
            prev_set = set(model_graph.predecessors(node_id))

            self.previous_map[node_id] = prev_set if len(prev_set) > 0 else set()



    def compute_components_graph(self, layers_assignments : dict[str, int], model_graph : nx.DiGraph) -> nx.DiGraph :
        start = time.perf_counter_ns()
        nodes_components : dict[str, tuple[int,int]] = self.find_nodes_components(layers_assignments, model_graph)
        end = time.perf_counter_ns()
        # print("Find Components Time Time >> ", (end - start) * 1e-9)

        start = time.perf_counter_ns()
        component_graph = nx.DiGraph()
        
        for layer_id in model_graph.nodes:
            component_id = nodes_components[layer_id]
            if component_id not in component_graph.nodes:
                component_graph.add_node(component_id, nodes = set())

            component_graph.nodes[component_id]["nodes"].add(layer_id)
            
        for layer_edge in model_graph.edges:
            first_comp_id = nodes_components[layer_edge[0]]
            second_comp_id = nodes_components[layer_edge[1]]

            if first_comp_id != second_comp_id :
                if (first_comp_id, second_comp_id) not in component_graph.edges :
                    component_graph.add_edge(first_comp_id, second_comp_id, tensors = set())

                edge_tensors = model_graph.edges[layer_edge]["tensor_name_list"]
                component_graph.edges[first_comp_id, second_comp_id]["tensors"].update(edge_tensors)

        end = time.perf_counter_ns()
        # print("Build Component Graph Time >> ", (end - start) * 1e-9)
        
        start = time.perf_counter_ns()
        if nx.is_directed_acyclic_graph(component_graph):
            end = time.perf_counter_ns()
            # print("Check Component Graph Time >> ", (end - start) * 1e-9)
            return component_graph
        else:
            cycle = nx.find_cycle(component_graph, orientation="original")
            print(cycle)
            # for edge in cycle:
            #     first, second, dir = edge
            #     print(f"First Comp {first} Nodes >> ", component_graph.nodes[first]["nodes"])
            #     print(f"Second Comp {second} Nodes >> ", component_graph.nodes[second]["nodes"])
            #     print("")
            raise Exception("Component Graph is not DAG")


    def find_nodes_components(self,
        layers_assignments: dict[str, int], model_graph: nx.DiGraph
    ):

        nodes_components : dict[str, tuple[int,int]] = {}

        # model_name = solved_model_graph.graph["name"]
        top_order: list[str] = list(nx.topological_sort(model_graph))

        next_comp_dict: dict[int, int] = {}

        used_net_nodes = set(layers_assignments.values())
        #print("Used Net Nodes: ", used_net_nodes)
        for net_node_id in used_net_nodes:
            next_comp_dict[net_node_id] = 0

        node_dependency_dict: dict[str, set[tuple[int,int]]] = {}
        node_possible_dict: dict[str, set[tuple[int,int]]] = {}
        component_dependency_dict: dict[tuple[int,int], set[tuple[int,int]]] = {}

        for node_id in top_order:
            node_dependency_dict[node_id] = set()
            node_possible_dict[node_id] = set()


        for node_id in top_order:
            # print("Node: ", node_id)
            server_id = layers_assignments[node_id]

            node_dependency_set = node_dependency_dict[node_id]
            node_possible_set = node_possible_dict[node_id]

            exclude_set = set()
            for dep_comp_id in node_dependency_set:
                for poss_comp_id in node_possible_set:
                    if poss_comp_id in component_dependency_dict[dep_comp_id]:
                        exclude_set.add(poss_comp_id)
            
            for prev_node_id in self.previous_map[node_id]:
                prev_node_comp = nodes_components[prev_node_id]
                for dep_comp_id in node_dependency_set:
                    for poss_comp_id in node_possible_set:
                        if poss_comp_id in component_dependency_dict[prev_node_comp]:
                            exclude_set.add(poss_comp_id)
                        # if poss_comp_id in node_dependency_dict[prev_node_id]:
                        #     exclude_set.add(poss_comp_id)
            
            
            # for prev_node_id in self.previous_map[node_id]:
            #     prev_node_comp = nodes_components[prev_node_id]
            #     for poss_comp_id in node_possible_set:
            #         if poss_comp_id in component_dependency_dict[prev_node_comp]:
            #             exclude_set.add(poss_comp_id)
            #         if poss_comp_id in node_dependency_dict[prev_node_id]:
            #             exclude_set.add(poss_comp_id)
                # prev_comp_id = nodes_components[prev_node_id]
                # exclude_set.update(component_dependency_dict[prev_comp_id])
                # if prev_comp_id not in exclude_set:
                #     node_dependency_set.add(prev_comp_id)
                #     component_dependency_dict[prev_comp_id].add(nodes_components[node_id])

            difference_set = node_possible_set - exclude_set

            if (
                len(difference_set) == 0
                or node_id == "InputGenerator" ## Generator Node
                or node_id == "OutputReceiver" ## Receiver Node
            ):
                ## No possible component
                ## Generate new component
                curr_comp_idx = next_comp_dict[server_id]
                node_comp = (server_id, curr_comp_idx)
                next_comp_dict[server_id] += 1
            else:
                ## Take one component in the difference set
                node_comp = list(difference_set)[0]

            ## Setting found node comp
            nodes_components[node_id] = node_comp

            ## Following nodes having the same server can be in the same component except for
            ## input and output node
            if node_id != "InputGenerator" and node_id != "OutputReceiver" :
                for next_node_id in self.successors_map[node_id]:
                    neigh_server_id = layers_assignments[next_node_id]
                    if (server_id == neigh_server_id):
                        ## Same server --> Setting possible component
                        node_possible_dict[next_node_id].add(node_comp)

                # parallel_nodes = (
                #     set(model_graph.nodes)
                #     - self.descendants_map[node_id]
                #     - self.ancestors_map[node_id]
                #     - {node_id}
                # )

                # ## Parallel nodes having the same server can be in the same component
                # for paral_node_id in parallel_nodes:
                #     par_server_id = layers_assignments[paral_node_id]

                #     if (
                #         par_server_id == server_id
                #     ):
                #         ## Same server --> Setting possible component
                #         node_possible_dict[paral_node_id].add(node_comp)

            ## Expanding component dependency
            component_dependency_dict.setdefault(node_comp, set())

            ## The component we add the node to will depend by all the dependencies
            ## of the added node
            component_dependency_dict[node_comp] = component_dependency_dict[
                node_comp
            ].union(node_dependency_dict[node_id] - set([node_comp]))

            ## All other components having the current component as dependecy
            ## Will have their dependencies updated with the added node dependencies
            # for other_comp in component_dependency_dict.keys() :
            #     if node_comp in component_dependency_dict[other_comp]:
            #         component_dependency_dict[other_comp] = component_dependency_dict[other_comp].union(node_dependency_dict[node_id] - set([node_comp]))

            ## All descendants node will depend by both this component
            ## and all its dependencies
            for descendant_id in self.descendants_map[node_id]:
                node_dependency_dict[descendant_id].add(node_comp)
                # node_dependency_dict[descendant_id] = node_dependency_dict[descendant_id].union(component_dependency_dict[node_comp])
        # print("Components found >> ", next_comp_dict)

        return nodes_components



def kahn_topo_sort_with_dfs(model_graph : nx.DiGraph):

    topo_graph = model_graph.copy()

    topo_sort = []
    next_nodes = []

    for curr_node in topo_graph:
        if topo_graph.in_degree(curr_node) == 0:
            next_nodes.append(curr_node)

    while len(next_nodes) > 0:
        curr_node = next_nodes.pop(0)

        curr_node_child = list(model_graph.successors(curr_node))
        curr_node_child.sort()
        
        topo_graph.remove_node(curr_node)
        topo_sort.append(curr_node)

        for child in curr_node_child:
            if topo_graph.in_degree(child) == 0:
                ## Traversing current nodes children first
                next_nodes.insert(0, child)
    
    for node_idx, node_name in enumerate(topo_sort):
        for prev_node_name in model_graph.predecessors(node_name):
            prev_node_idx = topo_sort.index(prev_node_name)

            if prev_node_idx > node_idx:
                raise Exception("Invalid Topo Sort")
    
    # print(topo_sort)

    return topo_sort



## CORRRECTED VERSION TO BE USED IN THE THESIS
## Check Last Work Part
# def find_nodes_components(self,
#         layers_assignments: dict[str, int], model_graph: nx.DiGraph
#     ):

#         nodes_components : dict[str, tuple[int,int]] = {}

#         # model_name = solved_model_graph.graph["name"]
#         top_order: list[str] = list(nx.topological_sort(model_graph))

#         next_comp_dict: dict[int, int] = {}

#         used_net_nodes = set(layers_assignments.values())
#         #print("Used Net Nodes: ", used_net_nodes)
#         for net_node_id in used_net_nodes:
#             next_comp_dict[net_node_id] = 0

#         node_dependency_dict: dict[str, set[tuple[int,int]]] = {}
#         node_possible_dict: dict[str, set[tuple[int,int]]] = {}
#         component_dependency_dict: dict[tuple[int,int], set[tuple[int,int]]] = {}

#         for node_id in top_order:
#             node_dependency_dict[node_id] = set()
#             node_possible_dict[node_id] = set()

#         for node_id in top_order:
#             server_id = layers_assignments[node_id]

#             node_dependency_set = node_dependency_dict[node_id]
#             node_possible_set = node_possible_dict[node_id]

#             exclude_set = set()
#             for dep_comp_id in node_dependency_set:
#                 for poss_comp_id in node_possible_set:
#                     if poss_comp_id in component_dependency_dict[dep_comp_id]:
#                         exclude_set.add(poss_comp_id)

#             difference_set = node_possible_set - exclude_set

#             if (
#                 len(difference_set) == 0
#                 or node_id == "InputGenerator" ## Generator Node
#                 or node_id == "OutputReceiver" ## Receiver Node
#             ):
#                 ## No possible component
#                 ## Generate new component
#                 curr_comp_idx = next_comp_dict[server_id]
#                 node_comp = (server_id, curr_comp_idx)
#                 next_comp_dict[server_id] += 1
#             else:
#                 ## Take one component in the difference set
#                 node_comp = list(difference_set)[0]

#             ## Setting found node comp
#             nodes_components[node_id] = node_comp

#             ## Following nodes having the same server can be in the same component except for
#             ## input and output node
#             if node_id != "InputGenerator" and node_id != "OutputReceiver" :
#                 for next_node_id in self.successors_map[node_id]:
#                     neigh_server_id = layers_assignments[next_node_id]
#                     if (server_id == neigh_server_id):
#                         ## Same server --> Setting possible component
#                         node_possible_dict[next_node_id].add(node_comp)

#                 parallel_nodes = (
#                     set(model_graph.nodes)
#                     - self.descendants_map[node_id]
#                     - self.ancestors_map[node_id]
#                     - {node_id}
#                 )

#                 ## Parallel nodes having the same server can be in the same component
#                 for paral_node_id in parallel_nodes:
#                     par_server_id = layers_assignments[paral_node_id]

#                     if (
#                         par_server_id == server_id
#                     ):
#                         ## Same server --> Setting possible component
#                         node_possible_dict[paral_node_id].add(node_comp)

#             ## Expanding component dependency
#             component_dependency_dict.setdefault(node_comp, set())

#             ## The component we add the node to will depend by all the dependencies
#             ## of the added node
#             component_dependency_dict[node_comp] = component_dependency_dict[
#                 node_comp
#             ].union(node_dependency_dict[node_id] - set([node_comp]))

#             ## All other components having the current component as dependecy
#             ## Will have their dependencies updated with the added node dependencies
#             for other_comp in component_dependency_dict.keys() :
#                 if node_comp in component_dependency_dict[other_comp]:
#                     component_dependency_dict[other_comp] = component_dependency_dict[other_comp].union(node_dependency_dict[node_id] - set([node_comp]))

#             ## All descendants node will depend by both this component
#             ## and all its dependencies
#             for descendant_id in self.descendants_map[node_id]:
#                 node_dependency_dict[descendant_id].add(node_comp)
#                 node_dependency_dict[descendant_id] = node_dependency_dict[descendant_id].union(component_dependency_dict[node_comp])

#         # print("Components found >> ", next_comp_dict)

#         return nodes_components


def draw_parallel_component_graph(component_graph: nx.DiGraph, filename=None):
    """
    Draw a component graph with a hierarchical layout highlighting parallelism.
    Nodes on the same topological level (parallelizable components) are aligned vertically.
    """
    plt.figure(figsize=(14, 10))

    # Compute topological levels
    levels = {}
    for node in nx.topological_sort(component_graph):
        if component_graph.in_degree(node) == 0:
            levels[node] = 0
        else:
            levels[node] = 1 + max(levels[pred] for pred in component_graph.predecessors(node))

    # Group nodes by level
    level_nodes = {}
    for node, lvl in levels.items():
        level_nodes.setdefault(lvl, []).append(node)

    # Assign positions
    pos = {}
    max_nodes_per_level = max(len(nodes) for nodes in level_nodes.values())
    for lvl, nodes in level_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            # Distribute nodes horizontally for parallelism
            x = lvl
            y = (max_nodes_per_level - n)/2 - i  # center vertically
            pos[node] = (x, y)

    # Node colors based on level
    colors = [levels[node] for node in component_graph.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(
        component_graph, pos,
        node_size=800,
        node_color=colors,
        cmap=plt.cm.viridis,
        alpha=0.9
    )

    # Draw labels
    labels = {node: f"{node[0]}:{node[1]}" for node in component_graph.nodes}  # e.g., server:comp_idx
    nx.draw_networkx_labels(component_graph, pos, labels=labels, font_size=10, font_weight='bold')

    # Draw edges
    nx.draw_networkx_edges(
        component_graph, pos,
        arrowstyle='-|>', arrowsize=20,
        edge_color='gray', width=2,
        connectionstyle='arc3,rad=0.2'  # <--- makes edges curved
    )


    plt.axis('off')
    plt.title("Parallel Component Graph", fontsize=16)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()