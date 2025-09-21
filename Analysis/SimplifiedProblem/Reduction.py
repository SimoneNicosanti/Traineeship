import networkx as nx

# from Parallel_Reduction import draw_sorted_dag, reduce_parallel_branches


def reduce_graph_dummy(model_graph: nx.DiGraph, max_nodes_per_part: int) -> nx.DiGraph:
    part_idx = 0
    part_size_counter = 0
    part_mapping = {}
    topo_sort = list(nx.topological_sort(model_graph))
    for layer_id in topo_sort:
        part_mapping.setdefault(part_idx, [])
        if layer_id == "InputGenerator" or layer_id == "OutputReceiver":
            part_mapping[part_idx].append(layer_id)
            part_idx += 1
            continue

        part_mapping[part_idx].append(layer_id)
        part_size_counter += 1

        if part_size_counter == max_nodes_per_part:
            part_idx += 1
            part_size_counter = 0

    coarsened_graph = nx.DiGraph()
    for part in range(part_idx):
        tot_flops = 0
        for layer_id in part_mapping[part]:
            tot_flops += model_graph.nodes[layer_id]["flops"]
        coarsened_graph.add_node(part, layers=part_mapping[part], flops=tot_flops)

    for first_part in coarsened_graph.nodes:
        first_layers = coarsened_graph.nodes[first_part]["layers"]
        for second_part in coarsened_graph.nodes:
            if first_part == second_part:
                continue

            second_layers = coarsened_graph.nodes[second_part]["layers"]

            for layer_1 in first_layers:
                for layer_2 in second_layers:
                    if model_graph.has_edge(layer_1, layer_2):
                        coarsened_graph.add_edge(first_part, second_part)

    return coarsened_graph


def find_valid_matching(curr_graph: nx.DiGraph) -> nx.DiGraph:

    matching = set()

    vertex_order = list(nx.topological_sort(curr_graph))
    edge_order = [(u, v) for u, v in curr_graph.edges]

    for node in curr_graph:
        if curr_graph.in_degree(node) == 0:
            topo_levels = {node: 0}

    for v in vertex_order:
        if curr_graph.in_degree(v) == 0:
            continue
        topo_levels[v] = 1 + max(
            topo_levels.get(u, 0) for u in curr_graph.predecessors(v)
        )

    mark = {}
    for v in vertex_order:
        mark[v] = False
        if curr_graph.in_degree(v) == 0 or curr_graph.out_degree(v) == 0:
            mark[v] = True
            print(v)

    for start_node in vertex_order:
        if mark[start_node]:
            continue

        check_edges = []
        for prev_node in curr_graph.predecessors(start_node):
            check_edges.append((prev_node, start_node))
        for next_node in curr_graph.successors(start_node):
            check_edges.append((start_node, next_node))

        check_edges.sort(key=lambda edge: edge_order.index(edge))

        for check_edge in check_edges:
            check_node = check_edge[1] if check_edge[0] == start_node else check_edge[0]

            if mark[check_node]:
                continue

            if (
                (topo_levels[start_node] != topo_levels[check_node] - 1)
                and (len(list(curr_graph.predecessors(check_node))) != 1)
                and (len(list(curr_graph.successors(start_node))) != 1)
            ):
                continue

            if check_node in curr_graph.predecessors(start_node):
                matching.add((check_node, start_node))
                for w in curr_graph.successors(check_node):
                    if topo_levels[check_node] == topo_levels[w] - 1:
                        mark[w] = True  ## Check if True of False
            else:
                matching.add((start_node, check_node))
                for w in curr_graph.successors(start_node):
                    if topo_levels[start_node] == topo_levels[w] - 1:
                        mark[w] = True  ## Check if True of False

    ## Check Matching
    for first_edge in matching:
        for second_edge in matching:
            if first_edge == second_edge:
                continue

            if not set(list(first_edge)).isdisjoint(set(list(second_edge))):
                print(first_edge, second_edge)
                raise Exception("Matching not valid")

    return matching

    pass


def reduce_with_matching(model_graph: nx.DiGraph, reduction_rounds: int = 3):
    curr_graph = model_graph

    for i in range(0, reduction_rounds):
        valid_match = find_valid_matching(curr_graph)

        merged_graph = nx.DiGraph()
        for node in curr_graph.nodes:
            in_match = False
            for edge in valid_match:
                if node in edge:
                    in_match = True
                    break
            if not in_match:
                merged_graph.add_node(
                    node,
                    is_match=False,
                    layers=[node],
                    flops=curr_graph.nodes[node]["flops"],
                )

        for edge_idx, edge in enumerate(list(valid_match)):
            match_name = f"match_{edge_idx}_{i}"
            merged_graph.add_node(
                match_name,
                is_match=True,
                layers=edge,
                flops=sum([curr_graph.nodes[node]["flops"] for node in edge]),
            )

        ## Creationg connections among nodes and matching nodes
        for first_node in merged_graph.nodes:
            for second_node in merged_graph.nodes:
                if first_node == second_node:
                    continue

                for first_node_lay in merged_graph.nodes[first_node]["layers"]:
                    for second_node_lay in merged_graph.nodes[second_node]["layers"]:
                        if (first_node_lay, second_node_lay) in curr_graph.edges:
                            merged_graph.add_edge(first_node, second_node)

        if not nx.is_directed_acyclic_graph(merged_graph):
            raise Exception("Coarsed Graph is Not acyclic")

        curr_graph = merged_graph

    return curr_graph
    pass
