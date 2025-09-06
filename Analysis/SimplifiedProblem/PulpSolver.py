import networkx as nx
import pulp
import Utils

CPLEX_PATH = "/opt/ibm/cplex/cplex/bin/x86-64_linux/cplex"
SCIP_PATH = "./SCIPOptSuite-9.2.3-Linux/bin/scip"


def define_vars(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
):
    # VARIABLES
    problem.layer_ass_vars = {}
    for layer_id in model_graph.nodes:
        for comp_id in comp_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            problem.layer_ass_vars[layer_ass_key] = pulp.LpVariable(
                name="lay_ass_" + str(layer_ass_key), cat="Binary"
            )

    problem.tensor_ass_vars = {}
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:
            tensor_ass_key = (tensor_id, comp_edge)
            problem.tensor_ass_vars[tensor_ass_key] = pulp.LpVariable(
                name="tens_ass_" + str(tensor_ass_key), cat="Binary"
            )

    problem.adj_vars = {}
    for comp_edge in comp_graph.edges:
        adj_key = (comp_edge[0], comp_edge[1])
        adj_var = pulp.LpVariable(name="adj_" + str(adj_key), cat="Binary")
        problem.adj_vars[adj_key] = adj_var

    problem.start_times = {}
    for comp_id in comp_graph.nodes:
        start_time_key = (comp_id,)
        problem.start_times[start_time_key] = pulp.LpVariable(
            name="start_time_" + str(start_time_key),
            cat=pulp.const.LpContinuous,
            lowBound=0,
        )


def define_layer_to_comp_assignment_constraints(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
):

    ## One layer per component
    for layer_id in model_graph.nodes:
        layer_ass_sum = 0
        for comp_id in comp_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            layer_ass_sum += problem.layer_ass_vars[layer_ass_key]
        problem += layer_ass_sum == 1

    ## Flow of tensors constraints
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:

            if comp_edge[0] == comp_edge[1]:
                continue

            tensor_ass_var = problem.tensor_ass_vars[(tensor_id, comp_edge)]

            tensor_source_node_id = model_graph.graph["tensor_size_dict"][tensor_id][0]
            tensor_source_node_ass_var = problem.layer_ass_vars[
                (tensor_source_node_id, comp_edge[0])
            ]
            tensor_dest_layers = set()
            for model_edge in model_graph.edges:
                if tensor_id in model_graph.edges[model_edge]["tensor_name_list"]:
                    tensor_dest_layers.add(model_edge[1])

            dest_nodes_ass_sum = 0
            for dest_layer_id in tensor_dest_layers:
                dest_node_ass_key = (dest_layer_id, comp_edge[1])
                dest_nodes_ass_sum += problem.layer_ass_vars[dest_node_ass_key]

            problem += tensor_ass_var <= tensor_source_node_ass_var
            problem += tensor_ass_var <= dest_nodes_ass_sum
            problem += (
                tensor_ass_var
                >= tensor_source_node_ass_var
                + (1 / len(tensor_dest_layers)) * dest_nodes_ass_sum
                - 1
            )

    ## Adjacency constraints
    ## Two components are adj if there is a tensor send between them
    for comp_edge in comp_graph.edges:
        adj_sum = 0
        tot_tensors = 0
        for tensor_id in model_graph.graph["tensor_size_dict"]:
            adj_sum += problem.tensor_ass_vars[(tensor_id, comp_edge)]
            tot_tensors += 1

        adj_key = (comp_edge[0], comp_edge[1])
        adj_var = problem.adj_vars[adj_key]

        problem += adj_var <= adj_sum
        problem += adj_var >= (1 / tot_tensors) * adj_sum

    ## Enforcing Topological Sort
    for comp_edge in comp_graph.edges:
        first_comp = comp_edge[0]
        second_comp = comp_edge[1]
        if second_comp < first_comp:
            problem += problem.adj_vars[(first_comp, second_comp)] == 0

    ## Input and Output Components have only one assigned layer
    for layer_id in model_graph.nodes:
        if layer_id == "InputGenerator":
            problem += problem.layer_ass_vars[(layer_id, 0)] == 1
        else:
            problem += problem.layer_ass_vars[(layer_id, 0)] == 0

        if layer_id == "OutputReceiver":
            problem += (
                problem.layer_ass_vars[(layer_id, len(comp_graph.nodes) - 1)] == 1
            )
        else:
            problem += (
                problem.layer_ass_vars[(layer_id, len(comp_graph.nodes) - 1)] == 0
            )


def compute_computation_time(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
    prev_comp_id: int,
    curr_comp_id: int,
    max_layer_flops: float,
    min_layer_flops: float,
):

    comp_time = 0
    for layer_id in model_graph.nodes:
        # comp_time += (
        #     server_profiles[prev_net_node_id].get(layer_id, 0)
        #     * problem.layer_ass_vars[(layer_id, prev_comp_id)]
        # )

        # comp_time += (
        #     server_profiles[prev_net_node_id].get(layer_id, 0)
        #     * problem.comp_time_prod_var[(layer_id, prev_comp_id, prev_net_node_id)]
        # )
        norm_flops = (model_graph.nodes[layer_id]["flops"] - min_layer_flops) / (
            max_layer_flops - min_layer_flops
        )
        comp_time += problem.layer_ass_vars[(layer_id, prev_comp_id)] * norm_flops

    # comp_time = (
    #     comp_time
    #     * problem.component_ass_vars[(prev_comp_id, prev_net_node_id)]
    #     * problem.adjacency_vars[(prev_comp_id, curr_comp_id)]
    # )

    return comp_time


def compute_transfer_time(
    problem,
    model_graph,
    network_graph,
    server_profiles,
    prev_comp_id,
    curr_comp_id,
    max_tensor_size,
    min_tensor_size,
):

    transfer_time = 0
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        tensor_size = model_graph.graph["tensor_size_dict"][tensor_id][1]

        # transfer_time += (
        #     tensor_size
        #     / net_bandwidth
        #     * problem.tensor_ass_vars[(tensor_id, (prev_comp_id, curr_comp_id))]
        # )
        norm_tensor_size = (tensor_size - min_tensor_size) / (
            max_tensor_size - min_tensor_size
        )

        transfer_time += (
            norm_tensor_size
            * problem.tensor_ass_vars[(tensor_id, (prev_comp_id, curr_comp_id))]
        )

    # transfer_time = (
    #     transfer_time
    #     * problem.comp_edge_ass_vars[
    #         ((prev_comp_id, curr_comp_id), (prev_net_node_id, curr_net_node_id))
    #     ]
    # )

    return transfer_time


def compute_prev_comp_latency(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
    prev_comp_id: int,
    curr_comp_id: int,
    max_layer_flops: float,
    min_layer_flops: float,
    max_tensor_size: float,
    min_tensor_size: float,
):

    prev_comp_latency = problem.start_times[(prev_comp_id,)]

    prev_comp_latency += compute_computation_time(
        problem,
        model_graph,
        network_graph,
        server_profiles,
        prev_comp_id,
        curr_comp_id,
        max_layer_flops,
        min_layer_flops,
    )
    # prev_comp_latency += compute_transfer_time(
    #     problem,
    #     model_graph,
    #     network_graph,
    #     server_profiles,
    #     prev_comp_id,
    #     curr_comp_id,
    #     max_tensor_size,
    #     min_tensor_size,
    # )

    return prev_comp_latency

    pass


def define_time_constraints(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
):
    ## Not bigger than the time on the device (??)
    max_layer_flops = 0
    min_layer_flops = 1e20
    big_M = 0
    for layer_id in model_graph.nodes:
        layer_flops = model_graph.nodes[layer_id]["flops"]
        if max_layer_flops < layer_flops:
            max_layer_flops = layer_flops
        if min_layer_flops > layer_flops:
            min_layer_flops = layer_flops
        big_M += layer_flops
    big_M = (big_M - min_layer_flops) / (max_layer_flops - min_layer_flops)
    print(big_M)

    max_tensor_size = 0
    min_tensor_size = 1e20
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        tensor_size = model_graph.graph["tensor_size_dict"][tensor_id][1]
        if max_tensor_size < tensor_size:
            max_tensor_size = tensor_size
        if min_tensor_size > tensor_size:
            min_tensor_size = tensor_size

    topo_sort = list(comp_graph.nodes)
    topo_sort.sort()

    constr_list = []

    problem.start_time_dict = {}
    for curr_comp_id in topo_sort:
        start_time_key = (curr_comp_id,)
        start_time_var = problem.start_times[start_time_key]

        aux_prev_comp_prod_var_list = []
        for prev_comp_id in range(curr_comp_id):
            adj_var = problem.adj_vars[(prev_comp_id, curr_comp_id)]

            prev_comp_latency = compute_prev_comp_latency(
                problem,
                model_graph,
                network_graph,
                server_profiles,
                prev_comp_id,
                curr_comp_id,
                max_layer_flops,
                min_layer_flops,
                max_tensor_size,
                min_tensor_size,
            )

            ## Product between adjacency and prev_comp_sum
            aux_prev_comp_prod_var = pulp.LpVariable(
                name="aux_prod_" + str((prev_comp_id, curr_comp_id)),
                lowBound=0,
            )
            problem += aux_prev_comp_prod_var <= big_M * adj_var
            problem += aux_prev_comp_prod_var <= prev_comp_latency
            problem += aux_prev_comp_prod_var >= prev_comp_latency - big_M * (
                1 - adj_var
            )

            aux_prev_comp_prod_var_list.append(aux_prev_comp_prod_var)

        ## Modelling max among previous components
        for aux_prev_comp_prod_var in aux_prev_comp_prod_var_list:
            problem += start_time_var >= aux_prev_comp_prod_var

    return constr_list


def solve_problem(
    model_graph: nx.DiGraph, network_graph: nx.DiGraph, server_profiles: dict
):

    tot_components = 0
    for net_node in network_graph.nodes:
        tot_components += network_graph.nodes[net_node]["tot_comps"]

    comp_graph = nx.DiGraph()
    for comp_id in range(tot_components):
        for next_comp_id in range(tot_components):
            comp_graph.add_edge(comp_id, next_comp_id)
    # Utils.draw_parallel_component_graph(comp_graph, "component_graph.png")

    problem = pulp.LpProblem("Optimization Problem", pulp.LpMinimize)

    # Defining Variables
    define_vars(problem, model_graph, comp_graph, network_graph)
    print("Done Defining Variables...")

    # init_problem(problem, model_graph, tot_components)
    # print("Done Initializing Variables...")

    # Define Layer to Component Assignment Constraints
    define_layer_to_comp_assignment_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles
    )
    print("Done Defining Layer Assignment Constraints...")

    # Define Start Time Constraints For Max Resolution
    define_time_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles
    )
    print("Done Defining Start Time Constraints...")
    obj = problem.start_times[(tot_components - 1,)]
    problem += obj

    # problem.write("model.mps", io_options={"symbolic_solver_labels": True})
    solver = pulp.CPLEX_CMD(
        path=CPLEX_PATH,
    )
    problem.solve(solver=solver)

    print("Done Solving Problem...")
    print("Status >> ", pulp.LpStatus[problem.status])
    print("Optimal Value >> ", problem.objective.value())

    for comp_id in comp_graph.nodes:
        tot_ass_layers = 0
        for layer_id in model_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            if pulp.value(problem.layer_ass_vars[layer_ass_key]) >= 0.9:
                tot_ass_layers += 1
        print(f"Component {comp_id} has {tot_ass_layers} layers assigned")

    ass_comp_graph = nx.DiGraph()
    for comp_id in comp_graph.nodes:
        ass_nodes = set()
        for layer_id in model_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            if pulp.value(problem.layer_ass_vars[layer_ass_key]) >= 0.9:
                ass_nodes.add(layer_id)

        ass_comp_graph.add_node(comp_id, nodes=ass_nodes)

    for model_edge in model_graph.edges:
        first_layer, second_layer = model_edge
        for first_comp_id in ass_comp_graph.nodes:
            for second_comp_id in ass_comp_graph.nodes:
                if first_comp_id == second_comp_id:
                    continue

                if (
                    first_layer in ass_comp_graph.nodes[first_comp_id]["nodes"]
                    and second_layer in ass_comp_graph.nodes[second_comp_id]["nodes"]
                ):
                    ass_comp_graph.add_edge(first_comp_id, second_comp_id)

    if not nx.is_directed_acyclic_graph(ass_comp_graph):
        raise Exception("Not Acyclic Graph")

    Utils.draw_parallel_component_graph(ass_comp_graph, "component_graph.png")

    for comp_id in comp_graph.edges:
        adj_var_key = (comp_id[0], comp_id[1])
        if pulp.value(problem.adj_vars[adj_var_key]) >= 0.9:
            print(f"Component {comp_id[0]} is adjacent to {comp_id[1]}")

    return
