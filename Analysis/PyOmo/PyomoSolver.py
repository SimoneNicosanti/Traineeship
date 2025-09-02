import networkx as nx
import pyomo.environ as pyo
import Utils
from amplpy import modules

CPLEX_PATH = "/opt/ibm/cplex/cplex/bin/x86-64_linux/cplex"
SCIP_PATH = "./SCIPOptSuite-9.2.3-Linux/bin/scip"


def define_vars(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
):
    # VARIABLES
    problem.layer_ass_vars = {}
    for layer_id in model_graph.nodes:
        for comp_id in comp_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            problem.layer_ass_vars[layer_ass_key] = pyo.Var(within=pyo.Binary)
            problem.add_component(
                "lay_ass_" + str(layer_ass_key), problem.layer_ass_vars[layer_ass_key]
            )

    problem.component_ass_vars = {}
    for comp_id in comp_graph.nodes:
        for net_node_id in network_graph.nodes:
            comp_ass_key = (comp_id, net_node_id)
            problem.component_ass_vars[comp_ass_key] = pyo.Var(within=pyo.Binary)
            problem.add_component(
                "comp_ass_" + str(comp_ass_key),
                problem.component_ass_vars[comp_ass_key],
            )

    problem.tensor_ass_vars = {}
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:
            tensor_ass_key = (tensor_id, comp_edge)
            problem.tensor_ass_vars[tensor_ass_key] = pyo.Var(within=pyo.Binary)
            problem.add_component(
                "tens_ass_" + str(tensor_ass_key),
                problem.tensor_ass_vars[tensor_ass_key],
            )

    problem.comp_edge_ass_vars = {}
    for comp_edge in comp_graph.edges:
        for net_edge in network_graph.edges:
            comp_edge_ass_key = (comp_edge, net_edge)
            problem.comp_edge_ass_vars[comp_edge_ass_key] = pyo.Var(within=pyo.Binary)
            problem.add_component(
                "comp_edge_ass_" + str(comp_edge_ass_key),
                problem.comp_edge_ass_vars[comp_edge_ass_key],
            )

    problem.start_times = {}
    for comp_id in comp_graph.nodes:
        for net_node_id in network_graph.nodes:
            start_time_key = (comp_id, net_node_id)
            problem.start_times[start_time_key] = pyo.Var(within=pyo.NonNegativeReals)
            problem.add_component(
                "start_time_" + str(start_time_key), problem.start_times[start_time_key]
            )

    problem.comp_time_prod_var = {}
    for layer_id in model_graph.nodes:
        for comp_id in comp_graph.nodes:
            for net_node_id in network_graph.nodes:
                comp_time_prod_key = (layer_id, comp_id, net_node_id)
                problem.comp_time_prod_var[comp_time_prod_key] = pyo.Var(
                    within=pyo.Binary
                )
                problem.add_component(
                    "comp_time_prod_" + str(comp_time_prod_key),
                    problem.comp_time_prod_var[comp_time_prod_key],
                )

    problem.trans_time_prod_vars = {}
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:
            for net_edge in network_graph.edges:
                tensor_trans_time_prod_key = (tensor_id, comp_edge, net_edge)
                problem.trans_time_prod_vars[tensor_trans_time_prod_key] = pyo.Var(
                    within=pyo.Binary
                )
                problem.add_component(
                    "trans_time_prod_" + str(tensor_trans_time_prod_key),
                    problem.trans_time_prod_vars[tensor_trans_time_prod_key],
                )


def define_layer_to_comp_assignment_constraints(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
):
    all_constraints = []

    ## One layer per component
    for layer_id in model_graph.nodes:
        layer_ass_sum = 0
        for comp_id in comp_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            layer_ass_sum += problem.layer_ass_vars[layer_ass_key]
        all_constraints.append(layer_ass_sum == 1)

    ## Flow of tensors constraints
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:
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

            all_constraints.append(tensor_ass_var <= tensor_source_node_ass_var)
            all_constraints.append(tensor_ass_var <= dest_nodes_ass_sum)
            all_constraints.append(
                tensor_ass_var
                >= tensor_source_node_ass_var
                + (1 / len(tensor_dest_layers)) * dest_nodes_ass_sum
                - 1
            )

    ## Input and Output Components have only one assigned layer
    for layer_id in model_graph.nodes:
        if layer_id == "InputGenerator":
            all_constraints.append(problem.layer_ass_vars[(layer_id, 0)] == 1)
        else:
            all_constraints.append(problem.layer_ass_vars[(layer_id, 0)] == 0)

        if layer_id == "OutputReceiver":
            all_constraints.append(
                problem.layer_ass_vars[(layer_id, len(comp_graph.nodes) - 1)] == 1
            )
        else:
            all_constraints.append(
                problem.layer_ass_vars[(layer_id, len(comp_graph.nodes) - 1)] == 0
            )

    return all_constraints


def define_component_to_server_assignment_constraints(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
):
    all_constraints = []

    ## One server per component
    for comp_id in comp_graph.nodes:
        comp_ass_sum = 0
        for net_node_id in network_graph.nodes:
            comp_ass_key = (comp_id, net_node_id)
            comp_ass_sum += problem.component_ass_vars[comp_ass_key]
        all_constraints.append(comp_ass_sum == 1)

    ## One server_edge per component_edge
    for comp_edge in comp_graph.edges:
        comp_edge_ass_sum = 0
        for net_edge in network_graph.edges:
            comp_edge_ass_key = (comp_edge, net_edge)
            comp_edge_ass_sum += problem.comp_edge_ass_vars[comp_edge_ass_key]
        all_constraints.append(comp_edge_ass_sum == 1)

    ## Output Flow Constraint
    for comp_edge in comp_graph.edges:
        for src_net_node_id in network_graph.nodes:
            src_comp_ass_var = problem.component_ass_vars[
                (comp_edge[0], src_net_node_id)
            ]

            rcv_comp_sum = 0
            for rcv_net_node_id in network_graph.nodes:
                rcv_comp_sum += problem.comp_edge_ass_vars[
                    (comp_edge, (src_net_node_id, rcv_net_node_id))
                ]

            all_constraints.append(src_comp_ass_var == rcv_comp_sum)

    ## Input Flow Constraint
    for comp_edge in comp_graph.edges:
        for rcv_net_node_id in network_graph.nodes:
            rcv_comp_ass_var = problem.component_ass_vars[
                (comp_edge[1], rcv_net_node_id)
            ]

            src_comp_sum = 0
            for src_net_node_id in network_graph.nodes:
                src_comp_sum += problem.comp_edge_ass_vars[
                    (comp_edge, (src_net_node_id, rcv_net_node_id))
                ]

            all_constraints.append(rcv_comp_ass_var == src_comp_sum)

    ## Correct number of components
    for net_node_id in network_graph.nodes:
        net_node_comp_sum = 0
        for comp_id in comp_graph.nodes:
            net_node_comp_key = (comp_id, net_node_id)
            net_node_comp_sum += problem.component_ass_vars[net_node_comp_key]

        all_constraints.append(
            net_node_comp_sum == network_graph.nodes[net_node_id]["tot_comps"]
        )

    ## System Component Assignment
    all_constraints.append(problem.component_ass_vars[(0, 0)] == 1)
    all_constraints.append(
        problem.component_ass_vars[(len(comp_graph.nodes) - 1, 0)] == 1
    )

    return all_constraints


def define_product_constraints(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
    tot_components: int,
):
    all_constraints = []

    ## Constraints on products
    for layer_id in model_graph.nodes:
        for comp_id in comp_graph.nodes:
            for net_node_id in network_graph.nodes:

                comp_time_prod_key = (layer_id, comp_id, net_node_id)
                comp_time_prod_var = problem.comp_time_prod_var[comp_time_prod_key]

                layer_ass_var = problem.layer_ass_vars[(layer_id, comp_id)]
                comp_ass_var = problem.component_ass_vars[(comp_id, net_node_id)]

                all_constraints.append(comp_time_prod_var <= layer_ass_var)
                all_constraints.append(comp_time_prod_var <= comp_ass_var)
                all_constraints.append(
                    comp_time_prod_var >= layer_ass_var + comp_ass_var - 1
                )

    for tensor_id in model_graph.graph["tensor_size_dict"]:
        for comp_edge in comp_graph.edges:
            for net_edge in network_graph.edges:
                trans_time_prod_var = problem.trans_time_prod_vars[
                    (tensor_id, comp_edge, net_edge)
                ]

                tensor_tx_var = problem.tensor_ass_vars[(tensor_id, comp_edge)]
                comp_edge_ass_var = problem.comp_edge_ass_vars[(comp_edge, net_edge)]

                all_constraints.append(trans_time_prod_var <= tensor_tx_var)
                all_constraints.append(trans_time_prod_var <= comp_edge_ass_var)
                all_constraints.append(
                    trans_time_prod_var >= tensor_tx_var + comp_edge_ass_var - 1
                )

    return all_constraints


def compute_computation_time(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
    prev_comp_id: int,
    prev_net_node_id: int,
    curr_comp_id: int,
    curr_net_node_id: int,
):

    comp_time = 0
    for layer_id in model_graph.nodes:
        # comp_time += (
        #     server_profiles[prev_net_node_id].get(layer_id, 0)
        #     * problem.layer_ass_vars[(layer_id, prev_comp_id)]
        # )

        comp_time += (
            server_profiles[prev_net_node_id].get(layer_id, 0)
            * problem.comp_time_prod_var[(layer_id, prev_comp_id, prev_net_node_id)]
        )

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
    prev_net_node_id,
    curr_comp_id,
    curr_net_node_id,
):
    net_bandwidth = network_graph.edges[(prev_net_node_id, curr_net_node_id)][
        "bandwidth"
    ]
    if prev_net_node_id == curr_net_node_id:
        net_bandwidth = 1e9

    transfer_time = 0
    for tensor_id in model_graph.graph["tensor_size_dict"]:
        tensor_size = model_graph.graph["tensor_size_dict"][tensor_id][1]

        # transfer_time += (
        #     tensor_size
        #     / net_bandwidth
        #     * problem.tensor_ass_vars[(tensor_id, (prev_comp_id, curr_comp_id))]
        # )

        transfer_time += (tensor_size / net_bandwidth) * problem.trans_time_prod_vars[
            (
                tensor_id,
                (prev_comp_id, curr_comp_id),
                (prev_net_node_id, curr_net_node_id),
            )
        ]

    # transfer_time = (
    #     transfer_time
    #     * problem.comp_edge_ass_vars[
    #         ((prev_comp_id, curr_comp_id), (prev_net_node_id, curr_net_node_id))
    #     ]
    # )

    return transfer_time


def compute_prev_comp_latency(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
    prev_comp_id: int,
    prev_net_node_id: int,
    curr_comp_id: int,
    curr_net_node_id: int,
):

    prev_comp_latency = 0
    prev_comp_latency += problem.start_times[(prev_comp_id, prev_net_node_id)]

    prev_comp_latency += compute_computation_time(
        problem,
        model_graph,
        network_graph,
        server_profiles,
        prev_comp_id,
        prev_net_node_id,
        curr_comp_id,
        curr_net_node_id,
    )
    prev_comp_latency += compute_transfer_time(
        problem,
        model_graph,
        network_graph,
        server_profiles,
        prev_comp_id,
        prev_net_node_id,
        curr_comp_id,
        curr_net_node_id,
    )

    return prev_comp_latency

    pass


def define_start_time_constraints(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    server_profiles: dict,
):
    topo_sort = list(nx.topological_sort(comp_graph))
    constr_list = []
    problem.start_time_dict = {}
    for curr_comp_id in topo_sort:
        for curr_net_node_id in network_graph.nodes:
            start_time_key = (curr_comp_id, curr_net_node_id)
            start_time_var = problem.start_times[start_time_key]

            for prev_comp_id in range(curr_comp_id):
                prev_comp_latency_sum = 0
                for prev_net_node_id in network_graph.nodes:
                    prev_comp_latency = compute_prev_comp_latency(
                        problem,
                        model_graph,
                        network_graph,
                        server_profiles,
                        prev_comp_id,
                        prev_net_node_id,
                        curr_comp_id,
                        curr_net_node_id,
                    )
                    prev_comp_latency_sum += prev_comp_latency

                    problem.start_time_dict[
                        (prev_comp_id, prev_net_node_id, curr_comp_id, curr_net_node_id)
                    ] = prev_comp_latency

                constr_list.append(start_time_var >= prev_comp_latency_sum)

    return constr_list


def init_problem(
    problem: pyo.ConcreteModel, model_graph: nx.DiGraph, tot_components: int
):
    topo_sort = list(nx.lexicographical_topological_sort(model_graph))

    problem.layer_ass_vars[("InputGenerator", 0)].fix(1)
    problem.layer_ass_vars[("OutputReceiver", tot_components - 1)].fix(1)

    topo_sort.remove("InputGenerator")
    topo_sort.remove("OutputReceiver")

    tot_layers = len(topo_sort)
    layers_per_comp = tot_layers // (tot_components - 2)

    for comp_id in range(1, tot_components - 1):
        for layer_id in topo_sort[
            (comp_id - 1) * layers_per_comp : (comp_id) * layers_per_comp
        ]:
            problem.layer_ass_vars[(layer_id, comp_id)].fix(1)

        if comp_id == tot_components - 2:
            for layer_id in topo_sort[(comp_id) * layers_per_comp :]:
                problem.layer_ass_vars[(layer_id, comp_id)].fix(1)

    pass


def solve_problem(
    model_graph: nx.DiGraph, network_graph: nx.DiGraph, server_profiles: dict
):

    tot_components = 0
    for net_node in network_graph.nodes:
        tot_components += network_graph.nodes[net_node]["tot_comps"]

    comp_graph = nx.DiGraph()
    for comp_id in range(tot_components):
        for next_comp_id in range(tot_components):
            if comp_id < next_comp_id:
                comp_graph.add_edge(comp_id, next_comp_id)
    Utils.draw_parallel_component_graph(comp_graph, "component_graph.png")

    problem = pyo.ConcreteModel()

    # Defining Variables
    define_vars(problem, model_graph, comp_graph, network_graph)
    print("Done Defining Variables...")

    # init_problem(problem, model_graph, tot_components)
    # print("Done Initializing Variables...")

    # Define Layer to Component Assignment Constraints
    problem.layer_ass_constraints = pyo.ConstraintList()
    for const in define_layer_to_comp_assignment_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles
    ):
        problem.layer_ass_constraints.add(const)
    print("Done Defining Layer Assignment Constraints...")

    # Define Component to Server Assignment Constraints
    problem.component_ass_constraints = pyo.ConstraintList()
    for const in define_component_to_server_assignment_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles
    ):
        problem.component_ass_constraints.add(const)
    print("Done Defining Component Assignment Constraints...")

    # Define Product Validity Constraints
    problem.product_constraints = pyo.ConstraintList()
    for const in define_product_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles, tot_components
    ):
        problem.product_constraints.add(const)
    print("Defined Product Constraints...")

    # Define Start Time Constraints For Max Resolution
    problem.start_time_constraints = pyo.ConstraintList()
    for const in define_start_time_constraints(
        problem, model_graph, comp_graph, network_graph, server_profiles
    ):
        problem.start_time_constraints.add(const)
    print("Done Defining Start Time Constraints...")

    obj = problem.start_times[(tot_components - 1, 0)]
    problem.obj = pyo.Objective(expr=obj, sense=pyo.minimize)

    solver = pyo.SolverFactory(
        "cplex",  # modules.find("scip-cpx"),
        #     # options={"bonmin.milp_solver": "Cplex"},
        executable=CPLEX_PATH,
    )  # o 'cbc', 'gurobi', 'cplex' se installatitensors
    print("Solving...")

    solver.solve(problem, tee=True)
    print(f"Idx {0} >> {problem.obj()}")

    for comp_id in comp_graph.nodes:
        for net_node_id in network_graph.nodes:
            comp_ass_key = (comp_id, net_node_id)
            if pyo.value(problem.component_ass_vars[comp_ass_key]) >= 0.9:
                print(f"Component {comp_id} on Server {net_node_id}")

    for comp_id in comp_graph.nodes:
        tot_ass_layers = 0
        for layer_id in model_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            if pyo.value(problem.layer_ass_vars[layer_ass_key]) >= 0.9:
                tot_ass_layers += 1
        print(f"Component {comp_id} has {tot_ass_layers} layers assigned")

    ass_comp_graph = nx.DiGraph()
    for comp_id in comp_graph.nodes:
        ass_nodes = set()
        for layer_id in model_graph.nodes:
            layer_ass_key = (layer_id, comp_id)
            if pyo.value(problem.layer_ass_vars[layer_ass_key]) >= 0.9:
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

    # for comp_id in :
    #     adj_var_key = (comp_id[0], comp_id[1])
    #     if problem.adjacency_vars[adj_var_key].value >= 0.9:
    #         print(f"Component {comp_id[0]} is adjacent to {comp_id[1]}")

    # for key in problem.start_time_dict:
    #     print(f"Start Time {key} >> {pyo.value(problem.start_time_dict[key])}")

    return

    for idx in range(0, 4):

        if idx % 2 == 0:
            ## Minimized in component assignment
            ## Next Will be minimized in layer
            fix_layer_ass_vars(
                problem, model_graph, network_graph, tot_components, False
            )
            fix_comp_ass_vars(problem, model_graph, network_graph, tot_components, True)

        else:
            ## Minimized in layer assignment
            ## Next Will be minimized in component
            fix_layer_ass_vars(
                problem, model_graph, network_graph, tot_components, True
            )
            fix_comp_ass_vars(
                problem, model_graph, network_graph, tot_components, False
            )

    pass


def fix_layer_ass_vars(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    tot_components: int,
    to_fix: bool,
):
    for layer_id in model_graph.nodes:
        for comp_id in range(tot_components):
            layer_ass_key = (layer_id, comp_id)
            if to_fix:
                problem.layer_ass_vars[layer_ass_key].fix()
            else:
                problem.layer_ass_vars[layer_ass_key].unfix()
    pass


def fix_comp_ass_vars(
    problem: pyo.ConcreteModel,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    tot_components: int,
    to_fix: bool,
):
    for comp_id in range(tot_components):
        for net_node_id in network_graph.nodes:
            comp_ass_key = (comp_id, net_node_id)
            if to_fix:
                problem.component_ass_vars[comp_ass_key].fix()
            else:
                problem.component_ass_vars[comp_ass_key].unfix()
    pass
