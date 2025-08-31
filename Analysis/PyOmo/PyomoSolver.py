import networkx as nx
import pyomo.environ as pyo
import itertools


def define_vars(problem : pyo.ConcreteModel, model_graph : nx.DiGraph, network_graph : nx.DiGraph, tot_components : int) :
    # VARIABLES
    problem.layer_ass_vars = {}
    for layer_id in model_graph.nodes :
        for comp_id in range(tot_components) :
            layer_ass_key = (layer_id, comp_id)
            problem.layer_ass_vars[layer_ass_key] = pyo.Var(within = pyo.Binary)
    
    problem.component_ass_vars = {}
    for comp_id in range(tot_components) :
        for net_node_id in network_graph.nodes :
            comp_ass_key = (comp_id, net_node_id)
            problem.component_ass_vars[comp_ass_key] = pyo.Var(within = pyo.Binary)
    
    problem.tensor_ass_vars = {}
    for tensor_id in model_graph.graph["tensors"] :
        for comp_edge in itertools.product(range(tot_components), range(tot_components)) :
            tensor_ass_key = (tensor_id, comp_edge)
            problem.tensor_ass_vars[tensor_ass_key] = pyo.Var(within = pyo.Binary)
    
    problem.comp_edge_ass_vars = {}
    for comp_edge in itertools.product(range(tot_components), range(tot_components)) :
        for net_edge in network_graph.edges :
            comp_edge_ass_key = (comp_edge, net_edge)
            problem.comp_edge_ass_vars[comp_edge_ass_key] = pyo.Var(within = pyo.Binary)
    
    problem.adjacency_vars = {}
    for comp_edge in itertools.product(range(tot_components), range(tot_components)) :
        adj_key = (comp_edge[0], comp_edge[1])
        problem.adjacency_vars[adj_key] = pyo.Var(within = pyo.Binary)

    problem.start_times = {}
    for comp_id in range(tot_components) :
        for net_node_id in network_graph.nodes :
            start_time_key = (comp_id, net_node_id)
            problem.start_times[start_time_key] = pyo.Var(within = pyo.NonNegativeReals)
    

def define_constraints(problem : pyo.ConcreteModel, model_graph : nx.DiGraph, network_graph : nx.DiGraph, server_profiles : dict, tot_components : int) :
    pass

def compute_computation_time(problem : pyo.ConcreteModel, model_graph : nx.DiGraph, network_graph : nx.DiGraph, server_profiles : dict, prev_comp_id : int, prev_net_node_id : int, curr_comp_id : int, curr_net_node_id : int) :

    comp_time = 0
    for layer_id in model_graph.nodes :
        comp_time += server_profiles[prev_net_node_id].get(layer_id, 0) * problem.layer_ass_vars[(layer_id, prev_comp_id)]
    
    comp_time = comp_time * problem.component_ass_vars[(prev_comp_id, prev_net_node_id)] * problem.adjacency_vars[(prev_comp_id, curr_comp_id)]

    return comp_time

def compute_transfer_time() :
    return 0

def compute_prev_comp_latency(
        problem : pyo.ConcreteModel, 
        model_graph : nx.DiGraph, 
        network_graph : nx.DiGraph, 
        server_profiles : dict, 
        prev_comp_id : int, 
        prev_net_node_id : int, 
        curr_comp_id : int, 
        curr_net_node_id : int
    ) :
    prev_comp_latency = 0

    prev_comp_latency += problem.start_times[(prev_comp_id, prev_net_node_id)]
    prev_comp_latency += compute_computation_time(problem, model_graph, network_graph, server_profiles, prev_comp_id, prev_net_node_id, curr_comp_id, curr_net_node_id)
    prev_comp_latency += compute_transfer_time()


    return prev_comp_latency

    pass

def define_start_time_constraints(problem : pyo.ConcreteModel, model_graph : nx.DiGraph, network_graph : nx.DiGraph, server_profiles : dict, tot_components : int) :

    for curr_comp_id in range(tot_components) :
        for curr_net_node_id in network_graph.nodes :
            start_time_key = (curr_comp_id, curr_net_node_id)
            start_time_var = problem.start_times[start_time_key]

            
            for prev_comp_id in range(curr_comp_id) :
                prev_comp_latency_sum = 0
                for prev_net_node_id in network_graph.nodes :
                    prev_comp_latency = compute_prev_comp_latency(
                        problem, 
                        model_graph, 
                        network_graph, 
                        server_profiles, 
                        prev_comp_id, 
                        prev_net_node_id, 
                        curr_comp_id, 
                        curr_net_node_id
                    )
                    prev_comp_latency_sum += prev_comp_latency
                
                start_time_var >= prev_comp_latency_sum
            


    pass


def solve_problem(model_graph : nx.DiGraph, network_graph : nx.DiGraph, server_profiles : dict) :

    tot_components = 0
    for net_node in network_graph.nodes :
        tot_components += network_graph.nodes[net_node]["tot_comps"]
    
    problem = pyo.ConcreteModel()

    # Defining Variables
    define_vars(problem, model_graph, network_graph, tot_components)

    # Define Constraints
    define_constraints(problem, model_graph, network_graph, server_profiles, tot_components)
    define_start_time_constraints(problem, model_graph, network_graph, server_profiles, tot_components)

    obj = problem.start_times[tot_components - 1, 0]
    problem.obj = pyo.Objective(expr = obj, sense = pyo.minimize)


    pass


