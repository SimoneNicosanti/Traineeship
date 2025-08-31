import random
import time
import networkx as nx
import numpy as np
from deap import base, creator, tools, algorithms
import Utils


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MAX_NODES_PER_PART = 1e10
NODES_TAKE_PROB = 0.75
MUTATION_POINTS = 5

# ===================================
# 1. Setup DEAP e Fitness
# ===================================

# Fitness minimizza latency (quindi max di 1/latency)
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual(topo_sort, server_num):

    chromosome = []
    
    prev_server = None
    node_idx = 0
    while node_idx < len(topo_sort):

        ## Take Random Server
        curr_server = random.randint(0, server_num-1)
        while curr_server == prev_server :
            curr_server = random.randint(0, server_num-1)
        
        tot_comp_nodes = 0
        while (tot_comp_nodes == 0) or (tot_comp_nodes < MAX_NODES_PER_PART and random.random() < NODES_TAKE_PROB):
            tot_comp_nodes += 1
            chromosome.append(curr_server)

            node_idx += 1
            if node_idx == len(topo_sort):
                break

        prev_server = curr_server
        pass
    
    ## Assign first and last node to the device server
    chromosome[0] = 0
    chromosome[-1] = 0
    
    return chromosome


def compute_latency_comp(component_graph : nx.DiGraph, model_graph : nx.DiGraph, networkx_graph : nx.DiGraph, server_profiles : dict[int, dict[str, float]]) :

    topological_sort = nx.topological_sort(component_graph)

    component_comp_end_time_dict = {}
    component_trans_next_comp_finish_time = {}

    for curr_comp_id in topological_sort :
        ## This will be current component start time
        # print("Evaluating >> ", curr_comp_id)
        comp_start_time = 0
        for prev_comp_id in component_graph.predecessors(curr_comp_id) :
            prev_comp_to_curr_comp_trans_end_time = component_trans_next_comp_finish_time[prev_comp_id][curr_comp_id]
            comp_start_time = max(comp_start_time, prev_comp_to_curr_comp_trans_end_time)

        curr_compon_computation_end_time = comp_start_time
        for layer_id in component_graph.nodes[curr_comp_id]["nodes"]:
            curr_compon_computation_end_time += server_profiles[curr_comp_id[0]].get(layer_id, -1)#.get("nq_avg_time", 0)
            if server_profiles[curr_comp_id[0]] == 1 :
                curr_compon_computation_end_time -= 0.01
        component_comp_end_time_dict[curr_comp_id] = curr_compon_computation_end_time

        # print(f"Component {curr_comp_id} Comp End Time >> ", )
        curr_comp_trans_end_time = curr_compon_computation_end_time
        for next_comp_id in component_graph.successors(curr_comp_id) :
            next_comp_tx_time = 0
            for tensor_name in component_graph.edges[curr_comp_id, next_comp_id]["tensors"]:
                if curr_comp_id[0] == next_comp_id[0]:
                    # comp_tx_time += 1
                    network_edge_bw = 1e100
                else :
                    network_edge_bw = networkx_graph.edges[curr_comp_id[0], next_comp_id[0]]["bandwidth"]
                tensor_size = model_graph.graph["tensor_size_dict"][tensor_name][1]
                next_comp_tx_time += tensor_size / network_edge_bw
            
            curr_comp_trans_end_time += next_comp_tx_time

            component_trans_next_comp_finish_time.setdefault(curr_comp_id, {})
            component_trans_next_comp_finish_time[curr_comp_id][next_comp_id] = curr_comp_trans_end_time
    
    out_components = []
    for comp_id in component_graph.nodes :
        if component_graph.out_degree(comp_id) == 0 :
            out_components.append(comp_id)

    if len(out_components) == 0 :
        raise Exception("Invalid Out Component")

    return max(component_comp_end_time_dict[out_component] for out_component in out_components)



def topo_comp_graph(individual : list[int], topo_sort : list[str], model_graph : nx.DiGraph) -> nx.DiGraph:

    comp_graph = nx.DiGraph()

    gene_idx = 0
    curr_comp_idx = 0

    comp_graph.add_node((individual[gene_idx], curr_comp_idx), nodes = [topo_sort[gene_idx]])
    gene_idx += 1
    curr_comp_idx += 1

    ## Extracting nodes from topological sort
    while gene_idx < len(individual) - 1:
        layer_server = individual[gene_idx]

        curr_comp_layers = []
        next_gene_idx = gene_idx
        while next_gene_idx < len(individual) - 1 and individual[next_gene_idx] == layer_server :
            layer_name = topo_sort[next_gene_idx]
            curr_comp_layers.append(layer_name)
            next_gene_idx += 1
    
        comp_graph.add_node((layer_server, curr_comp_idx), nodes = curr_comp_layers)
        
        gene_idx = next_gene_idx
        curr_comp_idx += 1
    
    comp_graph.add_node((individual[-1], curr_comp_idx), nodes = [topo_sort[-1]])
    
    ## Building Edges in component graph
    for comp_node in comp_graph.nodes :
        for next_comp_node in comp_graph.nodes :
            if comp_node == next_comp_node :
                continue

            for layer_node in comp_graph.nodes[comp_node]["nodes"] :
                for next_layer_node in comp_graph.nodes[next_comp_node]["nodes"] :
                    if (layer_node, next_layer_node) in model_graph.edges :
                        if (comp_node, next_comp_node) not in comp_graph.edges :
                            comp_graph.add_edge(comp_node, next_comp_node, tensors = set())

                        edge_tensors = model_graph.edges[layer_node, next_layer_node]["tensor_name_list"]
                        comp_graph.edges[comp_node, next_comp_node]["tensors"].update(edge_tensors)


    if not nx.is_directed_acyclic_graph(comp_graph) :
        raise Exception("Invalid Component Graph")

    out_comps = 0
    for comp_node in comp_graph.nodes :
        if comp_graph.out_degree(comp_node) == 0 :
            out_comps += 1

    # if out_comps != 1 :
    #     # Utils.draw_parallel_component_graph(comp_graph)
    #     raise Exception("Invalid Component Graph")

    # Utils.draw_parallel_component_graph(comp_graph)

    return comp_graph

        

    pass


# ===================================
# 2. Fitness wrapper
# ===================================
def fitness_deap(individual, model_graph, topo_sort, network_graph, server_profiles, components_graph_builder):

    # Costruisco layer_assignment_map
    layer_assignments_map = {}
    for layer_idx, node in enumerate(model_graph.nodes):
        if model_graph.nodes[node]["idx"] == layer_idx:
            layer_assignments_map[node] = individual[layer_idx]

    layer_assignments_map["InputGenerator"] = 0
    layer_assignments_map["OutputReceiver"] = 0

    # Component graph
    component_graph = topo_comp_graph(individual, topo_sort, model_graph)

    # E2E latency
    e2e_latency = compute_latency_comp(component_graph, model_graph, network_graph, server_profiles)
    # print("Latency >> ", e2e_latency)

    # if e2e_latency == 0 :
    #     print(individual)

    return (e2e_latency, )

def mate_context(ind1, ind2) :

    new_ind_1 = []
    new_ind_2 = []

    for idx in range(0, len(ind1)) :

        if ind1[idx] == ind2[idx] :
            new_ind_1.append(ind1[idx])
            new_ind_2.append(ind2[idx])
        else :
            new_ind_1.append(ind2[idx])
            new_ind_2.append(ind1[idx])

    new_ind_1[0] = 0
    new_ind_1[-1] = 0
    new_ind_2[0] = 0
    new_ind_2[-1] = 0

    new_ind_1 = creator.Individual(new_ind_1)
    new_ind_2 = creator.Individual(new_ind_2)

    return (new_ind_1, new_ind_2)

def mutate_context(individual, mutpb):

    if random.random() >= mutpb:
        return (individual, )

    exclude_points = set()
    for i in range(0, MUTATION_POINTS) :
        mutation_idx = random.randint(0, len(individual) - 1)
        while mutation_idx in exclude_points :
            mutation_idx = random.randint(0, len(individual) - 1)

        if mutation_idx < len(individual) - 2 :
            individual[mutation_idx + 1] = individual[mutation_idx]
            individual[mutation_idx + 2] = individual[mutation_idx]
            exclude_points.add(mutation_idx + 1)
            exclude_points.add(mutation_idx + 2)
        if mutation_idx > 1 :
            individual[mutation_idx - 1] = individual[mutation_idx]
            individual[mutation_idx - 2] = individual[mutation_idx]
            exclude_points.add(mutation_idx - 1)
            exclude_points.add(mutation_idx - 2)
        
        exclude_points.add(mutation_idx)

    # mutated = tools.mutShuffleIndexes(individual, indpb)[0]

    individual[0] = 0
    individual[-1] = 0
    
    return (individual, )

# ===================================
# 3. DEAP Toolbox
# ===================================
def setup_deap(model_graph, network_graph, server_profiles):

    topo_sort = Utils.kahn_topo_sort_with_dfs(model_graph)

    nodes_num = len(model_graph.nodes)
    server_num = len(network_graph.nodes)
    components_graph_builder = Utils.ComponentGraphBuilder(model_graph)

    # CREATOR → Minimizzazione
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual(topo_sort, server_num))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_deap, 
                     model_graph=model_graph, 
                     topo_sort = topo_sort,
                     network_graph=network_graph, 
                     server_profiles=server_profiles,
                     components_graph_builder=components_graph_builder)
    
    # Individual Selection
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Individual Crossover
    toolbox.register("mate", mate_context,
        # indpb=0.2,
        # model_graph=model_graph, 
        # topo_sort=topo_sort, 
        # network_graph=network_graph, 
        # server_profiles=server_profiles, 
        # components_graph_builder=components_graph_builder
    )
    
    # Individual Mutation
    toolbox.register("mutate", mutate_context)
    

    return toolbox, components_graph_builder


# ===================================
# 4. Esecuzione GA
# ===================================
def solve_problem_deap(model_graph, network_graph, server_profiles):
    toolbox, components_graph_builder = setup_deap(model_graph, network_graph, server_profiles)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    # stats.register("e2e", lambda fits: 1 / np.max(fits))


    # Parallelizzazione (facoltativa)
    import multiprocessing
    pool = multiprocessing.Pool(10)
    toolbox.register("map", pool.map)

    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, ngen=500, 
                        stats=stats, halloffame=hof, verbose=True)

    best_solution = hof[0]
    best_fitness = hof[0].fitness.values[0]
    # print("Miglior soluzione:", best_solution)
    # print("Fitness:", best_fitness)
    print("E2E Latency:", best_fitness)

    best_solution_map = {}
    for layer_idx, server_id in enumerate(best_solution):
        for model_node in model_graph.nodes:
            if model_graph.nodes[model_node]["idx"] == layer_idx:
                best_solution_map[model_node] = server_id
                break
    best_solution_map["InputGenerator"] = 0
    best_solution_map["OutputReceiver"] = 0
    
    topo_sort = Utils.kahn_topo_sort_with_dfs(model_graph)
    best_component_graph = topo_comp_graph(best_solution, topo_sort, model_graph)
    print(best_component_graph.edges)

    Utils.draw_parallel_component_graph(best_component_graph, )

    # for node_idx, server_id in enumerate(best_solution):
    #     for model_node in model_graph.nodes:
    #         if model_graph.nodes[model_node]["idx"] == node_idx:
    #             print(f"Layer {model_node} -> Server {server_id}")
    #             break
        

    return best_solution, best_fitness
