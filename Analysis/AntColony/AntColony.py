import networkx as nx
import numpy as np
import Utils


class Problem:
    def __init__(
        self,
        model_graph: nx.DiGraph,
        network_graph: nx.DiGraph,
        server_profiles: dict[int, dict[str, float]],
    ) -> None:
        self.model_graph = model_graph
        self.server_profiles = server_profiles
        self.topo_sorting = list(nx.topological_sort(model_graph))
        self.network_graph = network_graph
        self.component_graph_builder = Utils.ComponentGraphBuilder(model_graph)

    def get_topo_sorting(self) -> list:
        return self.topo_sorting

    def get_num_layers(self):
        return len(self.model_graph.nodes)

    def get_num_servers(self):
        return len(self.network_graph.nodes)

    def compute_solution_value(self, assignments: list[int]):
        ## Compute graph components
        ## Compute problem latency
        layer_id_ass = {}
        for layer_idx, ass_server_id in enumerate(assignments):
            layer_id = self.topo_sorting[layer_idx]
            layer_id_ass[layer_id] = ass_server_id

        component_graph = self.component_graph_builder.compute_components_graph(
            layer_id_ass, self.model_graph
        )
        e2e_latency = self.__compute_e2e_latency(
            component_graph, self.model_graph, self.network_graph, self.server_profiles
        )
        return e2e_latency

    def __compute_e2e_latency(
        self, component_graph, model_graph, network_graph, server_profiles
    ):
        topological_sort = nx.topological_sort(component_graph)

        component_comp_end_time_dict = {}
        component_trans_next_comp_finish_time = {}

        for curr_comp_id in topological_sort:
            ## This will be current component start time
            # print("Evaluating >> ", curr_comp_id)
            comp_start_time = 0
            for prev_comp_id in component_graph.predecessors(curr_comp_id):
                prev_comp_to_curr_comp_trans_end_time = (
                    component_trans_next_comp_finish_time[prev_comp_id][curr_comp_id]
                )
                comp_start_time = max(
                    comp_start_time, prev_comp_to_curr_comp_trans_end_time
                )

            curr_compon_computation_end_time = comp_start_time
            for layer_id in component_graph.nodes[curr_comp_id]["nodes"]:
                curr_compon_computation_end_time += server_profiles[
                    curr_comp_id[0]
                ].get(
                    layer_id, -1
                )  # .get("nq_avg_time", 0)
                if server_profiles[curr_comp_id[0]] == 1:
                    curr_compon_computation_end_time -= 0.01
            component_comp_end_time_dict[curr_comp_id] = (
                curr_compon_computation_end_time
            )

            # print(f"Component {curr_comp_id} Comp End Time >> ", )
            curr_comp_trans_end_time = curr_compon_computation_end_time
            for next_comp_id in component_graph.successors(curr_comp_id):
                next_comp_tx_time = 0
                for tensor_name in component_graph.edges[curr_comp_id, next_comp_id][
                    "tensors"
                ]:
                    if curr_comp_id[0] == next_comp_id[0]:
                        # comp_tx_time += 1
                        network_edge_bw = 1e100
                    else:
                        network_edge_bw = network_graph.edges[
                            curr_comp_id[0], next_comp_id[0]
                        ]["bandwidth"]
                    tensor_size = model_graph.graph["tensor_size_dict"][tensor_name][1]
                    next_comp_tx_time += tensor_size / network_edge_bw

                curr_comp_trans_end_time += next_comp_tx_time

                component_trans_next_comp_finish_time.setdefault(curr_comp_id, {})
                component_trans_next_comp_finish_time[curr_comp_id][
                    next_comp_id
                ] = curr_comp_trans_end_time

        out_components = []
        for comp_id in component_graph.nodes:
            if component_graph.out_degree(comp_id) == 0:
                out_components.append(comp_id)

        if len(out_components) == 0:
            raise Exception("Invalid Out Component")

        return max(
            component_comp_end_time_dict[out_component]
            for out_component in out_components
        )


class ColonyParams:
    def __init__(self, num_ants: int, alpha: float, beta: float, rho: float) -> None:
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho


class PheromoneMap:
    def __init__(self, num_layers: int, num_servers: int) -> None:
        self.pheronmone_map = np.ones((num_layers, num_servers))

    def get_pheromone_vector(self, layer_idx: int) -> np.ndarray[float]:
        return self.pheronmone_map[layer_idx]

    def pheronmone_evaporate(self, evaporation_rate: float):
        self.pheronmone_map *= 1 - evaporation_rate

    def update_pheromone_value(
        self, layer_idx: int, server_id: int, contribution: float
    ):
        self.pheronmone_map[layer_idx][server_id] += contribution


class Colony:
    def __init__(self, problem: Problem, colony_params: ColonyParams) -> None:
        self.colony_params = colony_params
        self.problem = problem
        self.pheromone_map = PheromoneMap(
            self.problem.get_num_layers(), self.problem.get_num_servers()
        )
        self.ants = [
            Ant(problem, self.pheromone_map, ant_idx)
            for ant_idx in range(self.colony_params.num_ants)
        ]
        self.best_solution = None
        self.best_solution_value = float("inf")

    def run_colony(self, iterations: int):
        for _ in range(iterations):
            for ant in self.ants:
                ant.generate_solution()

            self.update_pheromone_map()
            self.update_best_solution()

            print("Best Solution Value >> ", self.best_solution_value)

        return self.best_solution

    def update_pheromone_map(self):
        self.pheromone_map.pheronmone_evaporate(self.colony_params.rho)
        for ant in self.ants:
            contribution = 1 / ant.current_assignment_value
            for layer_idx, server_id in enumerate(ant.current_assignment):
                self.pheromone_map.update_pheromone_value(
                    layer_idx, server_id, contribution
                )

    def update_best_solution(self):
        best_value = self.best_solution_value
        for ant in self.ants:
            ant_value = ant.current_assignment_value
            if ant_value < best_value:
                best_value = ant_value
                self.best_solution = ant.current_assignment
        self.best_solution_value = best_value


class Ant:
    def __init__(
        self, problem: Problem, pheronmone_map: PheromoneMap, ant_idx: int
    ) -> None:
        self.problem = problem
        self.pheronmone_map = pheronmone_map
        self.current_assignment = None
        self.random_generator = np.random.default_rng(seed=ant_idx)

    ## Returns a list of layer assignments
    def generate_solution(self) -> list[int]:
        assignments: list[int] = []
        for layer_idx, _ in enumerate(self.problem.get_topo_sorting()):
            assigned_net_id = self.choose_layer_assignment(layer_idx)
            assignments.append(assigned_net_id)
        self.current_assignment = assignments
        self.current_assignment_value = self.problem.compute_solution_value(
            self.current_assignment
        )

        return assignments

    def choose_layer_assignment(self, layer_idx: int) -> int:
        if layer_idx == 0 or layer_idx == self.problem.get_num_layers() - 1:
            ## Input and Output layer always assigned to server 0
            return 0

        pheronmone_vector = self.pheronmone_map.get_pheromone_vector(layer_idx)
        server_id = self.random_generator.choice(
            len(pheronmone_vector), p=pheronmone_vector / sum(pheronmone_vector)
        )
        return int(server_id)
