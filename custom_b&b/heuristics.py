import numpy as np
import random
import copy
import time


def getAssignmentCost(routes, cost_matrix, depot_id):
    """
    total cost of the GVRP solution (sum of all route distances).
    """
    total_cost = 0
    for route in routes:
        if not route: continue
        # Depot -> First Node
        total_cost += cost_matrix.get((depot_id, route[0]), 1e9)
        # Node -> Node
        for i in range(len(route)-1):
            total_cost += cost_matrix.get((route[i], route[i+1]), 1e9)
        # Last Node -> Depot
        total_cost += cost_matrix.get((route[-1], depot_id), 1e9)

    return total_cost

def check_feasibility(routes, Q, q_cluster, a, M):
    """
    Checks if the GVRP solution is feasible (Capacity + Cluster Constraints).
    """

    # # check vehicle count
    # if K is not None:
    #     num_vehicles = len([r for r in routes if len(r) > 0])
    #     # If your problem is "Exactly K":
    #     if num_vehicles != K:
    #         return False
    #     # If your problem is "At most K":
    #     # if num_vehicles > K:
    #     #    return False

    # # check if edges exist
    # for i in range(len(route)-1):
    #     u, v = route[i], route[i+1]
    #     if (u,v) not in cost_matrix and (u,v) not in arc_list_set:
    #         return False # Impossible path


    # capacity check
    for route in routes:
        load = sum(q_cluster[a[node]] for node in route)
        if load > Q:
            return False

    # all clusters visited exactly once
    visited_clusters = set()
    for route in routes:
        for node in route:
            clust = a[node]
            if clust in visited_clusters:
                return False # twice
            visited_clusters.add(clust)

    if len(visited_clusters) != M:
        return False # some are not visited

    return True

# myopic heurisitc
def myopic_heuristic(N, K, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes):
    print("************************ Running myopic heuristic    ************************\n\n")

    routes = []
    unvisited = set(range(1, M + 1))

    # construction
    for k in range(K):
        if not unvisited: break
        route = []
        curr = depot_id
        load = 0

        while True:
            best_node = None
            best_dist = 1e9
            best_cl = None

            # nearest feasible neighbor
            possible_clusters = list(unvisited)
            for cl in possible_clusters:
                dem = q_cluster[cl]
                if load + dem > Q: continue

                for node in cluster_nodes[cl]:
                    d = cost_matrix.get((curr, node), 1e9)
                    if d < best_dist:
                        best_dist = d
                        best_node = node
                        best_cl = cl

            if best_node is not None:
                route.append(best_node)
                curr = best_node
                load += q_cluster[best_cl]
                unvisited.remove(best_cl)
            else:
                break
        routes.append(route)

    # check status
    status = check_feasibility(routes, Q, q_cluster, a, M)
    total_cost = getAssignmentCost(routes, cost_matrix, depot_id)

    if status:
        print(f"Myopic Status: Feasible, Cost: {total_cost:.2f}")
    else:
        print(f"Myopic Status: Infeasible ({len(unvisited)} clusters unvisited)")

    return status, routes, total_cost

# VNS metaheuristic

def VNS_algorithm(kmax, max_iterations, solution_routes, solution_cost, N, K, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes):
    print("************************ Running Variable Neighborhood Search    ************************\n\n")

    current_sol = copy.deepcopy(solution_routes)
    current_sol_cost = solution_cost

    # Create neighborhoods (for 'k' intensities 1..kmax)
    neighborhoods = create_neighborhoods(kmax)

    iterations = 0
    while iterations < max_iterations:
        k = 0
        while k < kmax:
            k_intensity = neighborhoods[k]

            # shaking
            new_sol, new_sol_cost = shaking(current_sol, k_intensity, N, Q, q_cluster, a, cost_matrix, depot_id)

            # local search
            #  iterate all clusters to re-optimize their positions
            all_clusters_neighborhood = list(range(1, M + 1))

            new_sol, new_sol_cost = local_search_vns(all_clusters_neighborhood, new_sol, N, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes)

            # accpted
            if new_sol_cost < current_sol_cost - 1e-6:
                current_sol_cost = new_sol_cost
                current_sol = new_sol
                print(f"--Improved bound: Iteration {iterations}, k {k_intensity}, Best Evaluation {current_sol_cost:.2f}")
                k = 0
            else:
                k += 1

        if iterations % 10 == 0:
            print(f"Iteration {iterations}, Best Evaluation {current_sol_cost:.2f}")
        iterations += 1

    print(f"Last Iteration {iterations-1}, Best Evaluation {current_sol_cost:.2f}")
    return current_sol, current_sol_cost

def create_neighborhoods(kmax):
    # in our gvrp problem, VNS 'neighborhoods' are levels of shaking intensity
    return list(range(1, kmax + 1))

def shaking(solution, k, N, Q, q_cluster, a, cost_matrix, depot_id):
    """
    Perturbs solution with 'k' random relocate moves.
    """
    temp_routes = copy.deepcopy(solution)
    moves_done = 0
    attempts = 0

    while moves_done < k and attempts < k*10:
        attempts += 1

        # random source
        non_empty = [i for i, r in enumerate(temp_routes) if len(r) > 0]
        if not non_empty: break
        r_idx = random.choice(non_empty)
        node_idx = random.randint(0, len(temp_routes[r_idx])-1)
        val = temp_routes[r_idx][node_idx]

        # random target
        target_r_idx = random.randint(0, len(temp_routes)-1)

        # check for capacity
        demand = q_cluster[a[val]]
        current_load = sum(q_cluster[a[n]] for n in temp_routes[target_r_idx])

        if r_idx != target_r_idx and current_load + demand > Q:
            continue

        # the move
        temp_routes[r_idx].pop(node_idx)
        insert_pos = random.randint(0, len(temp_routes[target_r_idx]))
        temp_routes[target_r_idx].insert(insert_pos, val)
        moves_done += 1

    cost = getAssignmentCost(temp_routes, cost_matrix, depot_id)
    return temp_routes, cost

def local_search_vns(neighborhood, solution, N, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes):
    """
    Iterates through the neighborhood (list of clusters) and tries to find the best
    assignment/position for each cluster using find_best_val.
    """
    new_sol = copy.deepcopy(solution)

    for clust_idx in neighborhood:
        # Pass the current state (new_sol) to find_best_val to improve it for one cluster
        new_sol = find_best_val(clust_idx, new_sol, N, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes)

    sol_cost = getAssignmentCost(new_sol, cost_matrix, depot_id)
    return new_sol, sol_cost

def find_best_val(clust_idx, solution, N, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes):
    """
    Finds the best position (route & index) and best node for the given cluster 'clust_idx'.
    """
    # remove the cluster from its current position
    # use a copy to try all values for this variable
    # get current pos
    current_r_idx = -1
    current_n_idx = -1
    for r_i, route in enumerate(solution):
        for n_i, node in enumerate(route):
            if a[node] == clust_idx:
                current_r_idx = r_i
                current_n_idx = n_i
                break
        if current_r_idx != -1: break

    if current_r_idx == -1: return solution

    # store original node to restore if no better move found
    removed_node = solution[current_r_idx][current_n_idx]

    # create empty slot by removing the node
    temp_sol = copy.deepcopy(solution)
    temp_sol[current_r_idx].pop(current_n_idx)

    best_sol_found = copy.deepcopy(solution) # Default to current state
    min_cost = getAssignmentCost(solution, cost_matrix, depot_id) # Baseline cost

    # iterate ALL possible assignments (route, idx, node_in_clust)
    # equiv is `for val in range(N)` in p-center

    for r_idx in range(len(temp_sol)):
        route = temp_sol[r_idx]

        # check_feasibility inside the loop will handle a forbidden load based ono the capacity
        for i in range(len(route) + 1):
            for candidate_node in cluster_nodes[clust_idx]:

                # insert node in the route
                route.insert(i, candidate_node)

                # check feasibility from gvrp_model constraints
                if check_feasibility(temp_sol, Q, q_cluster, a, M):
                    # Calculate Cost
                    cost = getAssignmentCost(temp_sol, cost_matrix, depot_id)

                    # update if better
                    if cost < min_cost:
                        min_cost = cost
                        best_sol_found = copy.deepcopy(temp_sol)

                # backtrack. remove inserted node -> try next
                route.pop(i)

    return best_sol_found
