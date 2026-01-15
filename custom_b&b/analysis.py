import gvrp_model_gurobi as pg
import b_and_b_cuts as bb
import b_and_b_algorithm as bb_v1
import numpy as np
import heuristics as hr
import time

# 1. Load Data
filename = "./15_90_7/0.txt"

# Load model for Custom B&B
model_bb, ub, lb, integer_var, num_vars, c, arc_list, arc_to_idx, vars_list = pg.gvrp_model_gurobi(filename)

# Load raw data for Heuristics
N, K, Q, M, q_cluster, a, _, cost_param, depot_id, cluster_nodes = pg.read_data_gvrp(filename)

cost_matrix = {}
for (u, v) in arc_list: cost_matrix[(u, v)] = cost_param[(u, v)]

print("\n\n")

# ==============================================================================
# 1. Run Heuristics (Myopic -> VNS)
# ==============================================================================
print("************************ Running Heuristics    ************************\n")
heur_start = time.time()

# A. Myopic Heuristic
status, init_routes, init_cost = hr.myopic_heuristic(N, K, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes)

heur_obj = np.inf
vns_obj = np.inf
vns_sol_vector = None

if status:
    heur_obj = init_cost
    print(f"Myopic found feasible solution. Cost: {heur_obj:.2f}")

    # B. VNS Metaheuristic
    print("\nStarting VNS improvement...")
    best_routes, best_cost = hr.VNS_algorithm(
        kmax=4,
        max_iterations=100,
        solution_routes=init_routes,
        solution_cost=init_cost,
        N=N, K=K, Q=Q, M=M,
        q_cluster=q_cluster, a=a,
        cost_matrix=cost_matrix,
        depot_id=depot_id,
        cluster_nodes=cluster_nodes
    )

    # Removed strict K-vehicle check per request.
    # Using VNS result as valid Upper Bound regardless of vehicle count.
    vns_obj = best_cost
    vns_vehicles = len([r for r in best_routes if len(r) > 0])
    print(f"VNS completed. Cost: {vns_obj:.2f}")
    print(f"VNS Vehicles Used: {vns_vehicles}/{K}")
else:
    print("Heuristics failed to find a feasible solution.")

heur_end = time.time()
print(f"Heuristics Time: {heur_end - heur_start:.4f} s")


# ==============================================================================
# 2. Run Custom Branch and Bound
# ==============================================================================
print("\n\n************************ Running Custom Branch & Bound    ************************\n")

# Initialize bounds
bb.isMax = False

# Use VNS bound if found, otherwise infinite
if vns_obj != np.inf:
    bb.upper_bound = vns_obj
else:
    bb.upper_bound = np.inf

bb.lower_bound = -np.inf
bb.nodes = 0

best_bound_per_depth = np.array([np.inf for _ in range(num_vars + 1)])
nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
nodes_per_depth[0] = 1
for i in range(1, num_vars + 1):
    nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

bb_start = time.time()
solutions, best_idx, count = bb.branch_and_bound(
    model_bb, ub, lb, integer_var,
    best_bound_per_depth, nodes_per_depth,
    # NEW: pass GVRP data for cuts
    num_arcs=len(arc_list),
    arc_list=arc_list,
    a=a,
    q_cluster=q_cluster,
    Q=Q,
    M=M,
    vars_list=vars_list
)
bb_end = time.time()

bb_obj = np.inf
final_solution = None

if count > 0:
    bb_obj = solutions[best_idx][1]
    final_solution = solutions[best_idx][0]
    print(f"Custom B&B Optimal Cost: {bb_obj:.2f}")
else:
    print("Custom B&B found no solution.")

print(f"Custom B&B Time: {bb_end - bb_start:.4f} s")
print(f"Nodes Explored: {bb.nodes}")

# Print Active Routes
if final_solution is not None:
    print("\nActive Routes in Final Solution:")
    num_arcs = num_vars // 2
    for i in range(num_arcs):
        if final_solution[i] > 0.5:
            u, v = arc_list[i]
            if count > 0:
                flow_val = final_solution[num_arcs + i]
                print(f"  Arc {u}->{v}: Flow={flow_val:.2f}")
            else:
                print(f"  Arc {u}->{v}")

print(f"Objective Value: {solutions[best_idx][1]}")
print(f"Tree depth: {solutions[best_idx][2]}")


# ==============================================================================
# 3. Summary
# ==============================================================================
print("\n\n************************ Summary    ************************\n")

print(f"{'Method':<25} | {'Cost':<15} | {'Time (s)':<15}")
print("-" * 60)
val_heur = f"{heur_obj:.2f}" if heur_obj != np.inf else "Inf"
val_vns = f"{vns_obj:.2f}" if vns_obj != np.inf else "Inf"
val_bb = f"{bb_obj:.2f}" if bb_obj != np.inf else "Inf"

print(f"{'Myopic Heuristic':<25} | {val_heur:<15} | {'-':<15}")
print(f"{'VNS Metaheuristic':<25} | {val_vns:<15} | {heur_end - heur_start:<15.4f}")
print(f"{'Custom B&B':<25} | {val_bb:<15} | {bb_end - bb_start:<15.4f}")
