import gvrp_model_gurobi as pg
import b_and_b_algorithm as bb
import numpy as np
import heuristics_gvrp as heur
import time
from gurobipy import GRB

# 1. Load Data
filename = "./15_100_12/0.txt"
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
status, init_routes, init_cost = heur.myopic_heuristic(N, K, Q, M, q_cluster, a, cost_matrix, depot_id, cluster_nodes)

heur_obj = np.inf
vns_obj = np.inf
vns_sol_vector = None

if status:
    heur_obj = init_cost
    print(f"Myopic found feasible solution. Cost: {heur_obj:.2f}")

    # B. VNS Metaheuristic
    print("\nStarting VNS improvement...")
    best_routes, best_cost = heur.VNS_algorithm(
        kmax=3,
        max_iterations=50,
        solution_routes=init_routes,
        solution_cost=init_cost,
        N=N, K=K, Q=Q, M=M,
        q_cluster=q_cluster, a=a,
        cost_matrix=cost_matrix,
        depot_id=depot_id,
        cluster_nodes=cluster_nodes
    )

    # CHECK: Does heuristic solution use EXACTLY K vehicles?
    active_vehicles = len([r for r in best_routes if len(r) > 0])

    if active_vehicles == K:
        vns_obj = best_cost
        print(f"VNS completed. Valid solution (K={active_vehicles}). Cost: {vns_obj:.2f}")
    else:
        print(f"WARNING: VNS found solution with {active_vehicles} vehicles, but K={K} required.")
        print(f"Heuristic cost ({best_cost:.2f}) is likely invalid/loose for the exact-K problem.")
        # We assume the user might want to see this cost but we won't treat it as a strict UB for the exact-K model
        # Or we treat it as valid if <= K is allowed.
        # But per request, we flag it.
        vns_obj = best_cost # Still store it to see if B&B beats it

    # Convert VNS routes to solution vector for MIP Start / Comparison
    if active_vehicles <= K: # Can only map to variables if valid count
        vns_sol_vector = np.zeros(num_vars)
        for route in best_routes:
            if not route: continue
            # Depot -> First
            if (depot_id, route[0]) in arc_to_idx:
                vns_sol_vector[arc_to_idx[(depot_id, route[0])]] = 1.0
            # Inter-nodes
            for i in range(len(route)-1):
                if (route[i], route[i+1]) in arc_to_idx:
                    vns_sol_vector[arc_to_idx[(route[i], route[i+1])]] = 1.0
            # Last -> Depot
            if (route[-1], depot_id) in arc_to_idx:
                vns_sol_vector[arc_to_idx[(route[-1], depot_id)]] = 1.0
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
# Only set valid UB if vehicle count matches, otherwise start infinite to avoid pruning valid solutions
if vns_sol_vector is not None and (active_vehicles == K):
    bb.upper_bound = vns_obj
else:
    bb.upper_bound = np.inf

bb.lower_bound = -np.inf
bb.nodes = 0

best_bound_per_depth = np.array([np.inf for _ in range(num_vars + 1)])
nodes_per_depth = np.array([0] * (num_vars + 1), dtype=object)
nodes_per_depth[0] = 1
for i in range(1, num_vars + 1):
    nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

bb_start = time.time()
solutions, best_idx, count = bb.branch_and_bound(
    model_bb, ub, lb, integer_var,
    best_bound_per_depth, nodes_per_depth
)
bb_end = time.time()

bb_obj = np.inf
if count > 0:
    bb_obj = solutions[best_idx][1]
    print(f"Custom B&B Optimal Cost: {bb_obj:.2f}")
elif vns_sol_vector is not None and bb.upper_bound != np.inf:
    bb_obj = vns_obj
    print(f"Custom B&B finished without finding better solution. Using VNS Cost: {bb_obj:.2f}")
else:
    print("Custom B&B found no solution (or heuristic was invalid).")

print(f"Custom B&B Time: {bb_end - bb_start:.4f} s")
print(f"Nodes Explored: {bb.nodes}")


# ==============================================================================
# 3. Run Gurobi Exact (No MIP Start)
# ==============================================================================
print("\n\n************************ Running Gurobi (Exact)    ************************\n")
# Create fresh model for exact run
model_grb, _, _, _, _, _, _, _, _ = pg.gvrp_model_gurobi(filename)

grb_start = time.time()
model_grb.optimize()
grb_end = time.time()

grb_obj = np.inf
if model_grb.status == GRB.OPTIMAL:
    grb_obj = model_grb.ObjVal
    print(f"Gurobi Exact Objective: {grb_obj:.2f}")
else:
    print("Gurobi Exact did not find optimal solution.")
print(f"Gurobi Exact Time: {grb_end - grb_start:.4f} s")


# ==============================================================================
# 4. Run Gurobi with MIP Start (if heuristic feasible)
# ==============================================================================
mip_start_obj = np.inf
if vns_sol_vector is not None:
    print("\n\n************************ Running Gurobi (MIP Start)    ************************\n")
    # Create another fresh model for MIP start run
    model_mip, _, _, _, _, _, _, _, _ = pg.gvrp_model_gurobi(filename)

    # Inject VNS solution as start
    # Note: vars_list order in gvrp_model_gurobi matches indices 0..num_vars-1
    all_vars = model_mip.getVars()

    # Reset all starts first
    for v in all_vars:
        v.Start = 0.0

    # Apply heuristic solution
    for i in range(num_vars):
        if vns_sol_vector[i] > 0.5:
            all_vars[i].Start = 1.0

    mip_start_time = time.time()
    model_mip.optimize()
    mip_end_time = time.time()

    if model_mip.status == GRB.OPTIMAL:
        mip_start_obj = model_mip.ObjVal
        print(f"Gurobi MIP Start Objective: {mip_start_obj:.2f}")
    print(f"Gurobi MIP Start Time: {mip_end_time - mip_start_time:.4f} s")


# ==============================================================================
# 5. Optimality Gaps & Summary
# ==============================================================================
print("\n\n************************ Summary & Gaps    ************************\n")

print(f"{'Method':<25} | {'Cost':<15} | {'Time (s)':<15}")
print("-" * 60)
val_heur = heur_obj if heur_obj != np.inf else "Inf"
val_vns = vns_obj if vns_obj != np.inf else "Inf"
val_bb = bb_obj if bb_obj != np.inf else "Inf"
val_grb = grb_obj if grb_obj != np.inf else "Inf"
val_mip = mip_start_obj if mip_start_obj != np.inf else "-"

print(f"{'Myopic Heuristic':<25} | {str(val_heur):<15} | {'-':<15}")
print(f"{'VNS Metaheuristic':<25} | {str(val_vns):<15} | {heur_end - heur_start:<15.4f}")
print(f"{'Custom B&B':<25} | {str(val_bb):<15} | {bb_end - bb_start:<15.4f}")
print(f"{'Gurobi (Exact)':<25} | {str(val_grb):<15} | {grb_end - grb_start:<15.4f}")
print(f"{'Gurobi (MIP Start)':<25} | {str(val_mip):<15} | {mip_end_time - mip_start_time if mip_start_obj != np.inf else '-':<15}")

print("\n--- Optimality Gaps (relative to Gurobi Exact) ---")
if grb_obj != np.inf and grb_obj != 0:
    if vns_obj != np.inf:
        gap_vns = (vns_obj - grb_obj) / grb_obj * 100
        print(f"VNS Gap: {gap_vns:.2f}%")

    if bb_obj != np.inf:
        gap_bb = (bb_obj - grb_obj) / grb_obj * 100
        print(f"Custom B&B Gap: {gap_bb:.2f}%")
else:
    print("Cannot calculate gaps (Gurobi exact solution invalid/zero).")
