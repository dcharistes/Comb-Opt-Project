import gvrp_model_gurobi as pg
import b_and_b_algorithm as bb  # Assuming your b&b script is named this
import numpy as np

# 1. Load the Model
filename = "./10_35_6/0.txt"
model, ub, lb, integer_var, num_vars, c = pg.gvrp_model_gurobi(filename)

# 2. Initialize B&B Structures
# Note: GVRP is a Minimization problem, so isMax = False
bb.isMax = False

# Initialize tracking arrays
best_bound_per_depth = np.array([np.inf for _ in range(num_vars)])
nodes_per_depth = np.array([0 for _ in range(num_vars)])

print("Starting Branch and Bound for GVRP...")

# 3. Run B&B
solutions, best_idx, count = bb.branch_and_bound(
    model, ub, lb, integer_var,
    best_bound_per_depth, nodes_per_depth
)

# 4. Output
if count > 0:
    best_sol = solutions[best_idx]
    print(f"Optimal Cost: {best_sol[1]}")
    # Decode solution (only print variables > 0.5)
    sol_vector = best_sol[0]
    num_arcs = num_vars // 2
    print("Routes:")
    # We need to access arc list again to print names, or just print indices
    # (For cleaner output, you might want to return 'arc_list' from gvrp_model_gurobi too)
    for i in range(num_arcs):
        if sol_vector[i] > 0.5:
            print(f"  Arc Index {i} is Active (x=1)")
else:
    print("No feasible solution found.")
