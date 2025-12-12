import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque

# ==========================================
# PART 1: Data Reading (Same as before)
# ==========================================

def read_data_gvrp(filename="gvrp_instance.txt"):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # parse header line
    header = lines[0].split()
    grid_total_nodes = int(header[0])  # grid_size * grid_size
    V = int(header[1])                 # number of nodes
    M = int(header[2])                 # number of clusters
    K = int(header[3])                 # number of vehicles
    Q = float(header[4])               # vehicle capacity

    # parse clusters
    line_idx = 2  # Skip header line + line with 'M Clusters:'
    cluster_nodes = {}
    a = []  # node-to-cluster mapping, sequential indexing will be used
    q_cluster = [0] * (M + 1)
    node_coords = []

    depot_node_id = None
    node_id_map = {}  # map (x,y) to sequential node ID, not grid id

    current_node_id = 0
    while "Arcs:" not in lines[line_idx]:
        parts = lines[line_idx].split()
        cluster_id = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        demand = float(parts[3])

        node_coords.append((x, y))

        node_id_map[(x, y)] = current_node_id
        a.append(cluster_id)

        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(current_node_id)

        # assign cluster demand (all nodes within a cluster have same demand)
        q_cluster[cluster_id] = demand

        if cluster_id == 0:
            depot_node_id = current_node_id

        current_node_id += 1
        line_idx += 1

    N = current_node_id  # number of nodes (depot + customers)

    # arcs
    line_idx += 1  # skip 'Arcs:' line
    arc_list = []
    cost_param = {}
    while line_idx < len(lines):
        parts = lines[line_idx].split()
        x1, y1, x2, y2 = map(int, parts[0:4])
        distance = int(parts[4])

        node_i = node_id_map[(x1, y1)]
        node_j = node_id_map[(x2, y2)]

        arc_list.append((node_i, node_j))
        cost_param[(node_i, node_j)] = distance

        line_idx += 1


    print(f"Successfully read GVRP instance:")
    print(f"  N (total nodes): {N}")
    print(f"  K (vehicles): {K}")
    print(f"  Q (capacity): {Q}")
    print(f"  M (clusters): {M}")
    print(f"  Cluster demands: {q_cluster}")
    print(f"  Nodes per cluster: {cluster_nodes}")
    # Return parsed data
    return N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id, cluster_nodes

# ==========================================
# PART 2: Model Building (Gurobi Native)
# ==========================================

def create_gvrp_model_gurobi(filename):
    # 1. Read Data
    N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id, cluster_nodes = read_data_gvrp(filename)
    
    # 2. Initialize Model
    model = gp.Model("GVRP_BnB")
    model.Params.OutputFlag = 0  # Silence Gurobi's internal output to see our B&B logs clearly
    model.Params.Method = 1      # Dual Simplex (good for B&B)

    # 3. Create Variables
    # We use a dictionary to store vars for easy constraint building, 
    # but Gurobi maintains an internal list we will use for B&B.
    
    x = {} # Binary variables (decision)
    f = {} # Continuous variables (flow)
    
    for i, j in arc_list:
        # x_ij: We create them as CONTINUOUS [0,1] initially. 
        # The Branch & Bound algorithm will force them to integers.
        x[i,j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
        
        # f_ij: Flow amount
        # Upper bound is Q (capacity). 
        # We can tighten this: f_ij <= Q - demand_j (if visiting j)
        f[i,j] = model.addVar(lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}")

    model.update()

    # 4. Constraints

    # (1) Visit In: Each customer cluster entered exactly once
    for k in range(1, M + 1): # Skip depot (0)
        # Sum of x_ji for all i in Cluster k, from all j in V
        model.addConstr(
            gp.quicksum(x[j, i] for i in cluster_nodes[k] for j in range(N) if (j, i) in x) == 1,
            name=f"visit_in_{k}"
        )

    # (2) Visit Out: Each customer cluster left exactly once
    for k in range(1, M + 1):
        model.addConstr(
            gp.quicksum(x[i, j] for i in cluster_nodes[k] for j in range(N) if (i, j) in x) == 1,
            name=f"visit_out_{k}"
        )

    # (3) Depot Flow: K vehicles leave, K return
    # Leave Depot
    model.addConstr(
        gp.quicksum(x[depot_node_id, j] for j in range(N) if (depot_node_id, j) in x) == K,
        name="depot_start"
    )
    # Return to Depot
    model.addConstr(
        gp.quicksum(x[j, depot_node_id] for j in range(N) if (j, depot_node_id) in x) == K,
        name="depot_end"
    )

    # (4) Flow Conservation (Node continuity)
    for i in range(N):
        sum_x_out = gp.quicksum(x[i, j] for j in range(N) if (i, j) in x)
        sum_x_in  = gp.quicksum(x[j, i] for j in range(N) if (j, i) in x)
        model.addConstr(sum_x_out == sum_x_in, name=f"flow_con_{i}")

    # (5) Commodity Flow Balance
    # If i is depot: Flow_Out - Flow_In = -Total_Demand
    total_demand = sum(q_cluster)
    
    f_out_depot = gp.quicksum(f[depot_node_id, j] for j in range(N) if (depot_node_id, j) in f)
    f_in_depot  = gp.quicksum(f[j, depot_node_id] for j in range(N) if (j, depot_node_id) in f)
    model.addConstr(f_out_depot - f_in_depot == -total_demand, name="depot_commodity")

    # If i is customer: Flow_Out - Flow_In = Demand_Cluster_i
    # Note: Logic assumes x_in + x_out = 2 for visited nodes
    for i in range(N):
        if i == depot_node_id: continue
        
        f_out = gp.quicksum(f[i, j] for j in range(N) if (i, j) in f)
        f_in  = gp.quicksum(f[j, i] for j in range(N) if (j, i) in f)
        
        x_sum = gp.quicksum(x[j, i] for j in range(N) if (j, i) in x) + \
                gp.quicksum(x[i, j] for j in range(N) if (i, j) in x)

        cid = a[i]
        dem = q_cluster[cid]
        
        # Simplified: Flow_Out - Flow_In = Demand * (x_in_i) -> Since x_in is 1 if visited
        # We use the formulation from your Pyomo code: 0.5 * dem * (x_in + x_out)
        model.addConstr(f_out - f_in == 0.5 * dem * x_sum, name=f"demand_sat_{i}")

    # (6) Capacity Constraints
    for i, j in arc_list:
        # Lower bound: Flow >= Demand_i * x_ij
        dem_i = q_cluster[a[i]]
        model.addConstr(f[i, j] >= dem_i * x[i, j], name=f"cap_low_{i}_{j}")
        
        # Upper bound: Flow <= (Q - Demand_j) * x_ij
        dem_j = q_cluster[a[j]]
        model.addConstr(f[i, j] <= (Q - dem_j) * x[i, j], name=f"cap_up_{i}_{j}")

    # 5. Objective: Minimize Distance
    obj_expr = gp.quicksum(cost_param[i,j] * x[i,j] for i, j in arc_list)
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    model.update()

    # 6. Extract Structures for B&B
    # We need linear lists of lb, ub, and boolean flags for integer variables
    
    all_vars = model.getVars()
    num_vars = len(all_vars)
    
    ub = np.array([v.UB for v in all_vars])
    lb = np.array([v.LB for v in all_vars])
    
    # Identify which variables MUST be Integer. 
    # In GVRP, only 'x' (binary) need to be integers. 'f' are continuous.
    integer_var = []
    for v in all_vars:
        if v.VarName.startswith("x_"):
            integer_var.append(True)
        else:
            integer_var.append(False)
            
    return model, ub, lb, integer_var, num_vars


# ==========================================
# PART 3: Branch and Bound Logic
# ==========================================

# Global variables for B&B state
isMax = False # GVRP is Minimization
WARM_START = True
DEBUG_MODE = True
nodes = 0
lower_bound = -np.inf if isMax else -np.inf # For Min, global lower bound isn't strictly needed for logic, but global UPPER bound is the incumbent
upper_bound = np.inf # The best solution found so far (Primal Bound)

def is_nearly_integer(value, tolerance=1e-5):
    return abs(value - round(value)) <= tolerance

class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

def debug_print(node:Node = None, x_obj = None, sol_status = None):
    global upper_bound
    print(f"\n--- DEBUG: Node {nodes} ---")
    print(f"Global Best (Incumbent): {upper_bound}")
    if node:
        print(f"Depth: {node.depth} | Type: {node.label}")
        if node.branching_var != -1: print(f"Branched on Var Index: {node.branching_var}")
    if x_obj is not None:
        print(f"Current Relaxed Obj: {x_obj:.4f}")
    if sol_status:
        print(f"Status: {sol_status}")
    print("--------------------------\n")

def branch_and_bound(model, initial_ub, initial_lb, integer_var, nodes_per_depth):
    global nodes, lower_bound, upper_bound
    
    stack = deque()
    solutions = []
    solutions_found = 0
    best_sol_idx = -1
    
    # Root Node
    root_node = Node(initial_ub, initial_lb, 0, [], [], -1, "root")
    
    # Solve Root
    model.setAttr("LB", model.getVars(), root_node.lb)
    model.setAttr("UB", model.getVars(), root_node.ub)
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print("Root relaxation Infeasible!")
        return [], -1, 0
        
    x_candidate = model.getAttr('X', model.getVars())
    x_obj = model.ObjVal
    
    # Check Integrality at Root
    vars_have_integer_vals = True
    selected_var_idx = -1
    
    # Heuristic: Pick the variable closest to 0.5 to branch on (most fractional)
    max_fractionality = -1
    
    for idx, is_int in enumerate(integer_var):
        if is_int:
            val = x_candidate[idx]
            if not is_nearly_integer(val):
                vars_have_integer_vals = False
                # Calculate how close to 0.5
                frac = abs(val - 0.5)
                # We want the one closest to 0.5, so smallest 'frac'
                # Or just simple: find first non-integer
                selected_var_idx = idx 
                break 

    if vars_have_integer_vals:
        print("Root is Integer Optimal!")
        upper_bound = x_obj
        solutions.append([x_candidate, x_obj, 0])
        return solutions, 0, 1
    else:
        # Initialize Bound
        if not isMax:
            pass # For min, root obj is a lower bound estimate
            
    # Prepare Root Children
    if WARM_START:
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())
    else:
        vbasis, cbasis = [], []

    # Branching logic (Binary vars 0 or 1)
    left_ub = np.copy(root_node.ub)
    left_ub[selected_var_idx] = 0.0 # Force 0
    
    right_lb = np.copy(root_node.lb)
    right_lb[selected_var_idx] = 1.0 # Force 1
    
    left_child = Node(left_ub, np.copy(root_node.lb), 1, list(vbasis), list(cbasis), selected_var_idx, "Left (x=0)")
    right_child = Node(np.copy(root_node.ub), right_lb, 1, list(vbasis), list(cbasis), selected_var_idx, "Right (x=1)")
    
    stack.append(right_child)
    stack.append(left_child)
    
    # --- Main Loop ---
    while stack:
        current_node = stack.pop()
        nodes += 1
        
        # Expand arrays if needed (for tracking depth stats)
        if current_node.depth >= len(nodes_per_depth):
            nodes_per_depth = np.append(nodes_per_depth, [0], axis=0)
        nodes_per_depth[current_node.depth] += 1
        
        # 1. Load State
        if WARM_START and current_node.vbasis:
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)
            
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()
        
        # 2. Optimize
        model.optimize()
        
        # 3. Pruning Conditions
        
        # A. Infeasibility
        if model.status != GRB.OPTIMAL:
            if DEBUG_MODE: debug_print(current_node, sol_status="Infeasible")
            continue
            
        x_obj = model.ObjVal
        x_candidate = model.getAttr('X', model.getVars())
        
        # B. Bound Pruning (Minimization)
        # If relaxed_obj >= best_integer_found_so_far, this branch is useless
        if not isMax:
            if x_obj >= upper_bound:
                if DEBUG_MODE: debug_print(current_node, x_obj, sol_status="Pruned by Bound")
                continue
        
        # 4. Check Integrality
        vars_have_integer_vals = True
        selected_var_idx = -1
        
        for idx, is_int in enumerate(integer_var):
            if is_int and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break
                
        # 5. Integer Solution Found
        if vars_have_integer_vals:
            if DEBUG_MODE: debug_print(current_node, x_obj, sol_status="Integer Feasible")
            
            # Update Best Solution (Minimization)
            if x_obj < upper_bound:
                upper_bound = x_obj
                solutions.append([x_candidate, x_obj, current_node.depth])
                solutions_found += 1
                best_sol_idx = len(solutions) - 1
                print(f"*** New Incumbent Found: {upper_bound} ***")
            continue # Don't branch further from an integer node
            
        # 6. Branching (Fractional Solution)
        if DEBUG_MODE: debug_print(current_node, x_obj, sol_status="Fractional")
        
        if WARM_START:
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())
        
        # Create Children
        # Left: x <= floor(val) -> x=0 for binary
        l_ub = np.copy(current_node.ub)
        l_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        l_lb = np.copy(current_node.lb) # Unchanged
        
        # Right: x >= ceil(val) -> x=1 for binary
        r_lb = np.copy(current_node.lb)
        r_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])
        r_ub = np.copy(current_node.ub) # Unchanged
        
        left_child = Node(l_ub, l_lb, current_node.depth + 1, list(vbasis), list(cbasis), selected_var_idx, "Left")
        right_child = Node(r_ub, r_lb, current_node.depth + 1, list(vbasis), list(cbasis), selected_var_idx, "Right")
        
        stack.append(right_child)
        stack.append(left_child)
        
    return solutions, best_sol_idx, solutions_found

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # IMPORTANT: Change this path to your actual .txt file generated by Program 1
    instance_filename = "0.txt" # Example path
    
    try:
        print("--- Building GVRP Model (Gurobi) ---")
        model, ub, lb, integer_var, num_vars = create_gvrp_model_gurobi(instance_filename)
        
        print(f"Model created. Total Vars: {num_vars}")
        
        # Stats arrays
        nodes_per_depth = np.array([0])
        best_bound_per_depth = np.array([]) # Not strictly used in this simplified version
        
        print("--- Starting Branch and Bound ---")
        start = time.time()
        
        sols, best_idx, count = branch_and_bound(model, ub, lb, integer_var, nodes_per_depth)
        
        end = time.time()
        
        print("\n\n==========================================")
        print("           OPTIMIZATION FINISHED          ")
        print("==========================================")
        print(f"Total Nodes Visited: {nodes}")
        print(f"Time Elapsed: {end - start:.4f} sec")
        print(f"Solutions Found: {count}")
        
        if best_idx != -1:
            best_sol = sols[best_idx]
            obj_val = best_sol[1]
            print(f"Optimal Objective: {obj_val}")
            
            # Decode the solution to show arcs
            all_vars = model.getVars()
            sol_vals = best_sol[0]
            
            print("\n--- Route (Active Arcs) ---")
            for i, v in enumerate(all_vars):
                if v.VarName.startswith("x_") and sol_vals[i] > 0.5:
                    print(f"{v.VarName} = 1.0")
        else:
            print("No feasible solution found.")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{instance_filename}'. Make sure you run Program 1 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()