import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import math
import heapq

# Set sense (min/max)
isMax = False
WARM_START = True
DEBUG_MODE = True

# Dynamic cuts configuration
ENABLE_DYNAMIC_CUTS = True
MAX_CUTS_PER_NODE = 3

# Total number of nodes visited
nodes = 0

# Lower bound of the problem
lower_bound = -np.inf

# Upper bound of the problems
upper_bound = np.inf

def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# A class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label
        self.id = id(self)

# A simple function to print debugging info
def debug_print(node=None, x_obj=None, sol_status=None):
    print("\n\n----------------- DEBUG OUTPUT -----------------\n\n")
    print(f"Global UB:{upper_bound}")
    print(f"Global LB:{lower_bound}")
    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")
    print("\n\n--------------------------------------------------\n\n")


def separate_capacity_cuts(model, vars_list, num_arcs, arc_list, a, q_cluster, Q, M, epsilon=1e-4):
    """
    Separates capacity/subtour cuts over small cluster subsets.
    Returns a list of violated cuts (gp.LinExpr >= rhs).
    """
    # FIX: Retrieve values safely (vars_list is a tupledict)
    all_vals = model.getAttr('X', vars_list)
    # Access by index since your keys are 0..num_vars-1
    x_val = [all_vals[i] for i in range(num_arcs)]

    # Build cluster-cluster edge weights
    cluster_edge_weight = [[0.0] * (M + 1) for _ in range(M + 1)]
    for idx, (u, v) in enumerate(arc_list):
        ku = a[u]
        kv = a[v]
        if ku == 0 and kv == 0:
            continue
        cluster_edge_weight[ku][kv] += x_val[idx]

    def x_delta_S(S):
        """Sum of x on arcs crossing the cluster set S."""
        Sset = set(S)
        total = 0.0
        for k in Sset:
            for l in range(M + 1):
                if l not in Sset:
                    total += cluster_edge_weight[k][l]
                    total += cluster_edge_weight[l][k]
        return total

    violated = []

    # Single clusters
    for k in range(1, M + 1):
        qS = q_cluster[k]
        rS = math.ceil(qS / Q)
        if rS <= 0:
            continue
        lhs = x_delta_S([k])
        rhs = 2 * rS
        if lhs < rhs - epsilon:
            violated.append(([k], lhs, rhs))

    # Pairs
    for k in range(1, M + 1):
        for l in range(k + 1, M + 1):
            qS = q_cluster[k] + q_cluster[l]
            rS = math.ceil(qS / Q)
            if rS <= 0:
                continue
            lhs = x_delta_S([k, l])
            rhs = 2 * rS
            if lhs < rhs - epsilon:
                violated.append(([k, l], lhs, rhs))

    # Sort by violation (most violated first)
    violated.sort(key=lambda x: x[2] - x[1], reverse=True)

    # Build cut expressions
    cuts = []
    for S, lhs_val, rhs_val in violated[:MAX_CUTS_PER_NODE]:
        Sset = set(S)
        expr = gp.LinExpr()
        for idx, (u, v) in enumerate(arc_list):
            ku = a[u]
            kv = a[v]
            if (ku in Sset and kv not in Sset) or (ku not in Sset and kv in Sset):
                expr += vars_list[idx]
        cuts.append((expr, rhs_val))

    return cuts


# Definition of the branch & bound algorithm.
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth,
                     # GVRP-specific data for cut separation
                     num_arcs=None, arc_list=None, a=None, q_cluster=None, Q=None, M=None, vars_list=None,
                     vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # Initialize Global Objective Bounds correctly as Scalars
    if isMax:
        lower_bound = -np.inf
        upper_bound = np.inf
    else:
        upper_bound = np.inf
        lower_bound = -np.inf

    # Create heap queue structure
    heap = []
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0

    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # Create root node
    # Note: ub and lb here are expected to be LISTS (variable bounds vectors)
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # =============== Root node ==========================
    if DEBUG_MODE:
        debug_print()

    # Solve relaxed problem
    model.optimize()

    # Check if the model was solved to optimality.
    if model.status != GRB.OPTIMAL:
        if DEBUG_MODE: print("Root Infeasible.")
        return [], -1, 0

    # Get the solution
    x_candidate = model.getAttr('X', model.getVars())
    x_obj = model.ObjVal

    # Check integer variables
    vars_have_integer_vals = True
    min_dist = 10
    selected_var_idx = -1
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            # Pick variable closest to 0.5 for branching
            if abs(x_candidate[idx] - 0.5) < min_dist:
                min_dist = abs(x_candidate[idx] - 0.5)
                selected_var_idx = idx

    # If root is integer, we are done
    if vars_have_integer_vals:
        solutions.append([x_candidate, x_obj, depth])
        if DEBUG_MODE: debug_print(node=root_node, x_obj=x_obj, sol_status="Integer at Root")
        return solutions, 0, 1

    # Otherwise update bounds
    if isMax:
        # For Max, root relaxation is the global Upper Bound
        pass 
    else:
        # For Min, root relaxation is the global Lower Bound
        pass

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Warm start info
    if WARM_START:
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

    # Update variable bounds for branching
    var_lbs = model.getAttr("LB", model.getVars())
    var_ubs = model.getAttr("UB", model.getVars())
    
    left_var_lb = np.array(var_lbs)
    left_var_ub = np.array(var_ubs)
    right_var_lb = np.array(var_lbs)
    right_var_ub = np.array(var_ubs)

    left_var_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_var_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    left_child = Node(left_var_ub, left_var_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_var_ub, right_var_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # PUSH TO HEAP
    if isMax:
        heapq.heappush(heap, (-x_obj, right_child.id, right_child))
        heapq.heappush(heap, (-x_obj, left_child.id, left_child))
    else:
        heapq.heappush(heap, (x_obj, right_child.id, right_child))
        heapq.heappush(heap, (x_obj, left_child.id, left_child))

    # ================= Main Loop =================
    while len(heap) != 0:
        
        # Increment total nodes
        nodes += 1

        # Pop best node
        parent_obj_bound, _, current_node = heapq.heappop(heap)
        
        # Determine the bound of the current node from the heap key
        current_node_bound = parent_obj_bound if not isMax else -parent_obj_bound
        
        # Check Global Convergence
        if isMax:
            # If best potential (UB of node) <= Global LB (Best Integer), prune/stop
            if current_node_bound <= lower_bound + 1e-6:
                if DEBUG_MODE: print("Global Bound Convergence (All remaining nodes worse than Incumbent). Stopping.")
                break
        else:
            # If best potential (LB of node) >= Global UB (Best Integer), prune/stop
            if current_node_bound >= upper_bound - 1e-6:
                if DEBUG_MODE: print("Global Bound Convergence (All remaining nodes worse than Incumbent). Stopping.")
                break

        print(f"\n*** Node {nodes} | Depth {current_node.depth} | Bound {current_node_bound:.2f} ***")

        # Load Node State
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
             constrs = model.getConstrs()
             if len(constrs) == len(current_node.cbasis):
                 model.setAttr("CBasis", constrs, current_node.cbasis)
             model.setAttr("VBasis", model.getVars(), current_node.vbasis)

        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        model.optimize()

        # --- Infeasibility Handling ---
        if model.status != GRB.OPTIMAL:
            continue

        x_candidate = model.getAttr('X', model.getVars())
        x_obj = model.ObjVal

        # --- Dynamic Cuts Loop ---
        if ENABLE_DYNAMIC_CUTS and current_node.depth > 0:
             if num_arcs is not None and arc_list is not None:
                cuts = separate_capacity_cuts(model, vars_list, num_arcs, arc_list, a, q_cluster, Q, M)
                if len(cuts) > 0:
                    print(f"  [Depth {current_node.depth}] Adding {len(cuts)} cuts...")
                    for expr, rhs in cuts:
                        model.addLConstr(expr >= rhs)
                    model.update()
                    model.optimize()
                    
                    if model.status != GRB.OPTIMAL:
                        continue 
                    
                    x_candidate = model.getAttr('X', model.getVars())
                    x_obj = model.ObjVal

        # --- Pruning by Bound ---
        if isMax:
            if x_obj <= lower_bound + 1e-6:
                if DEBUG_MODE: print("  Pruned by Bound (Maximization)")
                continue
        else:
            if x_obj >= upper_bound - 1e-6:
                if DEBUG_MODE: print("  Pruned by Bound (Minimization)")
                continue

        # --- Integer Check ---
        vars_have_integer_vals = True
        selected_var_idx = -1
        min_dist = 10
        
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                dist = abs(x_candidate[idx] - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    selected_var_idx = idx

        # --- Feasible Integer Solution Found ---
        if vars_have_integer_vals:
            print(f"  INTEGER SOLUTION FOUND: {x_obj}")
            solutions.append([x_candidate, x_obj, current_node.depth])
            solutions_found += 1
            
            if isMax:
                if x_obj > lower_bound:
                    lower_bound = x_obj
                    best_sol_obj = x_obj
                    best_sol_idx = solutions_found - 1
            else:
                if x_obj < upper_bound:
                    upper_bound = x_obj
                    best_sol_obj = x_obj
                    best_sol_idx = solutions_found - 1
            
            continue 

        # --- Branching ---
        if WARM_START:
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        left_var_lb = np.copy(current_node.lb)
        left_var_ub = np.copy(current_node.ub)
        right_var_lb = np.copy(current_node.lb)
        right_var_ub = np.copy(current_node.ub)

        left_var_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_var_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        left_child = Node(left_var_ub, left_var_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_var_ub, right_var_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        if isMax:
            heapq.heappush(heap, (-x_obj, right_child.id, right_child))
            heapq.heappush(heap, (-x_obj, left_child.id, left_child))
        else:
            heapq.heappush(heap, (x_obj, right_child.id, right_child))
            heapq.heappush(heap, (x_obj, left_child.id, left_child))

    return solutions, best_sol_idx, solutions_found