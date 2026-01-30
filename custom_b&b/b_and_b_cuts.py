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
    print(f"UB:{upper_bound:.1f}")
    print(f"LB:{lower_bound}")
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

    # Separates capacity cuts over small cluster subsets.
    # returns a list of violated cuts (they are then added as linear constraints in gp.LinExpr >= rhs).
    # retrieve values (vars_list is a tupledict)
    all_vals = model.getAttr('X', vars_list)
    # access by index since keys are 0..num_vars-1
    x_val = [all_vals[i] for i in range(num_arcs)]

    # cluster-cluster edge weights
    cluster_edge_vehicles = [[0.0] * (M + 1) for _ in range(M + 1)]
    for idx, (u, v) in enumerate(arc_list):
        ku = a[u]
        kv = a[v]
        if ku == 0 and kv == 0:
            continue
        cluster_edge_vehicles[ku][kv] += x_val[idx]

    def x_delta_S(S):
        # sum of x on arcs crossing the cluster set S.
        Sset = set(S)
        total = 0.0
        for k in Sset:
            for l in range(M + 1):
                if l not in Sset:
                    total += cluster_edge_vehicles[k][l]
                    total += cluster_edge_vehicles[l][k]
        return total

    violated = []

    # single clusters
    for k in range(1, M + 1):
        qS = q_cluster[k]
        rS = math.ceil(qS / Q)
        if rS <= 0:
            continue
        lhs = x_delta_S([k])
        rhs = 2 * rS
        if lhs < rhs - epsilon:
            violated.append(([k], lhs, rhs))

    # pairs
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

    # sort by violation (most violated first)
    violated.sort(key=lambda x: x[2] - x[1], reverse=True)

    # build cut expressions
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


# def branch & bound .
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth,
                     # GVRP-specific data for cut separation
                     num_arcs=None, arc_list=None, a=None, q_cluster=None, Q=None, M=None, vars_list=None,
                     vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # Initialize bounds ONLY if they haven't been set by heuristics
    if isMax:
        lower_bound = -np.inf
        if upper_bound == np.inf: # Only reset if not already set
             upper_bound = np.inf
    else:
        lower_bound = -np.inf
        # Only reset to Infinity if the user didn't provide a better start (like VNS)
        if upper_bound is None:
             upper_bound = np.inf

    # create stack using deque() structure
    pq = []
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0


    # create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    if DEBUG_MODE:
        debug_print()

    # relaxed problem bound
    model.optimize()

    # check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # Get the objective value
    x_obj = model.ObjVal

    # Check if all variables have integer values (from the ones that are supposed to be integers)
    vars_have_integer_vals = True
    selected_var_idx = -1
    min_dist = 1
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var:
            val = x_candidate[idx]
            if not is_nearly_integer(val):
                vars_have_integer_vals = False
                dist = abs(val - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    selected_var_idx = idx

    # Found feasible solution.
    if vars_have_integer_vals:
        # If we have feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1
        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found


    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Warm start simplex
    if WARM_START:
        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

    # Create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # Create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Create child nodes
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Add child nodes in stack
    if isMax:
        heapq.heappush(pq, (-x_obj, right_child.id, right_child))
        heapq.heappush(pq, (-x_obj, left_child.id, left_child))
    else:
        heapq.heappush(pq, (x_obj, right_child.id, right_child))
        heapq.heappush(pq, (x_obj, left_child.id, left_child))

    # Solving sub problems
    # While the stack has nodes, continue solving
    while(len(pq) != 0):
        print("\n******************************** NEW NODE BEING EXPLORED ******************************** ")

        # Increment total nodes by 1
        nodes += 1

        # Get the child node on top of stack
        parent_obj_value, _, current_node = heapq.heappop(pq)


        # The node  popped is the "best" remaining node in the tree.
        # Therefore, its bound is the global bound for the remaining search space.
        if isMax:
            upper_bound = parent_obj_value
            # TERMINATION CHECK:
            # If the best potential (upper_bound) is worse than the best integer solution found (lower_bound), stop.
            if upper_bound <= lower_bound + 1e-6:
                if DEBUG_MODE: print("Global Upper Bound hit Incumbent. Search Complete.")
                return solutions, best_sol_idx, solutions_found
        else:
            lower_bound = parent_obj_value
            # TERMINATION CHECK:
            # If the best potential (lower_bound) is worse than the best integer solution found (upper_bound), stop.
            if lower_bound >= upper_bound - (1-1e-6):
                if DEBUG_MODE: print("Global Lower Bound hit Incumbent. Search Complete.")
                return solutions, best_sol_idx, solutions_found

        print(f"\n*** EXPLORING NODE | Depth: {current_node.depth} | Bound: {lower_bound:.4f} ***")

        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            # only load CBasis if constraint counts match (avoids crash when cuts are added)
            constrs = model.getConstrs()
            if len(constrs) == len(current_node.cbasis):
                model.setAttr("CBasis", constrs, current_node.cbasis)

            # VBasis is usually safe unless variable count changes
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)

        # Update the state of the model, passing the new lower bounds/upper bounds for the vars.
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # Optimize the model
        model.optimize()

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue
        else:
            # Get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # Get the objective value
            x_obj = model.ObjVal

            # DYNAMIC CAPACITY CUTS
            if ENABLE_DYNAMIC_CUTS and current_node.depth > 0 and len(pq) > 0:
                if num_arcs is not None and arc_list is not None:
                    cuts = separate_capacity_cuts(
                        model, vars_list, num_arcs, arc_list, a, q_cluster, Q, M
                    )
                    if len(cuts) > 0:
                        print(f"  [Depth {current_node.depth}] Adding {len(cuts)} capacity cuts...")
                        for expr, rhs in cuts:
                            model.addLConstr(expr >= rhs)
                        model.update()
                        model.optimize()

                        # Update solution after re-solve
                        if model.status == GRB.OPTIMAL:
                            x_candidate = model.getAttr('X', model.getVars())
                            x_obj = model.ObjVal
                        else:
                            # Cuts made problem infeasible
                            if DEBUG_MODE:
                                debug_print(node=current_node, sol_status="Infeasible after cuts")
                            continue
            # ===========================

            # update best bound per depth if a better solution was found
            if isMax:
                if x_obj <= lower_bound + 1e-6:
                    if DEBUG_MODE: print("  Pruned by Bound (Maximization)")
                    continue
            else:
                if x_obj >= upper_bound - 1e-6:
                    if DEBUG_MODE: print("  Pruned by Bound (Minimization)")
                    continue

        # If infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        selected_var_idx = -1
        min_dist = 1
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                dist = abs(x_candidate[idx] - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    selected_var_idx = idx

        # Found feasible solution.
        if vars_have_integer_vals:
            if isMax:
                if x_obj > lower_bound:
                    lower_bound = x_obj # Update Best Integer Sol
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    best_sol_idx = solutions_found - 1 # Use latest
                    if DEBUG_MODE: debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (New Incumbent)")

            else:
                if x_obj < upper_bound:
                    upper_bound = x_obj # Update Best Integer Sol
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    best_sol_idx = solutions_found - 1 # Use latest
                    if DEBUG_MODE: debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (New Incumbent)")
            continue

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Warm start simplex
        if WARM_START:
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        # Create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # Create left and right branches
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        if isMax:
            heapq.heappush(pq, (-x_obj, right_child.id, right_child))
            heapq.heappush(pq, (-x_obj, left_child.id, left_child))
        else:
            heapq.heappush(pq, (x_obj, right_child.id, right_child))
            heapq.heappush(pq, (x_obj, left_child.id, left_child))

    return solutions, best_sol_idx, solutions_found
