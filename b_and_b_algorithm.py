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
    print(f"UB:{upper_bound}")
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

def check_depth_completion(depth, nodes_per_depth, best_bound_per_depth, lb, ub, isMax, DEBUG_MODE):
    stop = False
    # If all nodes in the current depth have been visited
    if nodes_per_depth[depth] == 0:
        if isMax:
            # Update Global Upper Bound
            ub = best_bound_per_depth[depth]
            if ub <= lb + 1e-6:  # Check termination
                if DEBUG_MODE: print("Global UB hit LB (Level Completed). Stopping.")
                stop = True
        else:
            # Update Global Lower Bound
            lb = best_bound_per_depth[depth]
            if lb >= ub - 1e-6:  # Check termination
                if DEBUG_MODE: print("Global LB hit UB (Level Completed). Stopping.")
                stop = True
    return stop, lb, ub


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

    # We force conversion to Python INT (arbitrary precision) to avoid OverflowError
    if not isinstance(nodes_per_depth[0], (int, np.integer)) or nodes_per_depth[0] == 0:
        print("WARNING: Re-initializing nodes_per_depth as arbitrary-precision integers.")
        nodes_per_depth = np.array([0] * len(nodes_per_depth), dtype=object)
        nodes_per_depth[0] = 1
        for i in range(1, len(nodes_per_depth)):
            nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

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
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # =============== Root node ==========================
    if DEBUG_MODE:
        debug_print()

    # Solve relaxed problem
    model.optimize()

    # Check if the model was solved to optimality. If not then return (infeasible).
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
    min=10
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            if abs(x_candidate[idx] - 0.5) < min:
                min = abs(x_candidate[idx] - 0.5)
                selected_var_idx = idx

    # Found feasible solution.
    if vars_have_integer_vals:
        # If we have feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1
        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    # Otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj
        else:
            lower_bound = x_obj
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

    # Add child nodes in heap queue
    if isMax:
        heapq.heappush(heap, (-np.inf, right_child.id, right_child))
        heapq.heappush(heap, (-np.inf, left_child.id, left_child))
    else:
        heapq.heappush(heap, (np.inf, right_child.id, right_child))
        heapq.heappush(heap, (np.inf, left_child.id, left_child))

    # Solving sub problems
    # While the heap queue has nodes, continue solving
    while(len(heap) != 0):
        print("\n******************************** NEW NODE BEING EXPLORED ******************************** ")

        # Increment total nodes by 1
        nodes += 1

        _, _, current_node = heapq.heappop(heap)

        # Increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1

        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            # FIX: Only load CBasis if constraint counts match (avoids crash when cuts are added)
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
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 ** (i - current_node.depth)

            # if we reached the final node of a depth, then update the bounds
            stop, lower_bound, upper_bound = check_depth_completion(
                current_node.depth, nodes_per_depth, best_bound_per_depth,
                lower_bound, upper_bound, isMax, DEBUG_MODE
            )
            if stop:
                return solutions, best_sol_idx, solutions_found

        else:
            # Get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # Get the objective value
            x_obj = model.ObjVal

            # ===========================
            # DYNAMIC CAPACITY CUTS
            # ===========================
            if ENABLE_DYNAMIC_CUTS and current_node.depth > 0 and len(heap) > 0:
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
                            infeasible = True
                            if isMax:
                                x_obj = -np.inf
                            else:
                                x_obj = np.inf
                            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                                nodes_per_depth[i] -= 2 ** (i - current_node.depth)
                            stop, lower_bound, upper_bound = check_depth_completion(
                                current_node.depth, nodes_per_depth, best_bound_per_depth,
                                lower_bound, upper_bound, isMax, DEBUG_MODE
                            )
                            if stop:
                                return solutions, best_sol_idx, solutions_found
                            if DEBUG_MODE:
                                debug_print(node=current_node, sol_status="Infeasible after cuts")
                            continue
            # ===========================

            # update best bound per depth if a better solution was found
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

        # If infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

        # Found feasible solution.
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:
                    lower_bound = x_obj
                if abs(lower_bound - upper_bound) < 1e-6:
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                    return solutions, best_sol_idx, solutions_found

                solutions.append([x_candidate, x_obj, current_node.depth])
                solutions_found += 1
                if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                    best_sol_obj = x_obj
                    best_sol_idx = solutions_found - 1

                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth,
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop:
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                continue

            else:
                if upper_bound > x_obj:
                    upper_bound = x_obj
                if abs(lower_bound - upper_bound) < 1e-6:
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found >= 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                    return solutions, best_sol_idx, solutions_found

                solutions.append([x_candidate, x_obj, current_node.depth])
                solutions_found += 1
                if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found >= 1:
                    best_sol_obj = x_obj
                    best_sol_idx = solutions_found - 1

                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth,
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop:
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                continue

        # Prune by bound
        if isMax:
            if x_obj < lower_bound:
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth,
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop:
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        else:
            if x_obj > upper_bound:
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth,
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop:
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
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

        # Add child nodes in heap queue
        if isMax:
            heapq.heappush(heap, (-np.inf, right_child.id, right_child))
            heapq.heappush(heap, (-np.inf, left_child.id, left_child))
        else:
            heapq.heappush(heap, (np.inf, right_child.id, right_child))
            heapq.heappush(heap, (np.inf, left_child.id, left_child))

    return solutions, best_sol_idx, solutions_found