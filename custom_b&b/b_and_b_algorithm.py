import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import gvrp_model_gurobi as pr

# Set sense (min/max)
isMax = False

WARM_START = True
DEBUG_MODE = True

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

# A simple function to print debugging info
def debug_print(node:Node = None, x_obj = None, sol_status = None):

        print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
        print(f"UB:{upper_bound}")
        print(f"LB:{lower_bound}")
        if node is not None:
            print(f"Brancing Var: {node.branching_var}")
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
            if ub <= lb + 1e-6: # Check termination
                if DEBUG_MODE: print("Global UB hit LB (Level Completed). Stopping.")
                stop = True
        else:
            # Update Global Lower Bound
            lb = best_bound_per_depth[depth]
            if lb >= ub - 1e-6: # Check termination
                if DEBUG_MODE: print("Global LB hit UB (Level Completed). Stopping.")
                stop = True

    return stop, lb, ub

# Definition of the branch & bound algorithm.
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound


    # We force conversion to Python INT (arbitrary precision) to avoid OverflowError
    if not isinstance(nodes_per_depth[0], (int, np.integer)) or nodes_per_depth[0] == 0:
        print("WARNING: Re-initializing nodes_per_depth as arbitrary-precision integers.")
        # Re-initialize as object array or list of ints to support huge numbers
        # But actually, standard numpy array with dtype=object allows python ints
        nodes_per_depth = np.array([0] * len(nodes_per_depth), dtype=object)
        nodes_per_depth[0] = 1
        for i in range(1, len(nodes_per_depth)):
            nodes_per_depth[i] = nodes_per_depth[i-1] * 2


    # Create stack using deque() structure
    stack = deque()

    solutions = list()
    solutions_found = 0
    best_sol_idx = 0

    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # Create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # ===============  Root node  ==========================

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
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break



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
    left_child = Node(left_ub, left_lb,root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Add child nodes in stack
    stack.append(right_child)
    stack.append(left_child)

    # Solving sub problems
    # While the stack has nodes, continue solving
    while(len(stack) != 0):
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # Increment total nodes by 1
        nodes += 1

        # Get the child node on top of stack
        current_node = stack[-1]

        # Remove this node from stack
        stack.pop()

        # Increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1


        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        #print(f"LB: {current_node.lb}")
        #print(f"UB: {current_node.ub}")

        # Update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
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
        # If integer solution found, then:
            # 1) - If solution improves incumbent, then store otherwise reject (optional)
            # If improves:
            # 2) - Update lb/ub for max/min respectively.
            # 3) - Check optimality condition lb=ub.
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:
                    lower_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6:

                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1


                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                    # if we reached the final node of a depth, then update the bounds
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

                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found >= 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found >= 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                    # if we reached the final node of a depth, then update the bounds
                    stop, lower_bound, upper_bound = check_depth_completion(
                        current_node.depth, nodes_per_depth, best_bound_per_depth,
                        lower_bound, upper_bound, isMax, DEBUG_MODE
                    )
                    if stop:
                        return solutions, best_sol_idx, solutions_found


                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # do not branch further if is an equal solution
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 ** (i - current_node.depth)

            # if we reached the final node of a depth, then update the bounds
            stop, lower_bound, upper_bound = check_depth_completion(
                current_node.depth, nodes_per_depth, best_bound_per_depth,
                lower_bound, upper_bound, isMax, DEBUG_MODE
            )
            if stop:
                return solutions, best_sol_idx, solutions_found

            # Do not branch further if is an equal solution
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue


        # If lb/ub for max/min respectively, is greater/less than x_obj then prune.
        # Here we accept x_obj = lb/ub (to potentially discover another solution with equal obj value) but this is optional.
        # If we wanted to prune, the condition is: x_obj lower-equal (<=) to lower_bound    for a maximization problem.
        # For example:
        # if isMax:
        #   if (x_obj < lower_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue
        # else:
        #   if (x_obj > upper_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue


        if isMax:

            if x_obj < lower_bound:

                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                # if we reached the final node of a depth, then update the bounds
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

                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                # if we reached the final node of a depth, then update the bounds
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
            # Retrieve vbasis and cbasis
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        # Create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)


        # Create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        # Add child nodes in stack
        stack.append(right_child)
        stack.append(left_child)

    return solutions, best_sol_idx, solutions_found


# if __name__ == "__main__":

#     print("************************    Initializing structures...    ************************")

#     model, ub, lb, integer_var, num_vars, c = pr.pcenter("./3_9_2_4_10_2_2/9.txt")
#     isMax = False

#     # Initialize structures
#     # Keep the best bound per depth and the total nodels visited for each depth
#     if isMax == True:
#         best_bound_per_depth = np.array([-np.inf for i in range(num_vars + 1)])
#     else:
#         best_bound_per_depth = np.array([np.inf for i in range(num_vars + 1)])


#     nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
#     nodes_per_depth[0] = 1
#     for i in range(1, num_vars + 1):
#         nodes_per_depth[i] = nodes_per_depth[i - 1] * 2


#     # Start solving
#     print("************************    Solving problem...    ************************")
#     start = time.time()
#     solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth)
#     end = time.time()


#     # Print results
#     print("========= Optimal Solutions =========")
#     """print("Variable:")
#     for i in range(len(solutions[-2][0])):
#         print(f">  x{i} = {solutions[-2][0][i]}")

#     print(f"Objective Value: {solutions[-2][1]}")
#     print(f"Tree depth: {solutions[-2][2]}")"""

#     print(solutions[best_sol_idx][0])
#     print(f"Objective Value: {solutions[best_sol_idx][1]}")
#     print(f"Tree depth: {solutions[best_sol_idx][2]}")
#     print()
#     print(solutions)
#     print(f"Time Elapsed: {end - start}")
#     print(f"Total nodes: {nodes}")



#     """
#     For defining a simple problem (add in main in order to use it)


#     # Initialize structures and model
#     num_vars = 5
#     # Initialize structures
#     if isMax == True:
#         best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
#     else:
#         best_bound_per_depth = np.array([np.inf for i in range(num_vars)])
#     nodes_per_depth = np.array([0 for i in range(num_vars)])

#     # Create an empty model
#     model = gp.Model()

#     # Upper and lower bounds for variables
#     lb = [0, 0, 0, 0, 0]
#     ub = [1, 1, 1, 1, 1]

#     # Add variables to the model, as continuous (relaxed problem)
#     x = model.addVars(5, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

#     # Define objective function
#     model.setObjective(-8 * x[0] - 2 * x[1] - 4 * x[2] - 7 * x[3] - 5 * x[4] + 10)

#     # We have a maximization problem
#     model.ModelSense = GRB.MAXIMIZE

#     # Adding constraints
#     model.addLConstr(-3 * x[0] - 3 * x[1] + x[2] + 2 * x[3] + 3 * x[4] <= -2)
#     model.addLConstr(-5 * x[0] - 3 * x[1] - 2 * x[2] - x[3] + x[4] <= -4)

#     model.Params.method = 1  # 1 indicates the dual Simplex algorithm in Gurobi

#     model.update()
#     # Define which variables should have integer values
#     integer_var = [True, True, True, True, True]


#     """
