import json
import pyomo.environ as pyo
from pyomo.environ import Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals, minimize, inequality

def build_cvrp_model(instance_file="cvrp_instance.json"):
    """
    Builds the single-commodity flow formulation for the Capacitated Vehicle Routing Problem (CVRP)
    using a structure similar to the provided production.py, based on the flow model
    (Gavish and Graves, 1978).
    """

    # --- 1. Load Data ---
    try:
        with open(instance_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Instance file '{instance_file}' not found.")
        return None

    # Process input data
    N = data["N"]  # Number of vertices (0 is depot, 1 to N-1 are customers)
    V_set = range(N)
    A_set = [(i, j) for i in V_set for j in V_set if i != j]

    K_param = data["K"]
    Q_param = data["Q"]
    c_param = {(i, j): data["costs"][i][j] for i, j in A_set}
    q_param = {i: data["demands"][i] for i in V_set}

    # --- 2. Initialize Model ---
    model = pyo.ConcreteModel()

    # --- 3. Sets ---
    # V: Set of all vertices (0 is depot)
    model.V = Set(initialize=V_set)
    # V_cust: Set of customer vertices (excluding depot)
    model.V_cust = Set(initialize=[i for i in V_set if i != 0])
    # A: Set of all arcs
    model.A = Set(within=model.V * model.V, initialize=A_set)

    # --- 4. Parameters ---
    # c[i, j]: Cost of traversing arc (i, j)
    model.c = Param(model.A, initialize=c_param)
    # q[i]: Demand of vertex i (q[0] = 0)
    model.q = Param(model.V, initialize=q_param)
    # Q: Vehicle capacity
    model.Q = Param(initialize=Q_param)
    # K: Number of available vehicles
    model.K = Param(initialize=K_param)

    # --- 5. Variables ---
    # x[i, j]: 1 if arc (i, j) is traversed, 0 otherwise (Assignment/Route variable)
    model.x = pyo.Var(model.A, domain=pyo.Binary)
    # f[i, j]: Amount of commodity flow (load) carried on arc (i, j)
    model.f = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    # --- 6. Objective Function (Minimize total cost) ---
    def obj_rule(model):
        # Corresponds to (1) in the paper (Minimize sum(c_ij * x_ij))
        return sum(model.c[i, j] * model.x[i, j] for (i, j) in model.A)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 7. Constraints ---

    # Assignment Constraints (Adapted for CVRP where each vertex is a 'cluster' Ck)

    # 7.1. Each customer is entered exactly once (Equivalent to x(δ⁻(Ck)) = 1)
    def visit_in_rule(model, i):
        return sum(model.x[j,i] for j in model.V if (j,i) in model.A) == 1
    model.visit_in = pyo.Constraint(model.V_cust, rule=visit_in_rule, doc='Customer is entered once')

    # 7.2. Each customer is left exactly once (Equivalent to x(δ⁺(Ck)) = 1)
    def visit_out_rule(model, i):
        return sum(model.x[i,j] for j in model.V if (i,j) in model.A) == 1
    model.visit_out = pyo.Constraint(model.V_cust, rule=visit_out_rule, doc='Customer is left once')

    # 7.3. Exactly K vehicles depart from the depot (Equivalent to x(δ⁺(C0)) = K)
    def depot_out_rule(model):
        return sum(model.x[0,j] for j in model.V_cust if (0,j) in model.A) == model.K
    model.depot_out = pyo.Constraint(rule=depot_out_rule, doc='K vehicles depart depot')

    # 7.4. Route Continuity (Flow Conservation for Assignment Variables)
    # Corresponds to (5) in the paper: x(δ⁺(i)) = x(δ⁻(i))
    # Note: In CVRP/GVRP, the constraints (7.1) and (7.2) for customer nodes imply this.
    # It is mainly non-trivial for the depot (i=0), but the existing code used it for all i.
    def flow_conservation_rule(model, i):
        sum_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)
        sum_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        return sum_out == sum_in
    model.flow_conservation = pyo.Constraint(model.V, rule=flow_conservation_rule, doc='Assignment flow conservation')

    # 7.5. Flow Balance (Commodity Flow)
    # Corresponds to (6) in the paper for i in V\{0}, but uses the CVRP version from the original code.
    # Original code's flow balance (for CVRP, where q[i] is vertex demand):
    # sum(f_ij) - sum(f_ji) = 0.5 * q[i] * (sum(x_ji) + sum(x_ij))
    def flow_balance_rule(model, i):
        if i == 0:
            return Constraint.Skip  # Depot does not consume demand

        sum_f_out = sum(model.f[i,j] for j in model.V if (i,j) in model.A)
        sum_f_in = sum(model.f[j,i] for j in model.V if (j,i) in model.A)

        # The term (sum(x_ji) + sum(x_ij)) is 2 for a visited customer node
        sum_x_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        sum_x_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)

        # This constraint links flow with assignment and ensures the commodity (demand) is 'dropped off'
        # equal to its demand q[i] when the node is visited.
        lhs = sum_f_out - sum_f_in
        rhs = 0.5 * model.q[i] * (sum_x_in + sum_x_out)
        return lhs == rhs
    model.flow_balance = pyo.Constraint(model.V_cust, rule=flow_balance_rule, doc='Commodity flow balance at customer nodes')

    # 7.6. Capacity Constraints (Strengthened bounds)
    # Corresponds to the strengthened bound (9) from the paper:
    # q_alpha(i) * x_ij <= f_ij <= (Q - q_alpha(j)) * x_ij
    # For CVRP, alpha(i) is just i, and q_alpha(i) is q[i].
    # Constraint (9) in CVRP: q[i] * x_ij <= f_ij <= (Q - q[j]) * x_ij
    # --- 7.6a. Capacity Lower Bound (Flow must be at least the demand of the starting node, if arc is used)
    # f_ij >= q_i * x_ij  =>  f_ij - q_i * x_ij >= 0

    def capacity_lower_rule(model, i, j):
        if i == j:
            return pyo.Constraint().Skip()

        # Left side: f_ij
        # Right side: q_i * x_ij
        return model.f[i,j] >= model.q[i] * model.x[i,j]

    model.capacity_lower = pyo.Constraint(model.A, rule=capacity_lower_rule, doc='Flow must carry at least the starting node demand')

    # --- 7.6b. Capacity Upper Bound (Flow must not exceed vehicle capacity minus demand of arrival node)
    # f_ij <= (Q - q_j) * x_ij  =>  f_ij - (Q - q_j) * x_ij <= 0

    def capacity_upper_rule(model, i, j):
        if i == j:
            return pyo.Constraint.Skip

        # Left side: f_ij
        # Right side: (Q - q_j) * x_ij
        return model.f[i,j] <= (model.Q - model.q[j]) * model.x[i,j]

    model.capacity_upper = pyo.Constraint(model.A, rule=capacity_upper_rule, doc='Flow must respect remaining vehicle capacity')

    return model

if __name__ == "__main__":
    # Create a dummy instance file for testing the model structure
    dummy_instance = {
        "N": 4,  # Vertices 0 (depot), 1, 2, 3 (customers)
        "K": 2,  # 2 vehicles
        "Q": 10, # Capacity 10
        "costs": [
            [0, 1, 2, 3], # 0 to 1, 2, 3
            [1, 0, 1, 4], # 1 to 0, 2, 3
            [2, 1, 0, 1], # 2 to 0, 1, 3
            [3, 4, 1, 0]  # 3 to 0, 1, 2
        ],
        "demands": [0, 5, 3, 4] # q[0]=0, q[1]=5, q[2]=3, q[3]=4
    }
    instance_file_name = "cvrp_instance.json"
    with open(instance_file_name, "w") as f:
        json.dump(dummy_instance, f, indent=4)

    model = build_cvrp_model(instance_file_name)
    if model:
        try:
            # Note: SolverFactory("gurobi") requires Gurobi to be installed and licensed.
            # Using 'glpk' as a common alternative for demonstration, though 'gurobi'
            # is often faster for VRP.
            solver = pyo.SolverFactory("gurobi")
            # If the original solver was gurobi, keep it:
            # solver = pyo.SolverFactory("gurobi")
            result = solver.solve(model, tee=True)

            print("\n----- Printing the model (Partial) -----")
            model.pprint()
            print("\n----- Printing the results -----")
            print(result)

            # Display results
            print(f"\nOptimal objective value: {pyo.value(model.obj)}")

        except Exception as e:
            print(f"Could not solve the model. Ensure the solver is correctly installed and licensed. Error: {e}")
