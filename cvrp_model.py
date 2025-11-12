import json
import pyomo.environ as pyo
from pyomo.environ import Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals, minimize, inequality

def build_cvrp_model(instance_file="cvrp_instance.json"):

    # 1. load data
    try:
        with open(instance_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Instance file '{instance_file}' not found.")
        return None

    # process data
    N = data["N"]  # num of vertices (0 is depot, 1 to N-1 are customers)
    V_set = range(N)
    A_set = [(i, j) for i in V_set for j in V_set if i != j]

    K_param = data["K"]
    Q_param = data["Q"]
    c_param = {(i, j): data["costs"][i][j] for i, j in A_set}
    q_param = {i: data["demands"][i] for i in V_set}

    # 2. model init
    model = pyo.ConcreteModel()

    # 3. sets
    # V: Set of all vertices (0 is depot)
    model.V = Set(initialize=V_set)
    # V_cust: Set of customer vertices (excluding depot)
    model.V_cust = Set(initialize=[i for i in V_set if i != 0])
    # A: Set of all arcs
    model.A = Set(within=model.V * model.V, initialize=A_set)

    # 4. params
    # c[i, j]: Cost of arc (i, j)
    model.c = Param(model.A, initialize=c_param)
    # q[i]: Demand of vertex i (q[0] = 0)
    model.q = Param(model.V, initialize=q_param)
    # Q: Vehicle capacity
    model.Q = Param(initialize=Q_param)
    # K: Number of available vehicles
    model.K = Param(initialize=K_param)

    # 5. vars
    # x[i, j]: binary variable -> 1 if arc (i, j) is traversed, 0 otherwise
    model.x = pyo.Var(model.A, domain=pyo.Binary)
    # f[i, j]: Amount of commodity flow (load) carried on arc (i, j)
    model.f = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    # 6. obj func (minimize total cost)
    def obj_rule(model):
        # (1) in the paper (Minimize sum(c_ij * x_ij))
        return sum(model.c[i, j] * model.x[i, j] for (i, j) in model.A)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # 7. constraints

    # 7.1. each customer has a exactly one emtering arc (Equivalent to x(δ⁻(Ck)) = 1)
    def visit_in_rule(model, i):
        return sum(model.x[j,i] for j in model.V if (j,i) in model.A) == 1
    model.visit_in = pyo.Constraint(model.V_cust, rule=visit_in_rule, doc='customer is entered once')

    # 7.2. each customer has a exactly one leaving arc (x(δ⁺(Ck)) = 1)
    def visit_out_rule(model, i):
        return sum(model.x[i,j] for j in model.V if (i,j) in model.A) == 1
    model.visit_out = pyo.Constraint(model.V_cust, rule=visit_out_rule, doc='customer is left once')

    # 7.3. Exactly K vehicles depart from the depot (Equivalent to x(δ⁺(C0)) = K)
    def depot_out_rule(model):
        return sum(model.x[0,j] for j in model.V_cust if (0,j) in model.A) == model.K
    model.depot_out = pyo.Constraint(rule=depot_out_rule, doc='K vehicles depart depot')

    # 7.4. Route Continuity (Flow Conservation for Assignment Variables)
    # constraint (5) in the paper: x(δ⁺(i)) = x(δ⁻(i))
    def flow_conservation_rule(model, i):
        sum_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)
        sum_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        return sum_out == sum_in
    model.flow_conservation = pyo.Constraint(model.V, rule=flow_conservation_rule, doc='assignment flow conservation')

    # 7.5. Flow Balance (Commodity Flow)
    # constraint (6) in the paper for i in V\{0} without depot
    # flow balance -> q[i] is vertex demand:
    # sum(f_ij) - sum(f_ji) = 0.5 * q[i] * (sum(x_ji) + sum(x_ij))
    def flow_balance_rule(model, i):
        if i == 0:
            return Constraint.Skip  # depot does not consume demand

        sum_f_out = sum(model.f[i,j] for j in model.V if (i,j) in model.A)
        sum_f_in = sum(model.f[j,i] for j in model.V if (j,i) in model.A)

        # term (sum(x_ji) + sum(x_ij)) is 2 for a visited customer node
        sum_x_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        sum_x_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)

        # constraint that links flow and assignment. ensures the commodity (demand) is dropped
        lhs = sum_f_out - sum_f_in
        rhs = 0.5 * model.q[i] * (sum_x_in + sum_x_out)
        return lhs == rhs
    model.flow_balance = pyo.Constraint(model.V_cust, rule=flow_balance_rule, doc='commodity flow balance at customer nodes')

    # 7.6. Capacity Constraints
    # strengthened bound (9) from the paper:

    # capacity lower bound (flow at least the demand of the starting node, if the arc is used)
    # f_ij >= q_i * x_ij  =>  f_ij - q_i * x_ij >= 0

    def capacity_lower_rule(model, i, j):
        if i == j:
            return pyo.Constraint().Skip()

        return model.f[i,j] >= model.q[i] * model.x[i,j]

    model.capacity_lower = pyo.Constraint(model.A, rule=capacity_lower_rule, doc='flow must carry at least the starting node demand')

    # capacity upper bound -> flow not exceed vehicle capacity minus the arrival node's demand
    # f_ij <= (Q - q_j) * x_ij  =>  f_ij - (Q - q_j) * x_ij <= 0

    def capacity_upper_rule(model, i, j):
        if i == j:
            return pyo.Constraint.Skip

        return model.f[i,j] <= (model.Q - model.q[j]) * model.x[i,j]

    model.capacity_upper = pyo.Constraint(model.A, rule=capacity_upper_rule, doc='flow must respect remaining vehicle capacity')

    return model

if __name__ == "__main__":
    dummy_instance = {
        "N": 4,  # vertex 0 (depot), and 1, 2, 3 (custs)
        "K": 2,  # vehicles
        "Q": 10, # capacity
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
            solver = pyo.SolverFactory("gurobi")
            result = solver.solve(model, tee=True)

            print("\n----- model (partial) -----")
            model.pprint()
            print("\n----- results -----")
            print(result)

            # disp results
            print(f"\optimal objective value: {pyo.value(model.obj)}")

        except Exception as e:
            print(f"model could not be solved. error: {e}")
