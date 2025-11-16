import sys
import pyomo.environ as pyo
from pyomo.environ import Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals, minimize, inequality

def read_data_cvrp(filename="cvrp_instance.txt"):

    with open(filename, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    # --- N, K, Q ---
    header = lines[0].split()
    N = int(header[0])      # nodes (with depot)
    K = int(header[1])      # vehicles
    Q = float(header[2])    # vechicle capacity

    # read demands
    demands_line = lines[1].split()
    demands = [float(d) for d in demands_line]
    if len(demands) != N:
        raise ValueError(f"Mismatch: N={N} but {len(demands)} demands found on line 2.")

    #  read cost matrix
    costs = []
    cost_lines = lines[2:2+N] # get next N lines
    if len(cost_lines) != N:
        raise ValueError(f"Mismatch: N={N} but {len(cost_lines)} cost matrix rows found.")

    for line in cost_lines:
        cost_row = [float(c) for c in line.split()]
        if len(cost_row) != N:
            raise ValueError(f"Cost matrix row has {len(cost_row)} items, but expected {N}.")
        costs.append(cost_row)


    print(f"Successful data read: N={N}, K={K}, Q={Q}")
    return N, K, Q, demands, costs

def build_cvrp_model(N, K, Q, demands, costs, vertex_to_cluster, cluster_demands):

    # 1. process data
    V_set = range(N)
    A_set = [(i, j) for i in V_set for j in V_set if i != j] # set of arcs

    K_param = K #vehicle number
    Q_param = Q #capacity
    c_param = {(i, j): costs[i][j] for i, j in A_set} #costs
    q_param = {i: demands[i] for i in V_set} # demand of each cluster. maybe not in V_set. V_set is for all the nodes. we need the clusters here!

    # 2. model init
    model = pyo.ConcreteModel()

    # 3. sets
    # V: set of vertices (0 is depot)
    model.V = Set(initialize=V_set)
    # V_cust: set of customer vertices (without depot)
    model.V_cust = Set(initialize=[i for i in V_set if i != 0])
    # A: set of arcs
    model.A = Set(within=model.V * model.V, initialize=A_set)

    # 4. params
    # c[i, j]: cost of arc (i, j)
    model.c = Param(model.A, initialize=c_param)
    # q[i]: demand of vertex i (q[0] = 0)
    model.q = Param(model.V, initialize=q_param)
    # a(i): the cluster index of vertex i
    model.a = Param(model.V, initialize=vertex_to_cluster)
    # q[a(i)]: demand of cluster a(i) that the i node is in it.
    model.q_cluster = Param(initialize=cluster_demands)
    # Q: vehicle capacity
    model.Q = Param(initialize=Q_param)
    # K: number of available vehicles
    model.K = Param(initialize=K_param)

    # 5. vars
    # x[i, j]: binary variable -> 1 if arc (i, j) is traversed, 0 otherwise
    model.x = pyo.Var(model.A, domain=pyo.Binary)
    # f[i, j]: amount of commodity flow (load) carried on arc (i, j)
    model.f = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    # 6. obj func (minimize total cost)
    def obj_rule(model):
        # (1) in the paper (Minimize sum(c_ij * x_ij))
        return sum(model.c[i, j] * model.x[i, j] for (i, j) in model.A)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # 7. constraints

    # 7.1. each customer has a exactly one emtering arc (Equivalent to x(δ-(Ck)) = 1)
    def visit_in_rule(model, i):
        return sum(model.x[j,i] for j in model.V if (j,i) in model.A) == 1
    model.visit_in = pyo.Constraint(model.V_cust, rule=visit_in_rule, doc='customer is entered once')

    # 7.2. each customer has a exactly one leaving arc (x(δ+(Ck)) = 1)
    def visit_out_rule(model, i):
        return sum(model.x[i,j] for j in model.V if (i,j) in model.A) == 1
    model.visit_out = pyo.Constraint(model.V_cust, rule=visit_out_rule, doc='customer is left once')

    # 7.3. exactly K vehicles depart from the depot (Equivalent to x(δ+(C0)) = K)
    def depot_out_rule(model):
        return sum(model.x[0,j] for j in model.V_cust if (0,j) in model.A) == model.K
    model.depot_out = pyo.Constraint(rule=depot_out_rule, doc='K vehicles depart depot')

    # 7.4. route continuity
    # constraint 5 in the paper: x(δ+(i)) = x(δ-(i))
    def flow_conservation_rule(model, i):
        sum_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)
        sum_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        return sum_out == sum_in
    model.flow_conservation = pyo.Constraint(model.V, rule=flow_conservation_rule, doc='assignment flow conservation')

    # 7.5. flow blance (Commodity Flow)
    # constraint (6) in the paper for i in V\{0} without depot

    # flow balance -> q[a(i)] is the cluster demand of node i:
    def flow_balance_rule(model, i):
        if i == 0:
            return Constraint.Skip  # depot has no demand

        sum_f_out = sum(model.f[i,j] for j in model.V if (i,j) in model.A)
        sum_f_in = sum(model.f[j,i] for j in model.V if (j,i) in model.A)

        # term (sum(x_ji) + sum(x_ij)) is 2 for a visited customer node
        sum_x_in = sum(model.x[j,i] for j in model.V if (j,i) in model.A)
        sum_x_out = sum(model.x[i,j] for j in model.V if (i,j) in model.A)

        # constraint: flow and assignment link. ensures the commodity (demand) is dropped
        lhs = sum_f_out - sum_f_in
        cluster_i = model.a[i]
        rhs = 0.5 * model.q_cluster[cluster_i] * (sum_x_in + sum_x_out)
        return lhs == rhs
    model.flow_balance = pyo.Constraint(model.V_cust, rule=flow_balance_rule, doc='commodity flow balance at customer nodes')

    # 7.6. capacity constraints
    # strengthened bound 9 from the paper:

    # capacity lower bound -> flow of arc i,j >= the cluster demand of the node i. if the arc ij, is used
    def capacity_lower_rule(model, i, j):
        if i == j:
            return pyo.Constraint().Skip()

        cluster_i = model.a[i]
        return model.f[i,j] >= model.q_cluster[cluster_i] * model.x[i,j]

    model.capacity_lower = pyo.Constraint(model.A, rule=capacity_lower_rule, doc='flow must carry at least the starting node demand')

    # capacity upper bound -> (flow_ij) <=  (vehicle capacity) - (the cluster demand of node j)
    def capacity_upper_rule(model, i, j):
        if i == j:
            return pyo.Constraint.Skip

        cluster_j = model.a[j]
        return model.f[i,j] <= (model.Q - model.q_cluster[cluster_j]) * model.x[i,j]

    model.capacity_upper = pyo.Constraint(model.A, rule=capacity_upper_rule, doc='flow must respect remaining vehicle capacity')

    return model


# solve the generator's random instances
instance_file_name = "cvrp_instance.json"
with open(instance_file_name, "r") as f:
    instance_data = json.load(f)

model = build_cvrp_model(instance_file_name)

solver = pyo.SolverFactory("gurobi")
result = solver.solve(model, tee=True)

print("\n----- model (partial) -----")
model.pprint()
print("\n----- results -----")
print(result)

# disp results
print(f"\optimal objective value: {pyo.value(model.obj)}")

