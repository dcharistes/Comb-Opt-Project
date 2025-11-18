import sys
import pyomo.environ as pyo
from pyomo.environ import Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals, minimize, inequality

def read_data_gvrp(filename="gvrp_instance.txt"):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse header line
    header = lines[0].split()
    grid_total_nodes = int(header[0])  # grid_size * grid_size
    V = int(header[1])                 # number of nodes
    M = int(header[2])                 # number of clusters
    K = int(header[3])                 # number of vehicles
    Q = float(header[4])               # vehicle capacity

    # Parse clusters
    line_idx = 2  # Skip header line + line with 'M Clusters:'
    cluster_nodes = {}
    a = []  # node-to-cluster mapping, sequential indexing will be used
    q_cluster = [0] * (M + 1)
    node_coords = []

    depot_node_id = None
    node_id_map = {}  # Map from (x,y) to sequential node ID

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

        # Assign cluster demand (assume all nodes within a cluster have same demand)
        q_cluster[cluster_id] = demand

        if cluster_id == 0:
            depot_node_id = current_node_id

        current_node_id += 1
        line_idx += 1

    N = current_node_id  # Actual number of nodes (depot + customers)

    # Parse arcs
    line_idx += 1  # Skip 'Arcs:' line
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
    print(f"Arcs List:{arc_list}")
    # Return parsed data
    return N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id


def build_gvrp_model(N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id):

    # 1. process data
    V_set = range(N)
    A_set = arc_list # set of arcs

    a_param = {i: a[i] for i in V_set}
    c_param = cost_param #costs
    q_cluster_param = {k: q_cluster[k] for k in range(M + 1)}

    # 2. model init
    model = pyo.ConcreteModel()

    # 3. sets
    # V: set of vertices (0 is depot)
    model.V = Set(initialize=V_set, doc="Set of all nodes (0 is depot)")
    # V_cust: set of customer vertices (without depot)
    model.V_cust = Set(initialize=[i for i in V_set if i != depot_node_id])
    # A: set of arcs
    model.A = Set(dimen=2, initialize=A_set)
    # Cluster Set
    model.C = Set(initialize=range(M + 1), doc="Set of clusters")

    # 4. params
    # c[i, j]: cost of arc (i, j)
    model.c = Param(model.A, initialize=c_param)
    # a(i): the cluster index of vertex i
    model.a = Param(model.V, initialize=a_param)
    # q[a(i)]: demand of cluster a(i) that the i node is in it.
    model.q_cluster = Param(model.C, initialize=q_cluster_param)
    # Q: vehicle capacity
    model.Q = Param(initialize=Q)
    # K: number of available vehicles
    model.K = Param(initialize=K)

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

    # 7.x. exactly K vehicles return to the depot (Equivalent to x(δ-(C0)) = K)
    def depot_in_rule(model):
        # We check arcs from any customer node back to the depot
        return sum(model.x[j,depot_node_id] for j in model.V_cust if (j,depot_node_id) in model.A) == model.K
    model.depot_in = pyo.Constraint(rule=depot_in_rule, doc='K vehicles return to depot')

    # 7.3. exactly K vehicles depart from the depot (Equivalent to x(δ+(C0)) = K)
    def depot_out_rule(model):
        return sum(model.x[depot_node_id,j] for j in model.V_cust if (depot_node_id,j) in model.A) == model.K
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
        if i == depot_node_id:
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
if __name__ == "__main__":
    # Read instance from generator output
    instance_filename = "./5_15_4/0.txt"

    try:
        N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id = read_data_gvrp(instance_filename)

        # Build model
        model = build_gvrp_model(N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id)

        print("\n----- Model created successfully -----")
        print(f"Model has {len(model.A)} arcs, {len(model.V_cust)} customers, {M} clusters")

        # Solve model
        solver = pyo.SolverFactory("gurobi")

        print("\n----- Solving model -----")
        result = solver.solve(model, tee=True)

        print("\n----- Solution -----")
        print(result)

        if result.solver.status == 'ok':
            print(f"\nOptimal objective value: {pyo.value(model.obj)}")

            # Display solution
            print("\n--- Arcs used in solution ---")
            for (i, j) in model.A:
                if pyo.value(model.x[i, j]) > 0.5:
                    flow = pyo.value(model.f[i, j])
                    print(f"Arc ({i}, {j}): flow = {flow:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

