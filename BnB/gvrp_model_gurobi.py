import gurobipy as gp
from gurobipy import GRB
import numpy as np

def read_data_gvrp(filename):
    """
    Reads the GVRP instance and returns raw data structures.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    V = int(header[1])
    M = int(header[2])
    K = int(header[3])
    Q = float(header[4])

    line_idx = 2
    cluster_nodes = {}
    a = []  # map nodes to clusters
    q_cluster = [0] * (M + 1)

    node_id_map = {} # (x,y) -> id
    current_node_id = 0
    depot_node_id = 0

    # parsing nodes
    while "Arcs:" not in lines[line_idx]:
        parts = lines[line_idx].split()
        cluster_id = int(parts[0])
        x, y = int(parts[1]), int(parts[2])
        demand = float(parts[3])

        node_id_map[(x, y)] = current_node_id
        a.append(cluster_id)

        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(current_node_id)

        q_cluster[cluster_id] = demand
        if cluster_id == 0:
            depot_node_id = current_node_id

        current_node_id += 1
        line_idx += 1

    N = current_node_id
    line_idx += 1

    # parsing arcs
    arc_list = []
    cost_param = {}
    while line_idx < len(lines):
        parts = lines[line_idx].split()
        x1, y1, x2, y2 = map(int, parts[0:4])
        dist = float(parts[4])

        u = node_id_map[(x1, y1)]
        v = node_id_map[(x2, y2)]

        arc_list.append((u, v))
        cost_param[(u, v)] = dist
        line_idx += 1

    print(f"Successfully read GVRP instance:")
    print(f"  N (total nodes): {N}")
    print(f"  K (vehicles): {K}")
    print(f"  Q (capacity): {Q}")
    print(f"  M (clusters): {M}")
    print(f"  Cluster demands: {q_cluster}")
    print(f"  Nodes per cluster: {cluster_nodes}")

    return N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id, cluster_nodes

def gvrp_model_gurobi(filename):
    """
    Builds the GVRP Commodity Flow model in Gurobi.
    Returns the tuple expected by b&b_iterative.py:
    (model, ub, lb, integer_var, num_vars, c)
    """
    # 1. read data
    N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_id, cluster_nodes = read_data_gvrp(filename)

    # 2. setup var map
    # b&b_iterative.py works with linear indices, so we map arcs to 1D indices
    num_arcs = len(arc_list)
    # Ttotal vars = (x variables for arcs) + (f variables for arcs)
    num_vars = 2 * num_arcs

    # arc (u,v) -> index k (0 to num_arcs-1)
    arc_to_idx = {arc: i for i, arc in enumerate(arc_list)}

    # 3. lists for B&B

    # ub&lb: x is [0,1], f is [0, Q]
    lb = [0.0] * num_vars
    ub = [0.0] * num_vars
    c  = [0.0] * num_vars
    integer_var = [False] * num_vars

    for i in range(num_arcs):
        # relaxation -> bounds [0, 1]. integer -> (0, 1)
        lb[i] = 0.0
        ub[i] = 1.0
        c[i] = cost_param[arc_list[i]]  # cost is on x
        integer_var[i] = True           # these are the vars we want to branch (x)

        f_idx = num_arcs + i
        lb[f_idx] = 0.0
        ub[f_idx] = Q                   # max flow is Q
        c[f_idx] = 0.0                  # no cost for flow
        integer_var[f_idx] = False      # NO branching on flow vars (can be continous)

    # 4. build model
    model = gp.Model()

    # all variables are added at once -> index order 0..num_vars is maintained
    # b&b_iterative relies on model.getVars() returning this exact order
    vars_list = model.addVars(num_vars, lb=lb, ub=ub, obj=c, vtype=GRB.CONTINUOUS, name="v")
    model.update()

    # access variables by arc (helper func)
    def get_x(u, v):
        return vars_list[arc_to_idx[(u, v)]]

    def get_f(u, v):
        return vars_list[num_arcs + arc_to_idx[(u, v)]]

    # 5. Add Constraints

    # C1 & C2: Visit every customer cluster exactly once (skip depot 0)
    for k in range(1, M + 1):
        nodes_in_k = cluster_nodes[k]

        # entering k
        model.addLConstr(
            gp.quicksum(get_x(u, v) for v in nodes_in_k for u in range(N) if (u, v) in arc_to_idx) == 1,
            name=f"ClustIn_{k}"
        )

        # leaving k
        model.addLConstr(
            gp.quicksum(get_x(u, v) for u in nodes_in_k for v in range(N) if (u, v) in arc_to_idx) == 1,
            name=f"ClustOut_{k}"
        )

    # C3: depot flow (K vehicles)
    depot_out_arcs = [get_x(depot_id, v) for v in range(N) if (depot_id, v) in arc_to_idx]
    depot_in_arcs  = [get_x(u, depot_id) for u in range(N) if (u, depot_id) in arc_to_idx]

    model.addLConstr(gp.quicksum(depot_out_arcs) == K, name="DepotOut")
    model.addLConstr(gp.quicksum(depot_in_arcs) == K, name="DepotIn")

    # C4: route continuity (node flow conservation for x)
    for i in range(N):
        x_in = gp.quicksum(get_x(u, i) for u in range(N) if (u, i) in arc_to_idx)
        x_out = gp.quicksum(get_x(i, v) for v in range(N) if (i, v) in arc_to_idx)
        model.addLConstr(x_in == x_out, name=f"FlowCons_{i}")

    # C5: commodity flow balance
    total_demand = sum(q_cluster)

    for i in range(N):
        f_in = gp.quicksum(get_f(u, i) for u in range(N) if (u, i) in arc_to_idx)
        f_out = gp.quicksum(get_f(i, v) for v in range(N) if (i, v) in arc_to_idx)

        if i == depot_id:
            # Net flow OUT of depot must equal minus total demand of clusters
            model.addLConstr(f_out - f_in == -total_demand, name="DepotFlow")
        else:
            # flow picked up at node i = demand of cluster of node i
            # only if node i is visited

            x_sum = gp.quicksum(get_x(u, i) for u in range(N) if (u, i) in arc_to_idx) + \
                    gp.quicksum(get_x(i, v) for v in range(N) if (i, v) in arc_to_idx)

            demand = q_cluster[a[i]]
            model.addLConstr(f_out - f_in == 0.5 * demand * x_sum, name=f"Demand_{i}") #(exmp: 0.5 * customer_demand * 2 == customer_demand)

    # C6: Capacity Constraints (Strengthened)
    # q_i <= f_ij <= (Q - q_j)
    # f_ij is the load AFTER visiting i (so it includes q_i)
    # and BEFORE visiting j (so it has space for q_j)
    for (u, v) in arc_list:
        if u == v: continue

        x_var = get_x(u, v)
        f_var = get_f(u, v)
        dem_u = q_cluster[a[u]]
        dem_v = q_cluster[a[v]]

        # lb -> load must be at least what the truck picked up at u
        model.addLConstr(f_var >= dem_u * x_var, name=f"CapLow_{u}_{v}")

        # up -> load must have space for what the truck will pick up at v
        model.addLConstr(f_var <= (Q - dem_v) * x_var, name=f"CapUp_{u}_{v}")

    # 6. Final Setup
    model.ModelSense = GRB.MINIMIZE
    #model.Params.method = 1 # Dual Simplex
    model.update()

    return model, ub, lb, integer_var, num_vars, c, arc_list, arc_to_idx
