# cvrp_model.py
import math
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals,
    Objective, Constraint, SolverFactory, value
)
from cvrp_generator import generate_random_cvrp


def build_cvrp_pyomo(instance):
    n = instance["n_customers"]
    nodes = list(range(0, n + 1))
    demands = instance["demands"]
    Q = instance["Q"]
    K = instance["K"]
    cost = instance["cost"]

    model = ConcreteModel(name="CVRP_flow")
    model.N = Set(initialize=nodes)
    model.A = Set(initialize=[(i, j) for i in nodes for j in nodes if i != j], dimen=2)

    model.cost = Param(model.A, initialize=lambda m, i, j: float(cost[(i, j)]))
    model.demand = Param(model.N, initialize=lambda m, i: float(demands[i]))
    model.Q = Param(initialize=float(Q))
    model.K = Param(initialize=int(K))

    model.x = Var(model.A, domain=Binary)
    model.f = Var(model.A, domain=NonNegativeReals, bounds=(0, float(Q)))

    # Objective
    def obj_rule(m):
        return sum(m.cost[i, j] * m.x[i, j] for (i, j) in m.A)
    model.obj = Objective(rule=obj_rule, sense=1)

    # Degree constraints
    def out_deg_rule(m, i):
        if i == 0:
            return sum(m.x[i, j] for j in m.N if j != i) <= m.K
        return sum(m.x[i, j] for j in m.N if j != i) == 1
    model.out_deg = Constraint(model.N, rule=out_deg_rule)

    def in_deg_rule(m, i):
        if i == 0:
            return sum(m.x[j, i] for j in m.N if j != i) <= m.K
        return sum(m.x[j, i] for j in m.N if j != i) == 1
    model.in_deg = Constraint(model.N, rule=in_deg_rule)

    # Flow conservation
    total_demand = sum(demands.values())

    def flow_cons_rule(m, i):
        if i == 0:
            return (
                sum(m.f[j, 0] for j in m.N if j != 0)
                - sum(m.f[0, j] for j in m.N if j != 0)
                == -total_demand
            )
        return (
            sum(m.f[j, i] for j in m.N if j != i)
            - sum(m.f[i, j] for j in m.N if j != i)
            == m.demand[i]
        )
    model.flow_cons = Constraint(model.N, rule=flow_cons_rule)

    # Linking flow and binary
    def flow_link_rule(m, i, j):
        return m.f[i, j] <= m.Q * m.x[i, j]
    model.flow_link = Constraint(model.A, rule=flow_link_rule)

    return model


def solve_cvrp(instance, solver_name="gurobi", timelimit=60, mipgap=1e-4):
    model = build_cvrp_pyomo(instance)
    solver = SolverFactory(solver_name)
    if solver_name == "gurobi":
        solver.options["TimeLimit"] = timelimit
        solver.options["MIPGap"] = mipgap

    print("Solving CVRP instance...")
    results = solver.solve(model, tee=False)
    print(results.solver.status, results.solver.termination_condition)

    arcs = [(i, j) for (i, j) in model.A if value(model.x[i, j]) > 0.5]
    total_cost = value(model.obj)
    print(f"Total cost: {total_cost:.3f}")
    print("Arcs used:", arcs)
    return model, arcs


if __name__ == "__main__":
    inst = generate_random_cvrp(n_customers=12, seed=42)
    model, arcs = solve_cvrp(inst, solver_name="gurobi")
