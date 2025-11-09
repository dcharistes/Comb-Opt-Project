import json
from pyomo.environ import *

def build_cvrp_model(instance_file="cvrp_instance.json"):
    # Load data
    with open(instance_file, "r") as f:
        data = json.load(f)

    N = data["N"]
    V = range(N)
    A = [(i, j) for i in V for j in V if i != j]

    K = data["K"]
    Q = data["Q"]
    c = data["costs"]
    q = data["demands"]

    model = ConcreteModel()

    # Sets
    model.V = Set(initialize=V)
    model.A = Set(within=model.V * model.V, initialize=A)

    # Parameters
    model.c = Param(model.A, initialize={(i, j): c[i][j] for i, j in A})
    model.q = Param(model.V, initialize={i: q[i] for i in V})
    model.Q = Param(initialize=Q)
    model.K = Param(initialize=K)

    # Variables
    model.x = Var(model.A, within=Binary)
    model.f = Var(model.A, within=NonNegativeReals)

    # Objective
    def obj_rule(model):
        return sum(model.c[i, j] * model.x[i, j] for (i, j) in model.A)
    model.obj = Objective(rule=obj_rule, sense=minimize)

    # Constraints

    # Each customer visited once (in/out)
    def visit_out_rule(model, i):
        if i == 0:
            return Constraint.Skip
        return sum(model.x[i, j] for j in model.V if j != i) == 1
    model.visit_out = Constraint(model.V, rule=visit_out_rule)

    def visit_in_rule(model, i):
        if i == 0:
            return Constraint.Skip
        return sum(model.x[j, i] for j in model.V if j != i) == 1
    model.visit_in = Constraint(model.V, rule=visit_in_rule)

    # Exactly K vehicles depart from depot
    def depot_out_rule(model):
        return sum(model.x[0, j] for j in model.V if j != 0) == model.K
    model.depot_out = Constraint(rule=depot_out_rule)

    # Flow conservation
    def flow_conservation_rule(model, i):
        return sum(model.x[i, j] for j in model.V if j != i) == sum(model.x[j, i] for j in model.V if j != i)
    model.flow_conservation = Constraint(model.V, rule=flow_conservation_rule)

    # Flow balance (commodity flow)
    def flow_balance_rule(model, i):
        if i == 0:
            return Constraint.Skip
        lhs = sum(model.f[i, j] for j in model.V if j != i) - sum(model.f[j, i] for j in model.V if j != i)
        rhs = 0.5 * model.q[i] * (
            sum(model.x[j, i] for j in model.V if j != i) + sum(model.x[i, j] for j in model.V if j != i)
        )
        return lhs == rhs
    model.flow_balance = Constraint(model.V, rule=flow_balance_rule)

    # Capacity constraints
    def capacity_rule(model, i, j):
        return inequality(model.q[i] * model.x[i, j], model.f[i, j], (model.Q - model.q[j]) * model.x[i, j])
    model.capacity = Constraint(model.A, rule=capacity_rule)

    return model


if __name__ == "__main__":
    model = build_cvrp_model()

    solver = SolverFactory("gurobi")
    solver.solve(model, tee=True)

    print(f"Optimal objective value: {value(model.obj)}")
