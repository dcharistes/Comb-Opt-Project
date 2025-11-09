# cvrp_rand_gen.py
import math
import random
import numpy as np


def generate_random_cvrp(n_customers=20, coord_range=(0, 100),
                         demand_range=(1, 10), seed=None,
                         capacity=None, K=None):
    """
    Generate a random CVRP instance.

    Returns a dictionary with:
      - coords: list of (x, y) coordinates, depot = 0
      - demands: dict {i: demand}
      - Q: vehicle capacity
      - K: number of vehicles
      - cost: dict {(i, j): distance}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = n_customers
    coords = [(random.uniform(*coord_range),
               random.uniform(*coord_range)) for _ in range(n + 1)]
    demands = {0: 0}
    for i in range(1, n + 1):
        demands[i] = random.randint(*demand_range)

    total_demand = sum(demands.values())

    if capacity is None:
        avg_dem = (demand_range[0] + demand_range[1]) / 2.0
        capacity = max(1, int(round((n * avg_dem) / max(1, int(math.sqrt(n))))))
    if K is None:
        K = int(math.ceil(total_demand / capacity))

    cost = {}
    for i in range(n + 1):
        xi, yi = coords[i]
        for j in range(n + 1):
            xj, yj = coords[j]
            cost[(i, j)] = 0.0 if i == j else math.hypot(xi - xj, yi - yj)

    instance = {
        "n_customers": n,
        "coords": coords,
        "demands": demands,
        "Q": capacity,
        "K": K,
        "cost": cost,
    }
    return instance


def generate_multiple_instances(n_classes=10, problems_per_class=10,
                                n_customers=20, output_prefix="instances",
                                base_seed=42):
    """
    Generate multiple classes of random CVRP problems.
    Each class uses a different seed and capacity scaling.
    """
    instances = []
    for c in range(n_classes):
        for p in range(problems_per_class):
            seed = base_seed + 100 * c + p
            inst = generate_random_cvrp(
                n_customers=n_customers,
                coord_range=(0, 100),
                demand_range=(1, 10),
                seed=seed,
                capacity=None,
            )
            instances.append(inst)
    return instances


if __name__ == "__main__":
    # quick test
    inst = generate_random_cvrp(n_customers=12, seed=1)
    print("Random CVRP instance generated:")
    print(f"Customers: {inst['n_customers']}, Capacity: {inst['Q']}, Vehicles: {inst['K']}")
    print("Demands:", inst["demands"])
