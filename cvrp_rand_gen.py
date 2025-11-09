import numpy as np
import json

def generate_random_cvrp_instance(n_customers=10, vehicle_capacity=50, n_vehicles=3, seed=42):
    np.random.seed(seed)

    # Nodes: depot (0) + customers (1..n_customers)
    N = n_customers + 1
    coords = np.random.rand(N, 2) * 100

    # Euclidean distances as costs
    c = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                c[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Random demands (customers only)
    q = np.zeros(N)
    q[1:] = np.random.randint(1, 10, size=n_customers)

    instance = {
        "N": N,
        "K": n_vehicles,
        "Q": vehicle_capacity,
        "costs": c.tolist(),
        "demands": q.tolist(),
    }

    with open("cvrp_instance.json", "w") as f:
        json.dump(instance, f, indent=4)

    print(f"✅ CVRP instance generated: {n_customers} customers, saved to cvrp_instance.json")

if __name__ == "__main__":
    generate_random_cvrp_instance()
