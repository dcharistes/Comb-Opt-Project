import random
import os
import math

# --- Configuration ---
# 10 Classes: 5x5 up to 50x50
CLASSES = range(1, 11)
GRID_INCREMENT = 5
PROBLEMS_PER_TYPE = 5  # 5 Dense, 5 Sparse
DENSE_FILL_RATIO = 0.50 # 50% of grid points are customers
SPARSE_FILL_RATIO = 0.10 # 10% of grid points are customers

# Fixed Parameters
Q = 15 # Vehicle Capacity
K_BASE = 3 # Base number of vehicles (scales slightly with V)

# Codes for grid map
EMPTY_POINT = 0
CUSTOMER_POINT = 1
DEPOT_POINT = 2

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

def generate_dataset():
    base_dir = "gvrp_instances"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print(f"Starting generation of {len(CLASSES) * 2 * PROBLEMS_PER_TYPE} instances...")

    for class_idx in CLASSES:
        grid_dim = class_idx * GRID_INCREMENT
        grid_size = grid_dim  # The grid is grid_dim x grid_dim

        # Create class directory
        class_dir = os.path.join(base_dir, f"Class_{class_idx:02d}_{grid_dim}x{grid_dim}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # We generate 5 Dense and 5 Sparse
        types = [("Dense", DENSE_FILL_RATIO), ("Sparse", SPARSE_FILL_RATIO)]

        for type_name, ratio in types:
            for p_idx in range(PROBLEMS_PER_TYPE):

                # 1. Determine V (Total Vertices including Depot)
                # Calculate max potential points
                total_cells = grid_dim * grid_dim

                # Number of customers
                n_customers = int(total_cells * ratio)
                # Ensure at least 4 customers
                n_customers = max(4, n_customers)

                V = n_customers + 1 # +1 for Depot

                # 2. Determine M (Clusters)
                # Heuristic: Average 3 nodes per cluster
                num_clusters = max(2, V // 3)

                # 3. Determine K (Vehicles)
                # Heuristic: Scale K based on total customers and capacity
                # Estimate total demand approx (0.5 * Q * customers)
                # Needed vehicles = Total Demand / (0.8 * Q) -> allowing slack
                K = max(K_BASE, int((n_customers * (Q/2)) / (Q * 0.8)))

                # --- Generation Logic (Adapted from your script) ---

                # Init Grid
                grid = [[0 for _ in range(grid_dim)] for _ in range(grid_dim)]
                points = [Point() for _ in range(V)]

                # Add Depot (Points[0])
                depot_x = random.randint(0, grid_dim - 1)
                depot_y = random.randint(0, grid_dim - 1)
                points[0].x = depot_x
                points[0].y = depot_y
                grid[depot_x][depot_y] = DEPOT_POINT

                # Add Customers (Points[1...V-1])
                added_count = 0
                while added_count < n_customers:
                    rx = random.randint(0, grid_dim - 1)
                    ry = random.randint(0, grid_dim - 1)
                    if grid[rx][ry] == EMPTY_POINT:
                        grid[rx][ry] = CUSTOMER_POINT
                        points[added_count + 1].x = rx
                        points[added_count + 1].y = ry
                        added_count += 1

                # Create Clusters
                clusters = dict()
                clusters[0] = [0] # Depot cluster

                cust_indices = list(range(1, V))
                random.shuffle(cust_indices)

                for c in range(1, num_clusters + 1):
                    clusters[c] = []

                # Distribute customers
                for i, cust_idx in enumerate(cust_indices):
                    cid = (i % num_clusters) + 1
                    clusters[cid].append(cust_idx)

                # Connectivity (Arcs)
                # Note: This is an O(N^2) operation. For 50x50 dense, this might be slow.
                arcs = []
                # Simple optimization: we only need arcs between DIFFERENT clusters
                # Your model seems to support fully connected between clusters
                cluster_ids = sorted(clusters.keys())

                for cid_a in cluster_ids:
                    nodes_a = clusters[cid_a]
                    for cid_b in cluster_ids:
                        if cid_a == cid_b: continue

                        nodes_b = clusters[cid_b]
                        for u in nodes_a:
                            for v in nodes_b:
                                dist = abs(points[u].x - points[v].x) + abs(points[u].y - points[v].y)
                                arcs.append((u, v, dist))

                # Demands
                demands = [] # Tuples (cid, demand)
                for cid in range(1, num_clusters + 1):
                    d_val = random.randint(1, Q) # Demand between 1 and Q
                    demands.append((cid, d_val))

                demand_dict = {cid: d for cid, d in demands}

                # --- Save to Disk ---
                filename = f"{type_name}_{p_idx}.txt"
                filepath = os.path.join(class_dir, filename)

                with open(filepath, "w") as f:
                    # Header: GridSize V M K Q
                    f.write(f"{grid_dim*grid_dim} {V} {num_clusters} {K} {Q}\n")
                    f.write(f"{num_clusters} Clusters:\n")

                    for cid, member_nodes in clusters.items():
                        # Use first node of cluster for 'center' coords if needed,
                        # but your parser reads coords per node line anyway.
                        # The standard format seems to be: ClusterID X Y Demand
                        # Wait, your parser expects: "cluster_id x y demand"
                        # But multiple nodes can be in one cluster.
                        # Your parser logic: `if cluster_id not in cluster_nodes... append`
                        # So we print one line per NODE.

                        d_val = 0 if cid == 0 else demand_dict[cid]
                        for node_idx in member_nodes:
                            p = points[node_idx]
                            f.write(f"{cid} {p.x} {p.y} {d_val}\n")

                    f.write(f"{len(arcs)} Arcs:\n")
                    for u, v, dist in arcs:
                        pu = points[u]
                        pv = points[v]
                        f.write(f"{pu.x} {pu.y} {pv.x} {pv.y} {dist}\n")

                print(f"Generated: {filepath} (V={V}, M={num_clusters})")

if __name__ == "__main__":
    generate_dataset()
