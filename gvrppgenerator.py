import random
import numpy
import argparse
import os



V=0
D=0
K=0 #num of vehicles
num_clusters=0
Q=15 #capacity of vehicle
grid_size=0
CUSTOMER_POINT=1
DEPOT_POINT=2
num_clusters=0
PR=0

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

#function to add the Depot
def add_depot(grid, points):
    print("--Adding the Depot to the grid (step 1)")

    points[0].x = random.randint(0, grid_size - 1)
    points[0].y = random.randint(0, grid_size - 1)

    grid[points[0].x][points[0].y] = DEPOT_POINT


    return 0


# Function to add Customers to grid 
def add_customers(grid, points):
    print("--Adding Customers to the grid (step 2)")

    i = 1  
    while i < V:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)

        if grid[x][y] == 0:
            points[i].x = x
            points[i].y = y
            grid[x][y] = CUSTOMER_POINT
            i += 1


def create_clusters(points, depot_index, num_clusters):
    print("--Creating Clusters (step 3)")

    clusters = dict()
    clusters[0] = [depot_index]  # C0 περιέχει μόνο το depot

    # Πάρε όλους τους υπόλοιπους πελάτες
    customer_indices = [i for i in range(len(points)) if i != depot_index]

    # Ανακάτεψέ τους τυχαία
    random.shuffle(customer_indices)

    # Δημιούργησε m clusters (C1...Cm)
    for i in range(1, num_clusters + 1):
        clusters[i] = []

    # Μοίρασε τους πελάτες στους m clusters
    for idx, customer in enumerate(customer_indices):
        cluster_id = (idx % num_clusters) + 1  # κυκλικά στους m clusters
        clusters[cluster_id].append(customer) # στο cluster1 βαζει τον customer 0...

    return clusters 

def conectivity(points, clusters):
    print("--Creating all the arcs that connect all Clusters with each other (step 4)")

    A = [] 
    cluster_ids = list(clusters.keys())

    for cid_a in cluster_ids:
        customers_a = clusters[cid_a]

        for cid_b in cluster_ids:
            if cid_a == cid_b:
                continue  # αγνοούμε αποστάσεις μέσα στο ίδιο cluster

            customers_b = clusters[cid_b]

            # Για κάθε σημείο i στο cluster A
            for i in customers_a:
                # Για κάθε σημείο j στο cluster B
                for j in customers_b:
                    xi, yi = points[i].x, points[i].y
                    xj, yj = points[j].x, points[j].y

                    # Manhattan distance
                    dist = abs(xi - xj) + abs(yi - yj)

                    # Προσθήκη στο πίνακα των arcs
                    A.append((i, j, dist))

    return A

def add_Demand(clusters):
    print("--Adding a total demand to each Cluster (step 5)")

    D = []
    cluster_ids = [cid for cid in clusters.keys() if cid != 0]

    for cid in cluster_ids:

        d = random.randint(1,Q)
        D.append((cid, d))

    return D


def save_on_disk(grid, clusters, A, D, points, filename="0"):

    folder_name = str(grid_size) + "_" + str(V) + "_" + str(num_clusters)
    file_extension = ".txt"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        file_handler = open(folder_name+"/"+filename+file_extension, "w")

        print("--Writing problems on disk... (step 6)")

        text = list()
        text.append(f"{grid_size*grid_size} {V} {num_clusters} {K} {Q}\n")

        cluster_ids = [cid for cid in clusters.keys()]
        demand_dict = {cid: d for cid, d in D}
        
        text.append(f"{num_clusters} Clusters:\n")
        
        for cid, customers in clusters.items():
            
            for cust in customers:
                x = points[cust].x
                y = points[cust].y

                if cid == 0:
                    demand_value = 0   # depot has no demand
                else:
                    demand_value = demand_dict[cid]

                text.append(f"{cid} {x} {y} {demand_value}\n")

        text.append(f"{len(A)} Arcs:\n")
        for i,j,dist in A:
            text.append(f"{points[i].x} {points[i].y} {points[j].x} {points[j].y} {dist}\n")


        # Remove last \n character
        text[-1] = text[-1][:-1]
        file_handler.write(''.join(text))
        file_handler.close()
        print(f"\nSuccesfully created file {filename+file_extension}.\n\n")
    except Exception as e:
        print(str(e))
        print(f"\nError while creating file {filename+file_extension}!\n\n")
        

# Function that handles the input arguments from the user
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generalized Vehicle Routing Problem Generator\nUsage: <grid size> <total Vertices (V)> <clusters (num_clusters))> <number of Vehicles (K)> <Capacity of each vehicle (Q)> <# problems>"
    )

    # Define each positional argument
    parser.add_argument("grid_size", type=int, help="Size of the grid (e.g., 10 for a 10x10 grid)")
    parser.add_argument("V", type=int, help="Total number of Vertices (V), basically the total number of customers - 1")
    parser.add_argument("num_clusters", type=int, help="Number of clusters (num_clusters)")
    parser.add_argument("K", type=int, help="Total number of available identical vehicles (K)")
    parser.add_argument("Q", type=int, help="Max capacity of each vehicle (Q)")
    parser.add_argument("num_problems", type=int, help="Number of problems")

    # Parse arguments
    args = parser.parse_args()

    global grid_size, V, num_clusters, K, Q, PR
    grid_size = args.grid_size
    V = args.V
    num_clusters = args.num_clusters
    K = args.K
    Q = args.Q
    PR = args.num_problems

# To use the parser
if __name__ == "__main__":

    # Handle input arguments
    args = parse_arguments()
    
    for i in range(PR):
        # Define grid
        grid = numpy.zeros(shape=(grid_size,grid_size), dtype=int)
        
        # Define points list
        points = [Point() for i in range(V)]

        depot_index = add_depot(grid, points)

        add_customers(grid, points)
        
        clusters = create_clusters(points, depot_index, num_clusters)

        A = conectivity(points, clusters)

        D = add_Demand(clusters)


        # Generate (x, y) coordinates for all points in the grid
        #point_coords = numpy.array([(i // grid_size, i % grid_size) for i in range(grid_size * grid_size)])
        


        
        # Save problem on disk
        save_on_disk(grid, clusters, A, D, points, str(i))

        
        # Delete data structures
        del grid
        del clusters
        del A
        del D
        del points

        