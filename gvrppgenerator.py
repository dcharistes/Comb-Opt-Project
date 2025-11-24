import random
import numpy
import argparse
import os



V=0 #total number of vertices
K=0 #num of vehicles
num_clusters=0 #number of customer clusters
Q=15 #capacity of vehicle
grid_size=0
CUSTOMER_POINT=1
DEPOT_POINT=2
PR=0 #number of random problems

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

#function to add the Depot
def add_depot(grid, points):
    print("--Adding the Depot to the grid (step 1)")

    # Choose random coordinates
    points[0].x = random.randint(0, grid_size - 1)
    points[0].y = random.randint(0, grid_size - 1)

    # Mark depot position on grid
    grid[points[0].x][points[0].y] = DEPOT_POINT


    return 0 #depot index


# Function to add Customers to grid 
def add_customers(grid, points):
    print("--Adding Customers to the grid (step 2)")

    i = 1  # customer indices start from 1 (0 is the depot)
    while i < V:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)

        #cell must be empty
        if grid[x][y] == 0:
            points[i].x = x
            points[i].y = y
            grid[x][y] = CUSTOMER_POINT
            i += 1


# Function that creates random clusters between the customers
def create_clusters(points, depot_index, num_clusters):
    print("--Creating Clusters (step 3)")

    clusters = dict()
    clusters[0] = [depot_index]  # C0 is the depot cluster

    # Get all customer indices
    customer_indices = [i for i in range(len(points)) if i != depot_index]

    # shuffle randomly the indices
    random.shuffle(customer_indices)

    # Initialize empty clusters (C1..Cm)
    for i in range(1, num_clusters + 1):
        clusters[i] = []

    # Assign the customers to the clusters in a cycle until every one is part of a cluster
    for idx, customer in enumerate(customer_indices):
        cluster_id = (idx % num_clusters) + 1  
        clusters[cluster_id].append(customer) 

    return clusters 


# Function that create arcs between every cluster pair and calculates the distance using the Manhattan distance
def conectivity(points, clusters):
    print("--Creating all the arcs that connect all Clusters with each other (step 4)")

    A = [] 
    cluster_ids = list(clusters.keys())

    for cid_a in cluster_ids:
        customers_a = clusters[cid_a]

        for cid_b in cluster_ids:
            if cid_a == cid_b:
                continue  # no arcs inside the same cluster

            customers_b = clusters[cid_b]

            # for every customer in cluster A
            for i in customers_a:
                # for every customer in cluster B
                for j in customers_b:
                    xi, yi = points[i].x, points[i].y
                    xj, yj = points[j].x, points[j].y

                    # Manhattan distance
                    dist = abs(xi - xj) + abs(yi - yj)

                    A.append((i, j, dist))

    return A


# Function that assign demand to clusters (not per customer)
def add_Demand(clusters):
    print("--Adding a total demand to each Cluster (step 5)")

    D = []
    cluster_ids = [cid for cid in clusters.keys() if cid != 0]

    for cid in cluster_ids:

        d = random.randint(1,Q) # cluster demand
        D.append((cid, d))

    return D


# Function that saves the instance to disk in a text file
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

    # Update global variables
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
        
        # Define points list (0 = depot)
        points = [Point() for i in range(V)]

        depot_index = add_depot(grid, points)

        add_customers(grid, points)
        
        clusters = create_clusters(points, depot_index, num_clusters)

        A = conectivity(points, clusters)

        D = add_Demand(clusters)
        
        # Save problem on disk
        save_on_disk(grid, clusters, A, D, points, str(i))

        
        # Delete data structures
        del grid
        del clusters
        del A
        del D
        del points

        