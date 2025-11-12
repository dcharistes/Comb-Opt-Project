import random
import numpy
import argparse
import os

V=0
CU=0
CL=0
D=0
K=0 #num of vehicles
grid_size=0
CUSTOMER_POINT=1
DEPOT_POINT=2
num_clusters=0

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
        clusters[cluster_id].append(customer)

    return clusters    
