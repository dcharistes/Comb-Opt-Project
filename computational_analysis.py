import os
import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Import your model logic
# Ensure gvrp_model.py is in the same directory
try:
    from gvrp_model import read_data_gvrp, build_gvrp_model
except ImportError:
    print("Error: Could not import 'gvrp_model.py'. Make sure it is in the same folder.")
    exit()

# Configuration
INSTANCES_DIR = "gvrp_instances"
OUTPUT_FILE = "computational_analysis_results.csv"
SOLVER_NAME = "gurobi"  # Or 'cbc', 'glpk'
TIME_LIMIT = 60  # Seconds per instance

def run_analysis():
    results = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(INSTANCES_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                # Extract metadata from folder/filename
                # Folder format: Class_01_5x5
                folder_name = os.path.basename(root)
                parts = folder_name.split('_')
                class_id = parts[1]
                grid_dim = parts[2]

                # File format: Dense_0.txt
                type_name = file.split('_')[0]

                print(f"Processing: Class {class_id} ({grid_dim}) - {type_name} - {file}...")

                try:
                    # 1. Read Data
                    data = read_data_gvrp(file_path)
                    (N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id, cluster_nodes) = data

                    # 2. Build Model
                    build_start = time.time()
                    model = build_gvrp_model(N, K, Q, M, q_cluster, a, arc_list, cost_param, depot_node_id, cluster_nodes)
                    build_time = time.time() - build_start

                    # 3. Solve
                    solver = SolverFactory(SOLVER_NAME)

                    # Set Time Limit (Solver specific options)
                    if SOLVER_NAME == 'gurobi':
                        solver.options['TimeLimit'] = TIME_LIMIT
                        solver.options['MIPGap'] = 0.05 # 5% Gap tolerance
                    elif SOLVER_NAME == 'cbc':
                        solver.options['seconds'] = TIME_LIMIT

                    solve_start = time.time()
                    result = solver.solve(model, tee=False) # tee=False to reduce clutter
                    solve_time = time.time() - solve_start

                    # 4. Extract Metrics
                    status = str(result.solver.status)
                    termination = str(result.solver.termination_condition)

                    obj_val = -1
                    lower_bound = -1
                    gap = -1

                    if hasattr(result.problem, 'upper_bound'):
                         obj_val = result.problem.upper_bound
                    elif hasattr(model, 'obj'):
                         obj_val = pyo.value(model.obj)

                    if hasattr(result.problem, 'lower_bound'):
                        lower_bound = result.problem.lower_bound

                    # Calculate gap manually if feasible
                    if obj_val is not None and lower_bound is not None and obj_val != 0:
                        gap = abs(obj_val - lower_bound) / abs(obj_val)

                    # Record Data
                    results.append({
                        "Class": class_id,
                        "Grid": grid_dim,
                        "Type": type_name,
                        "File": file,
                        "Nodes_N": N,
                        "Clusters_M": M,
                        "Vehicles_K": K,
                        "Build_Time_s": round(build_time, 4),
                        "Solve_Time_s": round(solve_time, 4),
                        "Objective": obj_val,
                        "Status": status,
                        "Termination": termination,
                        "Gap": gap
                    })

                except Exception as e:
                    print(f"  FAILED: {str(e)}")
                    results.append({
                        "Class": class_id,
                        "Grid": grid_dim,
                        "Type": type_name,
                        "File": file,
                        "Status": "Error",
                        "Termination": str(e)
                    })

    # Save to CSV
    df = pd.DataFrame(results)
    # Sort for better readability
    df.sort_values(by=["Class", "Type", "File"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_analysis()
