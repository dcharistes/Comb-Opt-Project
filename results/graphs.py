import pandas as pd
import matplotlib.pyplot as plt

def create_diagrams(csv_file):
    # 1. Load the data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # 2. Data Cleaning
    # Convert columns to numeric, turning errors (like 'x' or strings) into NaN (Not a Number)
    # This prevents the script from crashing on incomplete data entries
    cols_to_convert = ['time_bnb', 'time_gurobi', 'nodes_expl_bnb', 'nodes_expl_gurobi']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Create First Diagram: Time Comparison
    plt.figure(figsize=(10, 6))

    # Plot BnB Time
    plt.plot(df['pr_num'], df['time_bnb'],
             marker='o', linestyle='-', color='blue', label='Custom BnB')

    # Plot Gurobi Time
    plt.plot(df['pr_num'], df['time_gurobi'],
             marker='s', linestyle='--', color='red', label='Gurobi')

    plt.xlabel('Problem Number')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)

    # Ensure x-axis only shows integer problem numbers
    if not df['pr_num'].empty:
        plt.xticks(df['pr_num'])

    plt.tight_layout()
    plt.savefig('./results/time_comparison.png')
    print("Created 'time_comparison.png'")
    plt.close() # Close figure to free memory

    # 4. Create Second Diagram: Nodes Explored Comparison
    plt.figure(figsize=(10, 6))

    # Plot BnB Nodes
    plt.plot(df['pr_num'], df['nodes_expl_bnb'],
             marker='o', linestyle='-', color='green', label='Custom BnB')

    # Plot Gurobi Nodes
    plt.plot(df['pr_num'], df['nodes_expl_gurobi'],
             marker='s', linestyle='--', color='orange', label='Gurobi')

    plt.xlabel('Problem Number')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored Comparison')
    plt.legend()
    plt.grid(True)

    if not df['pr_num'].empty:
        plt.xticks(df['pr_num'])

    plt.tight_layout()
    plt.savefig('./results/nodes_comparison.png')
    print("Created 'nodes_comparison.png'")
    plt.close()

if __name__ == "__main__":
    create_diagrams('./results/results.csv')
