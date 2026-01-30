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
    # Strip whitespace from column names (e.g., ' time_bnb ' -> 'time_bnb')
    df.columns = df.columns.str.strip()

    # Define the columns we want to convert
    cols_to_convert = [
        'time_bnb', 'time_bnb_no_cuts', 'time_gurobi',
        'nodes_expl_bnb', 'nodes_expl_bnb_no_cuts', 'nodes_expl_gurobi'
    ]

    # Convert to numeric, turning errors (like 'x' or strings) into NaN
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in CSV.")

    # 3. Create First Diagram: Time Comparison
    plt.figure(figsize=(10, 6))

    # Plot Custom BnB (Cuts)
    plt.plot(df['pr_num'], df['time_bnb'],
             marker='o', linestyle='-', color='green', label='Custom BnB (Cuts)')

    # Plot Simple BnB (No Cuts) - NEW
    if 'time_bnb_no_cuts' in df.columns:
        plt.plot(df['pr_num'], df['time_bnb_no_cuts'],
                 marker='^', linestyle='--', color='magenta', label='Simple BnB (No Cuts)')

    # Plot Gurobi
    plt.plot(df['pr_num'], df['time_gurobi'],
             marker='s', linestyle='-.', color='orange', label='Gurobi')

    plt.xlabel('Problem Number')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)

    if not df['pr_num'].empty:
        plt.xticks(df['pr_num'])

    plt.tight_layout()
    plt.savefig('./results/time_comparison.png')
    print("Created 'time_comparison.png'")
    plt.close()

    # 4. Create Second Diagram: Nodes Explored Comparison
    plt.figure(figsize=(10, 6))

    # Plot Custom BnB (Cuts)
    plt.plot(df['pr_num'], df['nodes_expl_bnb'],
             marker='o', linestyle='-', color='green', label='Custom BnB (Cuts)')

    # Plot Simple BnB (No Cuts) - NEW
    if 'nodes_expl_bnb_no_cuts' in df.columns:
        plt.plot(df['pr_num'], df['nodes_expl_bnb_no_cuts'],
                 marker='^', linestyle='--', color='magenta', label='Simple BnB (No Cuts)')

    # Plot Gurobi
    plt.plot(df['pr_num'], df['nodes_expl_gurobi'],
             marker='s', linestyle='-.', color='orange', label='Gurobi')

    plt.xlabel('Problem Number')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored Comparison')
    plt.legend()
    plt.grid(True)

    if not df['pr_num'].empty:
        plt.xticks(df['pr_num'])

    plt.tight_layout()
    plt.savefig('./results/renodes_comparison.png')
    print("Created 'nodes_comparison.png'")
    plt.close()

if __name__ == "__main__":
    create_diagrams('./results/results.csv')
