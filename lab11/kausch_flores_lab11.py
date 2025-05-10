from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



def read_data(file_name):
    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    header = lines[0].split(',')
    data = [dict(zip(header, row.split(','))) for row in lines[1:]]
    return header, data

def read_dag(dag_file):
    dag = {}
    with open(dag_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            node = parts[0].strip()
            parents = [p.strip() for p in parts[1:]] if len(parts) > 1 else []
            dag[node] = parents
    return dag

def initialize_cpts(dag):
    cpts = defaultdict(lambda: defaultdict(int))
    cpts_total = defaultdict(lambda: defaultdict(int))
    return cpts, cpts_total

def update_cpt_from_row(row, dag, cpts, cpts_total):
    for node, parents in dag.items():
        node_val = row[node]
        if parents:
            parent_vals = tuple(row[parent] for parent in parents)
            key = parent_vals + (node_val,)
            cpts[node][key] += 1
            cpts_total[node][parent_vals] += 1
        else:
            cpts[node][(node_val,)] += 1
            cpts_total[node]['__total__'] += 1

def compute_cpt_probabilities(cpts, cpts_total):
    probs = {}
    for node in cpts:
        probs[node] = {}
        for combo, count in cpts[node].items():
            parent_vals = combo[:-1] if '__total__' not in cpts_total[node] else '__total__'
            total = cpts_total[node][parent_vals]
            probs[node][combo] = count / total if total > 0 else 0
    return probs

def predict(row, dag, probs):
    def joint_prob(row_with_asthma):
        prob = 1.0
        # print(f"\nCalculating joint probability for row: {row_with_asthma}")
        for node, parents in dag.items():
            if parents:
                parent_vals = tuple(row_with_asthma[parent] for parent in parents)
                key = parent_vals + (row_with_asthma[node],)
            else:
                key = (row_with_asthma[node],)
            node_prob = probs.get(node, {}).get(key, 1e-6)  # Small epsilon to avoid zero
            prob *= node_prob
            # print(f"  Node: {node}")
            # print(f"    Parents: {parents}")
            # print(f"    Parent values: {parent_vals if parents else 'None'}")
            # print(f"    Key: {key}")
            # print(f"    Probability from CPT: {node_prob}")
            # print(f"    Cumulative joint probability: {prob}")
        return prob

    # Case 1: assume asthma = yes
    # print("\nAssuming asthma = 'yes'")
    row_yes = row.copy()
    row_yes['asthma'] = 'yes'
    yes_prob = joint_prob(row_yes)

    # Case 2: assume asthma = no
    # print("\nAssuming asthma = 'no'")
    row_no = row.copy()
    row_no['asthma'] = 'no'
    no_prob = joint_prob(row_no)

    # print()
    # print("Final Joint Probabilities:")
    # print(f"  Joint probability with asthma='yes': {yes_prob}")
    # print(f"  Joint probability with asthma='no': {no_prob}")
    # print()
    
    # Compare probabilities
    return 'yes' if yes_prob >= no_prob else 'no'

def compute_auc(accuracies, start=1, end=None):
    # Compute AUC using trapezoidal rule
    auc = 0.0
    end = end if end is not None else len(accuracies)
    for i in range(start, end):
        auc += (accuracies[i] + accuracies[i-1]) / 2
    return auc

def evaluate_predictions(data_rows, dag):
    cpts = {}
    cpts_total = {}
    cpts, cpts_total = initialize_cpts(dag)
    correct = 0
    total = 0
    accuracy_over_time = []

    for row in data_rows:
        # print(f"Row {total + 1}: {row}")
        if total > 0:
            probs = compute_cpt_probabilities(cpts, cpts_total)
            # Output the current probabilities
            # print(f"Probabilities after row {total}:")
            # for node, prob in probs.items():
            #     print(f"  {node}: {prob}")
            # Predict asthma
            prediction = predict(row, dag, probs)
            #print(f"Prediction: {prediction}, Actual: {row['asthma']}")
            # Check if the prediction is correct
            if prediction == row['asthma']:
                correct += 1
        total += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_over_time.append(accuracy)

        update_cpt_from_row(row, dag, cpts, cpts_total)
        # Output the current accuracy
        # print(f"Current accuracy: {accuracy:.2f}")
        # output the current CPTs
        # print("Current CPTs:")
        # for node, cpt in cpts.items():
        #     print(f"  {node}: {cpt}")
        # Output the current total counts
        # print("Current total counts:")
        # for node, total_counts in cpts_total.items():
        #     print(f"  {node}: {total_counts}")

        # Wait for user input to proceed (comment out for automatic execution)
        # input("Press Enter to continue...\n")

    # Compute AUC
    auc = compute_auc(accuracy_over_time, start=250)
    # print(f"AUC: {auc:.2f}")

    return accuracy_over_time, auc, cpts, cpts_total

def plot_accuracy_over_time(accuracies, title_prefix=''):
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel('Row Number')
    plt.ylabel('Accuracy')
    plt.title(f"{title_prefix} Accuracy Over Time")
    plt.grid(True)

    # Add a vertical line at x = 250
    plt.axvline(x=250, color='red', linestyle='--', label='x = 250')
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{title_prefix}.png")
    plt.show()

def plot_all_accuracies(accuracies_list, auc, title_prefix=''):
    for i, accuracies in enumerate(accuracies_list):
        plt.plot(range(1, len(accuracies)+1), accuracies, label=f'DAG {i+1}: AUC == {auc[i]:.2f}')

    # Labels
    plt.xlabel('Row Number')
    plt.ylabel('Accuracy')
    plt.title(f"{title_prefix} Accuracy Over Time")
    plt.grid(True)
    plt.legend()

    # Add a vertical line at x = 250, but do not include it in the legend
    plt.axvline(x=250, color='red', linestyle='--', label=None)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{title_prefix}_all_dags.png")
    plt.show()

def write_data_to_file(data, filename):
    with open(filename, 'w') as file:
        if isinstance(data, list):
            for item in data:
                file.write(f"{item}\n")
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        file.write(f"{key},{sub_key},{sub_value}\n")
                else:
                    file.write(f"{key},{value}\n")
        else:
            file.write(str(data) + '\n')
    print(f"Data written to {filename}")

def main():
    header, data_rows = read_data("BN_Asthma_data.csv")

    all_accuracies = []
    aucs = []

    for i in range(5, 14):
        # Read DAG
        path = f"dag{i}.txt"
        dag = read_dag(path)

        # Get accuracies, AUC, CPTs
        accuracies, auc, cpts, cpts_total = evaluate_predictions(data_rows, dag)
        all_accuracies.append(accuracies)
        aucs.append(auc)

        # Visualize and save results
        plot_accuracy_over_time(accuracies, title_prefix=f"DAG{i}")
        write_data_to_file(accuracies, f"dag{i}_accuracy.txt")
        write_data_to_file(cpts, f"dag{i}_cpts.txt")
        write_data_to_file(cpts_total, f"dag{i}_cpts_total.txt")

    # Print AUC
    print(f"AUCs for DAGs: {aucs}")

    # Plot all accuracies
    plot_all_accuracies(all_accuracies, aucs, title_prefix='All DAGs')

if __name__ == "__main__":
    main()