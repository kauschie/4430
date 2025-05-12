from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.use('TkAgg')
DEBUG = False

######
# File I/O
# ######

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


###########
# Cross-Entropy Method for Logarithmic Curve Fitting
# ###########

def logistic(x, k, x0):
    L = 1.0  # Max value (accuracy cap)
    return L / (1 + np.exp(-k * (x - x0)))

def log_func(x, a, b):
    return a * np.log(x + 1e-6) + b  # add small value to avoid log(0)


###########
# Bayesian Network Functions
# ###########


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
            if DEBUG:
                print(f"  Node: {node}")
                print(f"    Parents: {parents}")
                print(f"    Parent values: {parent_vals if parents else 'None'}")
                print(f"    Key: {key}")
                print(f"    Probability from CPT: {node_prob}")
                print(f"    Cumulative joint probability: {prob}")
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
    cpts, cpts_total = initialize_cpts(dag)
    correct = 0
    total = 0

    # Counts for binary classification
    TP = FP = FN = TN = 0

    accuracy_over_time = []
    precision_over_time = []
    recall_over_time = []
    f1_over_time = []

    for row in data_rows:
        if total > 0:
            probs = compute_cpt_probabilities(cpts, cpts_total)
            prediction = predict(row, dag, probs)
            actual = row['asthma']
            
            if prediction == actual:
                correct += 1

            # Confusion matrix logic
            if actual == 'yes':
                if prediction == 'yes':
                    TP += 1
                else:
                    FN += 1
            else:  # actual == 'no'
                if prediction == 'yes':
                    FP += 1
                else:
                    TN += 1

            # Compute metrics with small epsilon to avoid division by zero
            precision = TP / (TP + FP + 1e-6)
            recall    = TP / (TP + FN + 1e-6)
            f1_score  = 2 * precision * recall / (precision + recall + 1e-6)
        else:
            # First point, no predictions yet
            precision = recall = f1_score = 0

        total += 1
        accuracy = correct / total if total > 0 else 0

        # Append metrics
        accuracy_over_time.append(accuracy)
        precision_over_time.append(precision)
        recall_over_time.append(recall)
        f1_over_time.append(f1_score)

        update_cpt_from_row(row, dag, cpts, cpts_total)

    # # Compute AUC (same as before)
    # auc = compute_auc(accuracy_over_time, start=250)

    return accuracy_over_time, cpts, cpts_total, precision_over_time, recall_over_time, f1_over_time


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

def plot_all_accuracies(accuracies_list, auc, title_prefix='', start=1):
    # Start refers to which DAG to start plotting from
    for i, accuracies in enumerate(accuracies_list):
        plt.plot(range(1, len(accuracies)+1), accuracies, label=f'DAG {i+start}: AUC == {auc[i]:.2f}')

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
    all_precisions = []
    all_recalls = []
    all_f1s = []
    aucs = []

    # Loop through DAGs 5 to 13
    for i in range(1, 15):
        # Read DAG
        path = f"dag{i}.txt"
        dag = read_dag(path)

        # Get accuracies, AUC, CPTs
        accuracies, cpts, cpts_total, precisions, recalls, f1s = evaluate_predictions(data_rows, dag)
        

        y_accuracies = np.array(accuracies)
        y_precisions = np.array(precisions)
        y_recalls = np.array(recalls)
        y_f1s = np.array(f1s)
        x_vals = np.arange(1, len(recalls) + 1)

        # Fit the logistic function to the data

        params, _ = curve_fit(log_func, x_vals, y_accuracies, p0=[0.1, 0.5])
        y_pred = log_func(x_vals, params[0], params[1])
        auc = compute_auc(y_pred, start=1)

        print(f"Accuracy AUC for DAG {i}: {auc:.2f}")

        # Plot the original data and the fitted curve
        plt.plot(x_vals, y_accuracies, 'o', label='Original Data')
        plt.plot(x_vals, y_pred, 'r-', label='Fitted Curve')
        plt.xlabel('Row Number')
        plt.ylabel('Accuracy')
        plt.title(f"Accuracy Fitted Curve for DAG {i}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"fitted_curve_dag_{i}.png")
        plt.show()

        # plot the original precision data and fitted curve
        params, _ = curve_fit(log_func, x_vals, y_precisions, p0=[0.1, 0.5])
        y_pred = log_func(x_vals, params[0], params[1])
        auc = compute_auc(y_pred, start=1)
        print(f"Precision AUC for DAG {i}: {auc:.2f}")
        plt.plot(x_vals, y_precisions, 'o', label='Original Data')
        plt.plot(x_vals, y_pred, 'r-', label='Fitted Curve')
        plt.xlabel('Row Number')
        plt.ylabel('Precision')
        plt.title(f"Precision Fitted Curve for DAG {i}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"fitted_curve_dag_{i}_precision.png")
        plt.show()

        # plot the original recall data and fitted curve
        params, _ = curve_fit(log_func, x_vals, y_recalls, p0=[0.1, 0.5])
        y_pred = log_func(x_vals, params[0], params[1])
        auc = compute_auc(y_pred, start=1)
        print(f"Recall AUC for DAG {i}: {auc:.2f}")
        plt.plot(x_vals, y_recalls, 'o', label='Original Data')
        plt.plot(x_vals, y_pred, 'r-', label='Fitted Curve')
        plt.xlabel('Row Number')
        plt.ylabel('Recall')
        plt.title(f"Recall Fitted Curve for DAG {i}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"fitted_curve_dag_{i}_recall.png")
        plt.show()

        # plot the original f1 data and fitted curve
        params, _ = curve_fit(log_func, x_vals, y_f1s, p0=[0.1, 0.5])
        y_pred = log_func(x_vals, params[0], params[1])
        auc = compute_auc(y_pred, start=1)
        print(f"F1 AUC for DAG {i}: {auc:.2f}")
        plt.plot(x_vals, y_f1s, 'o', label='Original Data')
        plt.plot(x_vals, y_pred, 'r-', label='Fitted Curve')
        plt.xlabel('Row Number')
        plt.ylabel('F1')
        plt.title(f"F1 Fitted Curve for DAG {i}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"fitted_curve_dag_{i}_f1.png")
        plt.show()
        
        all_accuracies.append(accuracies)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_f1s.append(f1s)
        aucs.append(auc)

        # Visualize and save results (uncomment to see each individual plot)
        # plot_accuracy_over_time(accuracies, title_prefix=f"DAG{i}")
        write_data_to_file(accuracies, f"dag{i}_accuracy.txt")
        write_data_to_file(precisions, f"dag{i}_precision.txt")
        write_data_to_file(recalls, f"dag{i}_recall.txt")
        write_data_to_file(f1s, f"dag{i}_f1.txt")
        write_data_to_file(cpts, f"dag{i}_cpts.txt")
        write_data_to_file(cpts_total, f"dag{i}_cpts_total.txt")

    # Print AUC
    # print(f"AUCs for DAGs: {aucs}")

    # Plot all accuracies (start from DAG 5)
    plot_all_accuracies(all_accuracies, aucs, title_prefix='All DAG', start = 1)
    # Plot all precisions
    plot_all_accuracies(all_precisions, aucs, title_prefix='All DAG Precision', start = 1)
    # Plot all recalls
    plot_all_accuracies(all_recalls, aucs, title_prefix='All DAG Recall', start = 1)
    # Plot all f1s
    plot_all_accuracies(all_f1s, aucs, title_prefix='All DAG F1', start = 1)

if __name__ == "__main__":
    main()