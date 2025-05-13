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

def sliding_variance_reverse(data, window=5, threshold=1e-4):
    """
    Slides a window from the end of the list to the beginning,
    computing variance for each window. Returns the first index (from end)
    where the variance exceeds the threshold.

    Args:
        data (list or np.array): Time series data (e.g., accuracy, recall).
        window (int): Window size for computing variance.
        threshold (float): Variance threshold to consider the data as "unstable".

    Returns:
        int: Suggested cutoff index (start AUC from here forward)
    """
    N = len(data)
    data = np.array(data)

    for i in range(N - window, -1, -1):  # from end to start
        window_vals = data[i:i + window]
        var = np.var(window_vals)
        if var > threshold:
            return i + window  # start AUC just after stable period ends

    return 0  # if all variance is below threshold, start from beginning

def plot_accuracy_over_time(accuracies, title_prefix=''):
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel('Row Number')
    plt.ylabel('Accuracy')
    plt.title(f"{title_prefix} Accuracy Over Time")
    plt.grid(True)

    # Add a vertical line at x = 250
    plt.axvline(x=auc_start, color='red', linestyle='--', label=f'x = {auc_start}')
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{title_prefix}.png")
    plt.show()

def plot_all_accuracies(accuracies_list, auc, ylabel=None, title_prefix='', start=1, indexes=None):
    # Start refers to which DAG to start plotting from
    if not ylabel:
        ylabel = 'Accuracy'
    plt.figure(figsize=(16, 8))

    if indexes is None:
        for i, accuracies in enumerate(accuracies_list):
            plt.plot(range(1, len(accuracies)+1), accuracies, label=f'DAG {i+start}: AUC == {auc[i]:.2f}')
    else:
        for i, accuracies in enumerate(accuracies_list):
            plt.plot(range(1, len(accuracies)+1), accuracies, label=f'DAG {indexes[i]}: AUC == {auc[i]:.2f}')


    # Labels
    plt.xlabel('Row Number')

    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} Over Time")
    plt.grid(True)
    plt.legend()

    # Add a vertical line at x = 250, but do not include it in the legend
    plt.axvline(x=auc_start, color='red', linestyle='--', label=None)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{title_prefix}_all_dags.png")
    plt.show()

def plot_intersection(all_f1s, interect_pt, indexes, title_prefix='F1 Score', ylabel='F1'):

    # Start refers to which DAG to start plotting from
    plt.figure(figsize=(16, 8))

    for i, value in enumerate(all_f1s):
        plt.plot(range(1, len(value)+1), value, label=f'DAG {indexes[i]}')


    # Labels
    plt.xlabel('Row Number')
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} Over Time")
    plt.grid(True)
    plt.legend()

    # Add a vertical line at x = 250, but do not include it in the legend
    plt.axvline(x=auc_start, color='red', linestyle='--', label=None)
    plt.axvline(x=interect_pt, color='purple', linestyle=':', label=None)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{indexes[0]}x{indexes[1]}_{ylabel}.png")
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

def plot_all_metrics_with_overlay(fitted_curves, original_curves, aucs, title_prefix, ylabel, start=1, indexes=None):
    plt.figure(figsize=(16, 8))
    for i, (fit, raw) in enumerate(zip(fitted_curves, original_curves)):
        label = f"DAG {indexes[i] if indexes else i + start}: AUC = {aucs[i]:.2f}"
        x_vals = np.arange(1, len(fit) + 1)

        plt.plot(x_vals, fit, label=label)  # Fitted curve
        plt.scatter(x_vals, raw, s=10, alpha=0.4, marker='o', label=None)  # Original points

    plt.xlabel("Row Number")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} Over Time (Fitted + Raw)")
    plt.axvline(x=138, color='red', linestyle='--', label='auc_start')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_with_raw.png")
    plt.show()



auc_start = 138
# auc_start = 189
def main():
    header, data_rows = read_data("BN_Asthma_data.csv")

    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    accuracies_raw = []
    precisions_raw = []
    recalls_raw = []
    f1s_raw = []


    acc_aucs = []
    prec_aucs = []
    rec_aucs = []
    f1_aucs = []

    acc_var_indexes = []
    prec_var_indexes = []
    rec_var_indexes = []
    f1_var_indexes = []

    # list_of_good = [1]
    # list_of_good = [1,2,3,4,5]
    # list_of_good = [5, 6, 7, 12, 17]
    # list_of_good = [5, 6, 7, 12, 13, 17]
    list_of_good = [7, 13]
    # list_of_good = [i for i in range(1, 19)]

    # Loop through DAGs 5 to 13
    for i in range(1, 19):
        # Read DAG
        if i not in list_of_good:
            continue

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

        acc_var_indexes.append(sliding_variance_reverse(y_accuracies, window=5, threshold=1e-4))
        prec_var_indexes.append(sliding_variance_reverse(y_precisions, window=5, threshold=1e-4))
        rec_var_indexes.append(sliding_variance_reverse(y_recalls, window=5, threshold=1e-4))
        f1_var_indexes.append(sliding_variance_reverse(y_f1s, window=5, threshold=1e-4))
        # print(f"Cutoff index for DAG {i}: {cutoff_index}")


        # params, _ = curve_fit(log_func, x_vals, y_accuracies, p0=[0.1, 0.5])
        # y_pred_acc = log_func(x_vals, params[0], params[1])
        # acc_auc = compute_auc(y_pred_acc, start=auc_start)

        # Accuracy fit (fit and evaluate only on stable region)
        x_acc_stable = x_vals[auc_start:]
        y_acc_stable = y_accuracies[auc_start:]
        params, _ = curve_fit(log_func, x_acc_stable, y_acc_stable, p0=[0.1, 0.5])
        y_pred_acc_full = log_func(x_vals, *params)  # for plotting
        y_pred_acc_stable = log_func(x_acc_stable, *params)  # for AUC
        acc_auc = compute_auc(y_pred_acc_stable, start=0)

        print(f"Accuracy AUC for DAG {i}: {acc_auc:.2f}")

        # Plot the original data and the fitted curve
        # plt.plot(x_vals, y_accuracies, 'o', label='Original Data')
        # plt.plot(x_vals, y_pred_acc, 'r-', label='Fitted Curve')
        # plt.xlabel('Row Number')
        # plt.ylabel('Accuracy')
        # plt.title(f"Accuracy Fitted Curve for DAG {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"fitted_curve_dag_{i}.png")
        # plt.show()

        # plot the original precision data and fitted curve
        # params, _ = curve_fit(log_func, x_vals, y_precisions, p0=[0.1, 0.5])
        # y_pred_prec = log_func(x_vals, params[0], params[1])
        # prec_auc = compute_auc(y_pred_prec, start=auc_start)
        # print(f"Precision AUC for DAG {i}: {prec_auc:.2f}")

        x_prec_stable = x_vals[auc_start:]
        y_prec_stable = y_precisions[auc_start:]
        params, _ = curve_fit(log_func, x_prec_stable, y_prec_stable, p0=[0.1, 0.5])
        y_pred_prec_full = log_func(x_vals, *params)
        y_pred_prec_stable = log_func(x_prec_stable, *params)
        prec_auc = compute_auc(y_pred_prec_stable, start=0)
        print(f"Precision AUC for DAG {i}: {prec_auc:.2f}")



        # plt.plot(x_vals, y_precisions, 'o', label='Original Data')
        # plt.plot(x_vals, y_pred_prec, 'r-', label='Fitted Curve')
        # plt.xlabel('Row Number')
        # plt.ylabel('Precision')
        # plt.title(f"Precision Fitted Curve for DAG {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"fitted_curve_dag_{i}_precision.png")
        # plt.show()

        # plot the original recall data and fitted curve


        # params, _ = curve_fit(log_func, x_vals, y_recalls, p0=[0.1, 0.5])
        # y_pred_recall = log_func(x_vals, params[0], params[1])
        # rec_auc = compute_auc(y_pred_recall, start=auc_start)
        # print(f"Recall AUC for DAG {i}: {rec_auc:.2f}")

        x_rec_stable = x_vals[auc_start:]
        y_rec_stable = y_recalls[auc_start:]
        params, _ = curve_fit(log_func, x_rec_stable, y_rec_stable, p0=[0.1, 0.5])
        y_pred_recall_full = log_func(x_vals, *params)
        y_pred_recall_stable = log_func(x_rec_stable, *params)
        rec_auc = compute_auc(y_pred_recall_stable, start=0)
        print(f"Recall AUC for DAG {i}: {rec_auc:.2f}")


        # plt.plot(x_vals, y_recalls, 'o', label='Original Data')
        # plt.plot(x_vals, y_pred_recall, 'r-', label='Fitted Curve')
        # plt.xlabel('Row Number')
        # plt.ylabel('Recall')
        # plt.title(f"Recall Fitted Curve for DAG {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"fitted_curve_dag_{i}_recall.png")
        # plt.show()

        # plot the original f1 data and fitted curve
        # params, _ = curve_fit(log_func, x_vals, y_f1s, p0=[0.1, 0.5])
        # y_pred_f1 = log_func(x_vals, params[0], params[1])
        # f1_auc = compute_auc(y_pred_f1, start=auc_start)
        # print(f"F1 AUC for DAG {i}: {f1_auc:.2f}")

        x_f1_stable = x_vals[auc_start:]
        y_f1_stable = y_f1s[auc_start:]
        params, _ = curve_fit(log_func, x_f1_stable, y_f1_stable, p0=[0.1, 0.5])
        y_pred_f1_full = log_func(x_vals, *params)
        y_pred_f1_stable = log_func(x_f1_stable, *params)
        f1_auc = compute_auc(y_pred_f1_stable, start=0)
        print(f"F1 AUC for DAG {i}: {f1_auc:.2f}")

        

        # plt.plot(x_vals, y_f1s, 'o', label='Original Data')
        # plt.plot(x_vals, y_pred_f1, 'r-', label='Fitted Curve')
        # plt.xlabel('Row Number')
        # plt.ylabel('F1')
        # plt.title(f"F1 Fitted Curve for DAG {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"fitted_curve_dag_{i}_f1.png")
        # plt.show()
        
        # all_accuracies.append(y_pred_acc)
        # all_precisions.append(y_pred_prec)
        # all_recalls.append(y_pred_recall)
        # all_f1s.append(y_pred_f1)

        all_accuracies.append(y_pred_acc_full)
        all_precisions.append(y_pred_prec_full)
        all_recalls.append(y_pred_recall_full)
        all_f1s.append(y_pred_f1_full)

        accuracies_raw.append(y_accuracies)
        precisions_raw.append(y_precisions)
        recalls_raw.append(y_recalls)
        f1s_raw.append(y_f1s)



        acc_aucs.append(acc_auc)
        prec_aucs.append(prec_auc)
        rec_aucs.append(rec_auc)
        f1_aucs.append(f1_auc)

        # Visualize and save results (uncomment to see each individual plot)
        # plot_accuracy_over_time(accuracies, title_prefix=f"DAG{i}")
        write_data_to_file(accuracies, f"dag{i}_accuracy.txt")
        write_data_to_file(precisions, f"dag{i}_precision.txt")
        write_data_to_file(recalls, f"dag{i}_recall.txt")
        write_data_to_file(f1s, f"dag{i}_f1.txt")
        write_data_to_file(cpts, f"dag{i}_cpts.txt")
        write_data_to_file(cpts_total, f"dag{i}_cpts_total.txt")

    # Print AUC
    # print(f"AUCs for DAGs: {acc_aucs}")
    max_index = 0

    for i in range(len(acc_var_indexes)):
        if acc_var_indexes[i] > max_index:
            max_index = acc_var_indexes[i]
        if prec_var_indexes[i] > max_index:
            max_index = prec_var_indexes[i]
        if rec_var_indexes[i] > max_index:
            max_index = rec_var_indexes[i]
        if f1_var_indexes[i] > max_index:
            max_index = f1_var_indexes[i]

    print(f"Max index for all DAGs: {max_index}")

    # Plot all accuracies (start from DAG 5)
    plot_all_accuracies(all_accuracies, acc_aucs, title_prefix='All DAG Accuracy', start = 1, indexes = list_of_good, ylabel='Accuracy')
    # Plot all precisions
    plot_all_accuracies(all_precisions, prec_aucs, title_prefix='All DAG Precision', start = 1, indexes = list_of_good, ylabel='Precision')
    # Plot all recalls
    plot_all_accuracies(all_recalls, rec_aucs, title_prefix='All DAG Recall', start = 1, indexes = list_of_good, ylabel='Recall')
    # Plot all f1s
    plot_all_accuracies(all_f1s, f1_aucs, title_prefix='All DAG F1', start = 1, indexes = list_of_good, ylabel='F1')

    # Plot all accuracies with overlay
    plot_all_metrics_with_overlay(all_accuracies, [np.array(a) for a in accuracies_raw], acc_aucs, title_prefix='Accuracy', ylabel='Accuracy', indexes=list_of_good)
    plot_all_metrics_with_overlay(all_precisions, [np.array(p) for p in precisions_raw], prec_aucs, title_prefix='Precision', ylabel='Precision', indexes=list_of_good)
    plot_all_metrics_with_overlay(all_recalls, [np.array(r) for r in recalls_raw], rec_aucs, title_prefix='Recall', ylabel='Recall', indexes=list_of_good)
    plot_all_metrics_with_overlay(all_f1s, [np.array(f) for f in f1s_raw], f1_aucs, title_prefix='F1 Score', ylabel='F1', indexes=list_of_good)
    
    if len(list_of_good) == 2:
        interect_pt = -1
        for i in range(len(all_f1s[0])):
            if all_f1s[0][i] > all_f1s[1][i]:
                interect_pt = i
                break

        print(f"Intersection point for DAG {list_of_good[0]} and {list_of_good[1]}: {interect_pt}")
        plot_intersection(all_f1s, interect_pt, indexes=list_of_good, title_prefix='F1 Score', ylabel='F1', )

    for i in range(len(list_of_good)):
        print(f"\nDAG {list_of_good[i]}:")
        print(f"\tFinal Accuracy: {sum(all_accuracies[i][-5:-1])/5}")
        print(f"\tFinal Precision: {sum(all_precisions[i][-5:-1])/5}")
        print(f"\tFinal Recall: {sum(all_recalls[i][-5:-1])/5}")
        print(f"\tFinal F1: {sum(all_f1s[i][-5:-1]) / 5}")

if __name__ == "__main__":
    main()