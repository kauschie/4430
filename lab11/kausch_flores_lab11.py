from collections import defaultdict
import matplotlib.pyplot as plt

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
    yes_prob = 1.0
    no_prob = 1.0

    for node, parents in dag.items():
        if node != 'asthma':
            continue  # We only want to predict asthma

        parent_vals = tuple(row[parent] for parent in parents)

        yes_key = parent_vals + ('yes',)
        no_key = parent_vals + ('no',)

        yes_prob *= probs.get(node, {}).get(yes_key, 0.5)
        no_prob *= probs.get(node, {}).get(no_key, 0.5)

    return 'yes' if yes_prob >= no_prob else 'no'

def evaluate_predictions(data_rows, dag):
    cpts, cpts_total = initialize_cpts(dag)
    correct = 0
    total = 0
    accuracy_over_time = []

    for row in data_rows:
        if total > 0:
            probs = compute_cpt_probabilities(cpts, cpts_total)
            prediction = predict(row, dag, probs)
            if prediction == row['asthma']:
                correct += 1
        total += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_over_time.append(accuracy)

        update_cpt_from_row(row, dag, cpts, cpts_total)

    return accuracy_over_time

def plot_accuracy_over_time(accuracies):
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel('Row Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Accuracy Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    header, data_rows = read_data("BN_Asthma_data.csv")
    dag = read_dag("dag.txt")

    accuracies = evaluate_predictions(data_rows, dag)
    plot_accuracy_over_time(accuracies)

if __name__ == "__main__":
    main()