import argparse
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import tqdm

# ------------------ Data Functions ------------------

def import_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            features = list(map(float, parts[:-1]))
            label = parts[-1]  # keep label as string for generality
            data.append(features + [label])
    return data

def k_fold_cross_validation(data, k):
    random.shuffle(data)
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        if i < k - 1:
            end = (i + 1) * fold_size
        else:
            end = len(data)
        folds.append(data[start:end])

    fold_paths, train_paths = [], []

    for i, fold in enumerate(folds):
        path = f'fold_{i}.txt'
        fold_paths.append(path)
        with open(path, 'w') as f:
            for item in fold:
                features = item[:-1]  # ❗ Remove label for fold files
                f.write(",".join(map(str, features)) + "\n")

    for i in range(k):
        path = f'train_{i}.txt'
        train_paths.append(path)
        with open(path, 'w') as f:
            for j, fold in enumerate(folds):
                if i != j:
                    for item in fold:
                        f.write(",".join(map(str, item)) + "\n")  # Full data including label

    print(f"{k} folds written to files.")
    return folds, fold_paths, train_paths



def run_magic_code(magic_number, known_path, unknown_path):
    result = subprocess.run(['python3', 'MagicCode.py', str(magic_number), known_path, unknown_path], capture_output=True, text=True)
    if result.returncode != 0:
        # print("Error running MagicCode.py")
        # print(f"Return code: {result.returncode}")
        # print(f"Error message: {result.stderr}")
        # print(f"Output:\n{result.stdout}\n")
        # print(f"Error running MagicCode.py with magic number {magic_number}")
        return None
    lines = result.stdout.strip().splitlines()
    return lines

# ------------------ Metrics Functions ------------------

def get_metrics_multiclass(labels, test_data, class_list):
    epsilon = 1e-20
    tp_dict = {c: 0 for c in class_list}
    fp_dict = {c: 0 for c in class_list}
    fn_dict = {c: 0 for c in class_list}

    correct = 0
    for i in range(len(labels)):
        y_hat = labels[i]
        y_actual = test_data[i][-1]

        if y_hat == y_actual:
            correct += 1

        for c in class_list:
            if y_hat == c and y_actual == c:
                tp_dict[c] += 1
            elif y_hat == c and y_actual != c:
                fp_dict[c] += 1
            elif y_hat != c and y_actual == c:
                fn_dict[c] += 1

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for c in class_list:
        precision = tp_dict[c] / (tp_dict[c] + fp_dict[c] + epsilon)
        recall = tp_dict[c] / (tp_dict[c] + fn_dict[c] + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        per_class_precision[c] = precision
        per_class_recall[c] = recall
        per_class_f1[c] = f1

    macro_precision = sum(per_class_precision.values()) / len(class_list)
    macro_recall = sum(per_class_recall.values()) / len(class_list)
    macro_f1 = sum(per_class_f1.values()) / len(class_list)
    accuracy = correct / len(labels)

    return macro_f1, accuracy, macro_precision, macro_recall, per_class_f1

# ------------------ Main Program ------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-class Magic Number Optimizer")
    parser.add_argument('--known', type=str, required=True, help='Path to known labeled data CSV')
    parser.add_argument('--unknown', type=str, required=True, help='Path to unknown unlabeled data CSV')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--tests', type=int, default=1, help='Number of repeated cross-validation tests')

    args = parser.parse_args()

    known_data = import_data(args.known)
    unknown_data_path = args.unknown
    k = args.folds
    num_tests = args.tests

    # Identify class labels dynamically
    class_list = sorted(set([row[-1] for row in known_data]))

    # Track overall test results
    avg_macro_f1_all_tests = []
    avg_accuracy_all_tests = []

    progress = tqdm.tqdm(total=120*num_tests, desc="Running Tests", dynamic_ncols=True)

    for test in range(1, num_tests + 1):
        folds, fold_paths, train_paths = k_fold_cross_validation(known_data, k)

        avg_macro_f1 = []
        avg_accuracy = []

        for magic_number in range(1, 121):
            f1s = []
            accuracies = []

            for j, fp in enumerate(fold_paths):
                result = run_magic_code(magic_number, train_paths[j], fp)
                if result:
                    macro_f1, accuracy, macro_precision, macro_recall, per_class_f1 = get_metrics_multiclass(result, folds[j], class_list)
                    f1s.append(macro_f1)
                    accuracies.append(accuracy)
                else:
                    f1s.append(0)
                    accuracies.append(0)

            avg_macro_f1.append(sum(f1s) / len(f1s))
            avg_accuracy.append(sum(accuracies) / len(accuracies))
            progress.update(1)

        avg_macro_f1_all_tests.append(avg_macro_f1)
        avg_accuracy_all_tests.append(avg_accuracy)

    progress.close()

    # Convert to arrays
    avg_macro_f1_all_tests = np.array(avg_macro_f1_all_tests)
    avg_accuracy_all_tests = np.array(avg_accuracy_all_tests)

    mean_macro_f1 = np.mean(avg_macro_f1_all_tests, axis=0)
    mean_accuracy = np.mean(avg_accuracy_all_tests, axis=0)

    # Pick best magic number based on highest macro F1
    best_magic_number = np.argmax(mean_macro_f1) + 1
    best_f1 = mean_macro_f1[best_magic_number - 1]
    best_acc = mean_accuracy[best_magic_number - 1]

    print("\n=== Best Magic Number Selected ===")
    print(f"Best magic number: {best_magic_number}")
    print(f"Mean Macro F1 Score: {best_f1:.4f}")
    print(f"Mean Accuracy: {best_acc:.4f}")

    # Optionally, run on unknown data now
    result_on_unknown = run_magic_code(best_magic_number, args.known, args.unknown)
    if result_on_unknown:
        with open("unknown_predictions.csv", "w") as f_out:
            for pred in result_on_unknown:
                f_out.write(f"{pred}\n")
        print("\nPredictions on unknown data saved to 'unknown_predictions.csv'.")

    # Find top 3 magic numbers based on mean macro F1
    top_3_indices = np.argsort(mean_macro_f1)[-3:][::-1]  # Indices of top 3 (descending order)

    print("\n=== Top 3 Magic Numbers (based on Macro-F1) ===")
    for rank, idx in enumerate(top_3_indices, start=1):
        magic_number = idx + 1
        print(f"\n--- Rank {rank} ---")
        print(f"Magic Number: {magic_number}")
        print(f"Mean Macro F1: {mean_macro_f1[idx]:.4f}")
        print(f"Mean Accuracy: {mean_accuracy[idx]:.4f}")

        # Now re-run MagicCode for this magic number on the training data to get per-class F1
        # (using full known data for evaluation if you want real per-class)
        result = run_magic_code(magic_number, args.known, args.known)  # classify known against itself
        if result:
            _, _, _, _, per_class_f1 = get_metrics_multiclass(result, known_data, class_list)
            print("Per-class F1 scores:")
            for label in sorted(class_list):
                print(f"  Class {label}: F1 = {per_class_f1[label]:.4f}")
        else:
            print("  Error running MagicCode for evaluation!")

if __name__ == "__main__":
    main()
