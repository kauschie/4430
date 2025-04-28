
# TODO: Use something else besides accuracy
#       - look into precision/recall/f1-score/AUC

import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib
import tqdm
# Set the backend for matplotlib to avoid GUI issues
# This is necessary for some environments where the default backend may not work
matplotlib.use('TkAgg')


# program is going to call 

# k-fold cross validation 
# split data into k folds
def k_fold_cross_validation(data, k):
    random.shuffle(data)
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        # for the last fold, make sure you go all the way to the end
        if i < k - 1:
            end = (i + 1) * fold_size
        else:
            end = len(data)
        folds.append(data[start:end])


    fold_paths, train_paths = [], []

    # write each test fold as CSV
    for i, fold in enumerate(folds):
        path = f'fold_{i}.txt'
        fold_paths.append(path)
        with open(path, 'w') as f:
            for item in fold:
                # item is a tuple like (idx, h, w, label)
                f.write(f"{item[0]}, {item[1]}\n")

    # write each training set as CSV
    for i in range(k):
        path = f'train_{i}.txt'
        train_paths.append(path)
        with open(path, 'w') as f:
            for j, fold in enumerate(folds):
                if i != j:
                    for item in fold:
                        f.write(",".join(map(str, item)) + "\n")

    print(f"{k} folds have been written to files.")
    return folds, fold_paths, train_paths


# import data from file
# each line has format height (int), weight (int), label (M or F)
def import_army_data(file_path, known=True):
    data = []
    with open(file_path, 'r') as f:
        heights = []
        weights = []
        labels = []
        full = []
        index = 0

        for line in f:
            height, weight, label = line.strip().split(',')
            heights.append(int(height))
            weights.append(int(weight))
            if known == True:
                labels.append(label)
            full.append((int(height), int(weight), label))
            index += 1
    # create a dictionary with the data
    if known == True:
        data = {
            'heights': np.array(heights),
            'weights': np.array(weights),
            'labels': np.array(labels),
            'full': full
        }
    else:
        data = {
            'heights': np.array(heights),
            'weights': np.array(weights),
            'full': full
        }

    return data

def run_magic_code(magic_number, known_data, unknown_data):
    # run the magic code
    result = subprocess.run(['python3', 'MagicCode.py', str(magic_number), known_data, unknown_data], capture_output=True, text=True)
    # get the output
    if result.returncode != 0:
        # print("Error running MagicCode.py")
        # print(f"Return code: {result.returncode}")
        # print(f"Error message: {result.stderr}")
        # print(f"Output:\n{result.stdout}\n")
        return None
    else:
        # print(f"Script ran successfully with return code {result.returncode}")
        # print(f"Output:\n{result.stdout}\n")
        lines = result.stdout.strip().splitlines()
        # print(f"Processed {len(lines)} output lines: {lines}")
        return lines  # Added return statement to return processed output lines

def get_accuracy(labels, test_data):
    # get the accuracy of the model
    correct = 0
    for i, line in enumerate(labels):
        if line == test_data[i][2]:
            correct += 1
    accuracy = correct / len(labels)
    # print(f"Accuracy: {accuracy * 100:.2f}%")  # Uncommented to print accuracy
    return accuracy


def get_metrics(labels, test_data, positive_class='M'):
    tp, tn, fp, fn = 0, 0, 0, 0
    epsilon = 1e-20  # small value to avoid division by zero

    for i, y_hat in enumerate(labels):
        y_actual = test_data[i][2]

        # Map based on desired positive class
        y_hat_bin = 1 if y_hat == positive_class else 0
        y_actual_bin = 1 if y_actual == positive_class else 0

        if y_actual_bin == 1 and y_hat_bin == 1:
            tp += 1
        elif y_actual_bin == 0 and y_hat_bin == 1:
            fp += 1
        elif y_actual_bin == 1 and y_hat_bin == 0:
            fn += 1
        elif y_actual_bin == 0 and y_hat_bin == 0:
            tn += 1
        else:
            print(f"Unexpected label: y_actual={y_actual}, y_hat={y_hat}")

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    tpr = recall  # true positive rate = recall
    fpr = fp / (fp + tn + epsilon)

    return f1, accuracy, precision, recall, tpr, fpr


# def plot_accuracies(accuracies, title=None, save_path=None):
#     # plot the accuracies
#     plt.plot(range(1, len(accuracies) + 1), accuracies)
#     plt.xlabel('Magic Number')
#     plt.ylabel('Accuracy')
#     if title:
#         plt.title(title)
#     else:
#         plt.title('Accuracy vs Magic Number')

#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
#     plt.close()


def plot_accuracies(acc_list, f_list, prec_list, rec_list, title=None, save_path=None):
    d_list = list(range(1, len(acc_list) + 1))  # x-axis = magic numbers 1,2,3,...

    plt.figure(figsize=(10,6))  # (optional) make the figure bigger and nicer

    plt.plot(d_list, f_list, alpha=0.8, label="F1 Score")
    plt.plot(d_list, acc_list, alpha=0.8, label="Accuracy")
    plt.plot(d_list, prec_list, alpha=0.8, label="Precision")
    plt.plot(d_list, rec_list, alpha=0.8, label="Recall")

    plt.xlabel('Magic Number')
    plt.ylabel('Score')
    
    if title:
        plt.title(title)
    else:
        plt.title('Model Performance Metrics vs Magic Number')

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # save cleanly
    plt.show()
    plt.close()

def plot_male_vs_female(male_vals, female_vals, ylabel, title=None, save_path=None):
    d_list = list(range(1, len(male_vals) + 1))
    plt.figure(figsize=(10,6))

    plt.plot(d_list, male_vals, label="Male", alpha=0.8)
    plt.plot(d_list, female_vals, label="Female", alpha=0.8)

    plt.xlabel('Magic Number')
    plt.ylabel(ylabel)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Male vs Female {ylabel} by Magic Number')

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Example data
    data_path = "Known.csv"
    data = import_army_data(data_path)
    
    # Number of folds
    k = 5  # 20% of the data
    
    # Number of repetitions of k-fold CV
    num_tests = 20

    # Create storage for all tests
    avg_accuracies_all_tests = []
    avg_precisions_all_tests = []
    avg_recalls_all_tests = []
    avg_f1s_all_tests = []
    avg_f_precisions_all_tests = []
    avg_f_recalls_all_tests = []
    avg_f_f1s_all_tests = []

    # Set up tqdm description
    progress = tqdm.tqdm(total=120*num_tests, desc="Running Tests", dynamic_ncols=True)

    for test in range(1, num_tests + 1):
        folds, fold_paths, train_paths = k_fold_cross_validation(data["full"], k)

        avg_accuracies = []  # store avg accuracy for each magic number in this test
        avg_precisions = []
        avg_recalls = []
        avg_f1s = []
        avg_f_precisions = []
        avg_f_recalls = []
        avg_f_f1s = []


        for i in range(1, 121):  # Magic numbers 1 to 120
            magic_number = i
            accuracies = []
            precisions = []
            recalls = []
            f1s = []

            f_precisions = []
            f_recalls = []
            f_f1s = []

            for j, fp in enumerate(fold_paths):
                result = run_magic_code(magic_number, train_paths[j], fp)
                if result:
                    f1_m, accuracy_m, precision_m, recall_m, tpr_m, fpr_m = get_metrics(result, folds[j], positive_class='M')
                    f1_f, accuracy_f, precision_f, recall_f, tpr_f, fpr_f = get_metrics(result, folds[j], positive_class='F')

                    # append to male lists
                    accuracies.append(accuracy_m)
                    precisions.append(precision_m)
                    recalls.append(recall_m)
                    f1s.append(f1_m)

                    # append to female lists
                    f_precisions.append(precision_f)
                    f_recalls.append(recall_f)
                    f_f1s.append(f1_f)
                else:
                    accuracies.append(0)
                    precisions.append(0)
                    recalls.append(0)
                    f1s.append(0)
                    f_precisions.append(0)
                    f_recalls.append(0)
                    f_f1s.append(0)

            # Average across folds
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_accuracies.append(avg_accuracy)

            avg_precision = sum(precisions) / len(precisions)
            avg_precisions.append(avg_precision)

            avg_recall = sum(recalls) / len(recalls)
            avg_recalls.append(avg_recall)

            avg_f1 = sum(f1s) / len(f1s)
            avg_f1s.append(avg_f1)

            # --- add these:
            avg_f_precision = sum(f_precisions) / len(f_precisions)
            avg_f_precisions.append(avg_f_precision)

            avg_f_recall = sum(f_recalls) / len(f_recalls)
            avg_f_recalls.append(avg_f_recall)

            avg_f_f1 = sum(f_f1s) / len(f_f1s)
            avg_f_f1s.append(avg_f_f1)


            progress.update(1)

        # Save all magic number results for this test
        avg_accuracies_all_tests.append(avg_accuracies)
        avg_precisions_all_tests.append(avg_precisions)
        avg_recalls_all_tests.append(avg_recalls)
        avg_f1s_all_tests.append(avg_f1s)
        avg_f_precisions_all_tests.append(avg_f_precisions)
        avg_f_recalls_all_tests.append(avg_f_recalls)
        avg_f_f1s_all_tests.append(avg_f_f1s)


    progress.close()

    # --- After all tests are finished ---

    # Convert to NumPy array for easy processing
    avg_accuracies_all_tests = np.array(avg_accuracies_all_tests)  # shape: (num_tests, 120)
    avg_precisions_all_tests = np.array(avg_precisions_all_tests)
    avg_recalls_all_tests = np.array(avg_recalls_all_tests)
    avg_f1s_all_tests = np.array(avg_f1s_all_tests)


    # Average across all tests
    mean_accuracies = np.mean(avg_accuracies_all_tests, axis=0)
    mean_precisions = np.mean(avg_precisions_all_tests, axis=0)
    mean_recalls = np.mean(avg_recalls_all_tests, axis=0)
    mean_f1s = np.mean(avg_f1s_all_tests, axis=0)

    avg_f_precisions_all_tests = np.array(avg_f_precisions_all_tests)
    avg_f_recalls_all_tests = np.array(avg_f_recalls_all_tests)
    avg_f_f1s_all_tests = np.array(avg_f_f1s_all_tests)

    mean_f_precisions = np.mean(avg_f_precisions_all_tests, axis=0)
    mean_f_recalls = np.mean(avg_f_recalls_all_tests, axis=0)
    mean_f_f1s = np.mean(avg_f_f1s_all_tests, axis=0)


    # Find the best magic number based on mean accuracies
    best_magic_number = np.argmax(mean_accuracies) + 1  # +1 because indexing starts at 0
    best_accuracy = mean_accuracies[best_magic_number - 1]

    print(f"Best magic number (averaged across tests): {best_magic_number} with mean accuracy: {best_accuracy*100:.2f}%")

    # Plot mean accuracies
    # plot_accuracies(mean_accuracies, title="Mean Accuracy vs Magic Number (across tests)", save_path="mean_accuracy_plot.png")
    plot_accuracies(mean_accuracies, mean_f1s, mean_precisions, mean_recalls,
                    title="Mean Metrics (Males as Positive Class)",
                    save_path="mean_metrics_male.png")

    plot_accuracies(mean_accuracies, mean_f_f1s, mean_f_precisions, mean_f_recalls,
                    title="Mean Metrics (Females as Positive Class)",
                    save_path="mean_metrics_female.png")

    # Precision comparison
    plot_male_vs_female(mean_precisions, mean_f_precisions,
                        ylabel="Precision",
                        title="Male vs Female Precision by Magic Number",
                        save_path="precision_m_vs_f.png")

    # Recall comparison
    plot_male_vs_female(mean_recalls, mean_f_recalls,
                        ylabel="Recall",
                        title="Male vs Female Recall by Magic Number",
                        save_path="recall_m_vs_f.png")

    # F1 Score comparison
    plot_male_vs_female(mean_f1s, mean_f_f1s,
                        ylabel="F1 Score",
                        title="Male vs Female F1 Score by Magic Number",
                        save_path="f1_m_vs_f.png")
    
    # Select the best magic number based on proportional F1 similarity and value
    best_magic_number = -1
    best_f1_min = -1  # Initialize to worst possible
    best_f1_ratio_diff = float('inf')

    for idx in range(len(mean_f1s)):
        f1_male = mean_f1s[idx]
        f1_female = mean_f_f1s[idx]

        # Avoid divide-by-zero
        if f1_male + f1_female < 1e-6:
            continue

        # Calculate ratio (always >= 1)
        ratio = f1_male / f1_female if f1_male >= f1_female else f1_female / f1_male
        ratio_diff = abs(ratio - 1)

        # Minimum F1
        min_f1 = min(f1_male, f1_female)

        # ✨ NEW: prioritize highest minimum F1 first
        if (min_f1 > best_f1_min) or (min_f1 == best_f1_min and ratio_diff < best_f1_ratio_diff):
            best_magic_number = idx + 1
            best_f1_min = min_f1
            best_f1_ratio_diff = ratio_diff

    # Final print
    print("\n=== Best Magic Number (High F1 first, then balanced) ===")
    print(f"Best magic number: {best_magic_number}")
    print(f"  Male F1: {mean_f1s[best_magic_number-1]:.4f}")
    print(f"  Female F1: {mean_f_f1s[best_magic_number-1]:.4f}")
    print(f"  Ratio (larger/smaller): {(max(mean_f1s[best_magic_number-1], mean_f_f1s[best_magic_number-1]) / min(mean_f1s[best_magic_number-1], mean_f_f1s[best_magic_number-1])):.4f}")

    # run the magic code on the unknown data
    result = run_magic_code(best_magic_number, "Known.csv", "Unknown.csv")
    if result:
        # write out results to final_classification.txt
        with open("final_classification.txt", 'w') as f:
            for line in result:
                f.write(line + "\n")
        print("Results written to final_classification.txt")
    else:
        print("Error running MagicCode.py on unknown data.")

