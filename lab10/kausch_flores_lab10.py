
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



def get_metrics_multiclass(labels, test_data, class_list):
    epsilon = 1e-20

    tp_dict = {c: 0 for c in class_list}
    fp_dict = {c: 0 for c in class_list}
    fn_dict = {c: 0 for c in class_list}

    n = len(labels)
    correct = 0

    for i in range(n):
        y_hat = labels[i]
        y_actual = test_data[i][-1]  # assuming label is last column

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

    # Macro average (simple unweighted average)
    macro_precision = sum(per_class_precision.values()) / len(class_list)
    macro_recall = sum(per_class_recall.values()) / len(class_list)
    macro_f1 = sum(per_class_f1.values()) / len(class_list)

    accuracy = correct / n

    return macro_f1, accuracy, macro_precision, macro_recall, per_class_f1, per_class_precision, per_class_recall


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

def plot_macro(macro_vals, ylabel, title=None, save_path=None):
    d_list = list(range(1, len(macro_vals) + 1))
    plt.figure(figsize=(10,6))

    plt.plot(d_list, macro_vals, label="Averaged Data", alpha=0.8)
    
    plt.xlabel('Magic Number')
    plt.ylabel(ylabel)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Averaged {ylabel} by Magic Number')

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def calc_auc_cutoff(tpr_list, fpr_list):
    """
    Calculates best cutoff that's closest to (0,1) on the ROC curve
    inputs:
        tpr_list: list of tpr values
        fpr_list: list of fpr values
    returns:
        index of best cutoff
    """
    index = None
    best = 0
    best_index = None
    for i in range(len(tpr_list)):
        youdens_j = tpr_list[i] - fpr_list[i]
        if youdens_j > best:
            best = youdens_j
            best_index = i
    return best_index

def precision_recall_curve(m_precisions, f_precisions, m_recalls, f_recalls):
    """
    Plots the precision-recall curve

    Parameters:
    - m_precisions (list of float): List of male precision values.
    - f_precisions (list of float): List of female precision values.
    - m_recalls (list of float): List of male recall values.
    - f_recalls (list of float): List of female recall values.

    Returns:
     - Nothing

    """

    plt.scatter(m_recalls, m_precisions, marker='.', alpha=0.8, label="Male PR Curve")
    plt.scatter(f_recalls, f_precisions, marker='.', alpha=0.8, label="Female PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.title("Male vs Female Precision-Recall Curve")
    plt.legend()
    plt.savefig("male_vs_female_precision_recall_curve.png", bbox_inches='tight')
    plt.show()

def plot_roc_curve(male_tpr_list, male_fpr_list, female_tpr_list, female_fpr_list):
    """
    Plots the ROC curve

    Parameters:
    - male_tpr_list (list of float): List of true positive rates for males.
    - male_fpr_list (list of float): List of false positive rates for males.
    - female_tpr_list (list of float): List of true positive rates for females.
    - female_fpr_list (list of float): List of false positive rates for females.

    Returns:
     - Nothing

    """
    plt.scatter(male_fpr_list, male_tpr_list, marker='.', alpha=0.8, label='Male ROC Curve')
    plt.scatter(female_fpr_list, female_tpr_list, marker='.', alpha=0.8, label='Female ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("mvsf_roc_curve.png", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Example data
    data_path = "Known.csv"
    data = import_army_data(data_path)
    last_magic_number = 120 # last magic number to test
    
    # Number of folds
    k = 5  # 20% of the data
    
    # Number of repetitions of k-fold CV
    num_tests = 10

    # Create storage for all tests
    avg_accuracies_all_tests = []
    avg_precisions_all_tests = []
    avg_recalls_all_tests = []
    avg_f1s_all_tests = []
    avg_f_precisions_all_tests = []
    avg_f_recalls_all_tests = []
    avg_f_f1s_all_tests = []
    avg_tpr_m_all_tests = []
    avg_fpr_m_all_tests = []
    avg_tpr_f_all_tests = []
    avg_fpr_f_all_tests = []

    avg_macro_f1_all_tests = []
    avg_macro_precision_all_tests = []
    avg_macro_accuracy_all_tests = []
    avg_macro_recall_all_tests = []
    avg_macro_tpr_all_tests = []
    avg_macro_fpr_all_tests = []

    # Set up tqdm description
    progress = tqdm.tqdm(total=last_magic_number*num_tests, desc="Running Tests", dynamic_ncols=True)

    for test in range(1, num_tests + 1):
        folds, fold_paths, train_paths = k_fold_cross_validation(data["full"], k)

        avg_accuracies = []  # store avg accuracy for each magic number in this test
        avg_precisions = []
        avg_recalls = []
        avg_f1s = []
        avg_f_precisions = []
        avg_f_recalls = []
        avg_f_f1s = []
        avg_tpr_m = []
        avg_fpr_m = []
        avg_tpr_f = []
        avg_fpr_f = []

        avg_macro_f1_list = []
        avg_macro_precision_list = []
        avg_macro_accuracy_list = []
        avg_macro_recall_list = []
        avg_macro_tpr_list = []
        avg_macro_fpr_list = []


        for i in range(1, last_magic_number + 1):  # Magic numbers 1 to 120
            magic_number = i
            accuracies = []
            precisions = []
            recalls = []
            f1s = []
            m_tpr = []
            m_fpr = []

            f_precisions = []
            f_recalls = []
            f_f1s = []
            f_tpr = []
            f_fpr = []

            macro_f1s = []
            macro_accuracys = []
            macro_precisions = []
            macro_recalls = []
            macro_tprs = []
            macro_fprs = []

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
                    m_tpr.append(tpr_m)
                    m_fpr.append(fpr_m)

                    # append to female lists
                    f_precisions.append(precision_f)
                    f_recalls.append(recall_f)
                    f_f1s.append(f1_f)
                    f_tpr.append(tpr_f)  # append to female TPR list
                    f_fpr.append(fpr_f)  # append to female FPR list

                    macro_f1 = (f1_m + f1_f) / 2
                    macro_accuracy = (accuracy_m + accuracy_f) / 2
                    macro_precision = (precision_m + precision_f) / 2
                    macro_recall = (recall_m + recall_f) / 2
                    macro_tpr = (tpr_m + tpr_f) / 2
                    macro_fpr = (fpr_m + fpr_f) / 2

                    macro_f1s.append(macro_f1)
                    macro_accuracys.append(macro_accuracy)
                    macro_precisions.append(macro_precision)
                    macro_recalls.append(macro_recall)
                    macro_tprs.append(macro_tpr)
                    macro_fprs.append(macro_fpr)


                else:
                    accuracies.append(0)
                    precisions.append(0)
                    recalls.append(0)
                    f1s.append(0)
                    f_precisions.append(0)
                    f_recalls.append(0)
                    f_f1s.append(0)
                    m_tpr.append(0)
                    m_fpr.append(0)
                    f_tpr.append(0)
                    f_fpr.append(0)
                    macro_f1s.append(0)
                    macro_accuracys.append(0)
                    macro_precisions.append(0)
                    macro_recalls.append(0)
                    macro_tprs.append(0)
                    macro_fprs.append(0)

            # Average across folds
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_accuracies.append(avg_accuracy)

            avg_precision = sum(precisions) / len(precisions)
            avg_precisions.append(avg_precision)
            avg_f_precision = sum(f_precisions) / len(f_precisions)
            avg_f_precisions.append(avg_f_precision)

            avg_recall = sum(recalls) / len(recalls)
            avg_recalls.append(avg_recall)
            avg_f_recall = sum(f_recalls) / len(f_recalls)
            avg_f_recalls.append(avg_f_recall)

            avg_f1 = sum(f1s) / len(f1s)
            avg_f1s.append(avg_f1)
            avg_f_f1 = sum(f_f1s) / len(f_f1s)
            avg_f_f1s.append(avg_f_f1)

            avg_tpr_m.append(sum(m_tpr) / len(m_tpr))
            avg_fpr_m.append(sum(m_fpr) / len(m_fpr))
            avg_tpr_f.append(sum(f_tpr) / len(f_tpr))
            avg_fpr_f.append(sum(f_fpr) / len(f_fpr))

            # Macro averages - across all classifiers
            avg_macro_f1 = sum(macro_f1s) / len(macro_f1s)
            avg_macro_accuracy = sum(macro_accuracys) / len(macro_accuracys)
            avg_macro_precision = sum(macro_precisions) / len(macro_precisions)
            avg_macro_recall = sum(macro_recalls) / len(macro_recalls)
            avg_macro_tpr = sum(macro_tprs) / len(macro_tprs)
            avg_macro_fpr = sum(macro_fprs) / len(macro_fprs)
            avg_macro_f1_list.append(avg_macro_f1)
            avg_macro_accuracy_list.append(avg_macro_accuracy)
            avg_macro_precision_list.append(avg_macro_precision)
            avg_macro_recall_list.append(avg_macro_recall)
            avg_macro_tpr_list.append(avg_macro_tpr)
            avg_macro_fpr_list.append(avg_macro_fpr)

            progress.update(1)

        # Save all magic number results for this test
        avg_accuracies_all_tests.append(avg_accuracies)
        avg_precisions_all_tests.append(avg_precisions)
        avg_recalls_all_tests.append(avg_recalls)
        avg_f1s_all_tests.append(avg_f1s)
        avg_f_precisions_all_tests.append(avg_f_precisions)
        avg_f_recalls_all_tests.append(avg_f_recalls)
        avg_f_f1s_all_tests.append(avg_f_f1s)
        
        avg_tpr_m_all_tests.append(avg_tpr_m)
        avg_fpr_m_all_tests.append(avg_fpr_m)
        avg_tpr_f_all_tests.append(avg_tpr_f)
        avg_fpr_f_all_tests.append(avg_fpr_f)

        avg_macro_f1_all_tests.append(avg_macro_f1_list)
        avg_macro_accuracy_all_tests.append(avg_macro_accuracy_list)
        avg_macro_precision_all_tests.append(avg_macro_precision_list)
        avg_macro_recall_all_tests.append(avg_macro_recall_list)
        avg_macro_tpr_all_tests.append(avg_macro_tpr_list)
        avg_macro_fpr_all_tests.append(avg_macro_fpr_list)


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

    # Average TPR and FPR
    avg_tpr_m_all_tests = np.array(avg_tpr_m_all_tests)
    avg_fpr_m_all_tests = np.array(avg_fpr_m_all_tests)
    avg_tpr_f_all_tests = np.array(avg_tpr_f_all_tests)
    avg_fpr_f_all_tests = np.array(avg_fpr_f_all_tests)
    mean_tpr_m = np.mean(avg_tpr_m_all_tests, axis=0)
    mean_fpr_m = np.mean(avg_fpr_m_all_tests, axis=0)
    mean_tpr_f = np.mean(avg_tpr_f_all_tests, axis=0)
    mean_fpr_f = np.mean(avg_fpr_f_all_tests, axis=0)

    # Average macro metrics
    avg_macro_f1_all_tests = np.array(avg_macro_f1_all_tests)
    avg_macro_accuracy_all_tests = np.array(avg_macro_accuracy_all_tests)
    avg_macro_precision_all_tests = np.array(avg_macro_precision_all_tests)
    avg_macro_recall_all_tests = np.array(avg_macro_recall_all_tests)
    avg_macro_tpr_all_tests = np.array(avg_macro_tpr_all_tests)
    avg_macro_fpr_all_tests = np.array(avg_macro_fpr_all_tests)
    mean_macro_f1 = np.mean(avg_macro_f1_all_tests, axis=0)
    mean_macro_accuracy = np.mean(avg_macro_accuracy_all_tests, axis=0)
    mean_macro_precision = np.mean(avg_macro_precision_all_tests, axis=0)
    mean_macro_recall = np.mean(avg_macro_recall_all_tests, axis=0)
    mean_macro_tpr = np.mean(avg_macro_tpr_all_tests, axis=0)
    mean_macro_fpr = np.mean(avg_macro_fpr_all_tests, axis=0)


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
    
    # Macro metrics comparison
    plot_accuracies(mean_macro_accuracy, mean_macro_f1, mean_macro_precision, mean_macro_recall,
                    title="Mean Macro Metrics (Averaged Across Tests)",
                    save_path="mean_macro_metrics.png")

    # Precision-Recall Curve
    precision_recall_curve(mean_precisions, mean_f_precisions, mean_recalls, mean_f_recalls)
    precision_recall_curve(avg_precisions_all_tests.mean(axis=0), avg_f_precisions_all_tests.mean(axis=0),
                           avg_recalls_all_tests.mean(axis=0), avg_f_recalls_all_tests.mean(axis=0))

    # # ROC Curve
    plot_roc_curve(avg_tpr_m, avg_fpr_m, avg_tpr_f, avg_fpr_f)

    # Precision-Recall Curve (macro averaged)
    plt.figure()
    plt.scatter(mean_macro_recall, mean_macro_precision, marker='o', label='Macro PR Curve (Averaged)')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro Precision-Recall Curve (Averaged Across Tests)")
    plt.grid(True)
    plt.legend()
    plt.savefig("macro_pr_curve_averaged.png", bbox_inches='tight')
    plt.show()

    # ROC Curve (macro averaged)
    plt.figure()
    plt.scatter(mean_macro_fpr, mean_macro_tpr, marker='o', label='Macro ROC Curve (Averaged)')
    plt.scatter([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro ROC Curve (Averaged Across Tests)")
    plt.grid(True)
    plt.legend()
    plt.savefig("macro_roc_curve_averaged.png", bbox_inches='tight')
    plt.show()

    best_idx = calc_auc_cutoff(mean_macro_tpr, mean_macro_fpr)
    best_magic_number = best_idx + 1  # if you used 1-based indexing
    print(f"Best magic number (Youden's J statistic): {best_magic_number} with mean accuracy: {mean_macro_accuracy[best_idx]*100:.2f}%")
    print(f"Best magic number (Youden's J statistic): {best_magic_number} with mean F1: {mean_macro_f1[best_idx]:.4f}")
    print(f"Best magic number (Youden's J statistic): {best_magic_number} with mean precision: {mean_macro_precision[best_idx]:.4f}")
    print(f"Best magic number (Youden's J statistic): {best_magic_number} with mean recall: {mean_macro_recall[best_idx]:.4f}")


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

