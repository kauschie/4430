
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
def import_army_data(file_path):
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
            labels.append(label)
            full.append((int(height), int(weight), label))
            index += 1
    # create a dictionary with the data
    data = {
        'heights': np.array(heights),
        'weights': np.array(weights),
        'labels': np.array(labels),
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

def plot_accuracies(accuracies, title=None, save_path=None):
    # plot the accuracies
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Magic Number')
    plt.ylabel('Accuracy')
    if title:
        plt.title(title)
    else:
        plt.title('Accuracy vs Magic Number')

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Example data
    data_path = "Known.csv"
    data = import_army_data(data_path)
    
    # Number of folds
    k = 5 # 20% of the data
    
    # Call the function
    folds, fold_paths, train_paths = k_fold_cross_validation(data["full"], k)

    # Not allowed to call the function directly in our program,
    # can only call it on the command line 
    # args are python MagicCode.py magic_number known_data unknown_data

    magic_number = 1 # must be > 0

    avg_accuracies = []
    max_accuracy = 0
    best_magic_number = [-1]

    # set up tqdm description

    tqdm_desc = f"Magic Number: {best_magic_number}, Accuracy: {max_accuracy * 100:.2f}%"
    progress = tqdm.tqdm(total=120*10, desc=tqdm_desc, dynamic_ncols=True)
    
    for test in range(1,11):   
        folds, fold_paths, train_paths = k_fold_cross_validation(data["full"], k)
        for i in range(1, 121):
            magic_number = i
            accuracies = []
            for j, fp in enumerate(fold_paths):
                result = run_magic_code(magic_number, train_paths[j], fp)
                if result:
                    accuracies.append(get_accuracy(result, folds[j]))
                else:
                    accuracies.append(0)
            
            # calculate the average accuracy
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_accuracies.append(avg_accuracy)
            # print(f"Average accuracy for magic number {magic_number}: {avg_accuracy * 100:.2f}%")
            
            # check if this is the best magic number
            if avg_accuracy > max_accuracy:
                max_accuracy = avg_accuracy
                best_magic_number.clear()
                best_magic_number.append(magic_number)
            elif avg_accuracy == max_accuracy and magic_number not in best_magic_number:
                best_magic_number.append(magic_number)


            progress.set_description(f"Best Magic Number: {best_magic_number}, Max Accuracy: {max_accuracy * 100:.2f}%")
            progress.update(1)

    print(f"Best magic number: {best_magic_number} with accuracy: {max_accuracy * 100:.2f}%")

    # plot the accuracies
    plot_accuracies(avg_accuracies, title="Accuracy vs Magic Number", save_path="accuracy_plot.png")
