import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import numpy as np
matplotlib.use('TkAgg')

def load_grape_data(file_path, use_space:bool = False):
    try:
        if use_space:
            df = pd.read_csv(file_path, sep='\\s+', header=None)
        else:
            df = pd.read_csv(file_path, header=None)
        df = df.T
        df.columns = ["diameter", "type"]
        df["diameter"] = df["diameter"].astype(float)
        df["type"] = df["type"].astype(int)
        return df
    except Exception as e:
        print(f"error loading file: {e}")
        return None
    

def get_f_val(df, val):
    tp, tn, fn, fp = 0, 0, 0, 0
    n = df["diameter"].size
    for i in range(n): # iterate through all samples
        diam = df.iloc[i]["diameter"]
        y_actual = df.iloc[i]["type"].astype(int)
        y_hat = (0 if diam < val else 1)


        if (y_actual == 1 and y_hat == 1):
        # get true pos
            tp += 1
        elif (y_actual == 0 and y_hat == 1):
        # get false pos
            fp += 1
        elif (y_actual == 1 and y_hat == 0):
        # get false neg
            fn += 1
        elif (y_actual == 0 and y_hat == 0):
        # get true neg
            tn += 1
        else:
            print("shits wrong bro, you shouldn't be here")

    epsilon = 1E-20
    accuracy = (tp+tn)/(tp+tn+fp+fn+epsilon)
    precision = (tp)/(tp+fp+epsilon)    # used when cost of false positives is high
    recall = (tp)/(tp+fn+epsilon)   # used when the cost of false negatives is high
    f1 = 2 * ((precision * recall) / (precision + recall+epsilon))

    # TPR and FPR are also useful metrics
    tpr = tp / (tp + fn) # just recall?
    fpr = fp / (fp + tn)    
    return f1, accuracy, precision, recall, tpr, fpr

def area_under_curve(x_vals, y_vals):
    """
    Computes the area under the curve using the trapezoidal rule.

    Parameters:
    - x_vals (list of float): List of x-axis values.
    - y_vals (list of float): List of corresponding y-axis values.

    Returns:
    - float: The computed area under the curve.
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length.")
    
    # Ensure x_vals and y_vals are sorted by x values
    sorted_pairs = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    x_sorted, y_sorted = zip(*sorted_pairs)

    # Use NumPy's trapezoidal rule for numerical integration
    auc = np.trapezoid(y_sorted, x_sorted)
    
    return auc

def auc_rect(x_vals, y_vals):
    """
    Computes auc using rectangles instead of trapezoids

    Parameters:
        x_vals: false positive rate (x-axis)
        y_vals: true positive rate (height of rectangle)

    Return:
        float: area under curve using rect. method

    """

    # right riemann sum (underestimate)
    sum1 = 0
    for i in range(len(x_vals)-1):
        sum1 += (y_vals[i] * abs(x_vals[i+1]-x_vals[i])) # calc width of rectangle by finding delta
    
    # left riemann sum (overestimate)
    sum2 = 0
    for i in range(1,len(x_vals)):
        sum2 += (y_vals[i] * abs(x_vals[i]-x_vals[i-1]))

    # return average    
    return (sum1+sum2)/2.0


def calc_best_cutoff(precisions, recalls):
    """
    Calculates the best cutoff for a given precision-recall curve
    inputs:
        precisions: list of precision values
        recalls: list of recall values
    returns:
        index of best cutoff
    """
    index = None
    for i in range(len(recalls)):
        if precisions[i] > recalls[i]:
            index = i
            break
    return index

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
            

def precision_recall_curve(precisions, recalls, best_cutoff_index, youdens_cutoff_index):
    """
    Plots the precision-recall curve

    Parameters:
    - precisions (list of float): List of precision values.
    - recalls (list of float): List of recall values.
    - best_cutoff_index (int): Index of best cutoff (recall x precision)
    - youdens_cutoff_index (int): Index of best cutoff that's closest to (0,1) on the ROC curve

    Returns:
     - Nothing

    """
    
    plt.scatter(recalls, precisions, marker='.', alpha=0.8, label="Precision-Recall Curve")
    plt.scatter(recalls[best_cutoff_index], precisions[best_cutoff_index], marker='*', color='red', label="Precision-Recall Intersection Cutoff") 
    plt.scatter(recalls[youdens_cutoff_index], precisions[youdens_cutoff_index], marker='*', color='green', label="Youden's Best Cutoff") 
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend()
    plt.show()
    

def main():

    df = load_grape_data("PerforEva_grape_data.txt", False)

    min_val = df.iloc[:,0].min()
    max_val = df.iloc[:,0].max()
    print(f"min_val: {min_val}")
    print(f"max_val: {max_val}")

    d_list = []
    f_list = []
    acc_list = []
    prec_list = []
    rec_list = []

    # Tpr and Fpr lists
    tpr_list = []
    fpr_list = []

    d = min_val
    delta = .01

    num_iter = int((max_val - min_val) / delta)
    with tqdm.tqdm(total=num_iter) as pbar:
        while d < max_val:
            f1, acc, prec, rec, tpr, fpr = get_f_val(df, d)
            d_list.append(d)
            f_list.append(f1)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            # Append to Tpr and Fpr
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            d += delta
            pbar.update(1)

    print(f"Number of points: %d", len(d_list))

    best_cutoff_index = calc_best_cutoff(prec_list, rec_list)
    best_cutoff = d_list[best_cutoff_index]
    youdens_cutoff_index = calc_auc_cutoff(tpr_list, fpr_list)
    youdens_cutoff = d_list[youdens_cutoff_index]
    
    # Scatter Plot
    plt.plot(d_list, f_list, marker='', alpha=0.8, label="f1 vals")
    plt.plot(d_list, acc_list, marker='', alpha=0.8, label="accuracy")
    plt.plot(d_list, prec_list, marker='', alpha=0.8, label="precision")
    plt.plot(d_list, rec_list, marker='', alpha=0.8, label="recall")
    plt.scatter(best_cutoff, prec_list[best_cutoff_index], marker='*', color='red', label="Precision-Recall Intersection Cutoff")
    plt.scatter(youdens_cutoff, prec_list[youdens_cutoff_index], marker='*', color='green', label="Youden's Best Cutoff")
    plt.scatter(youdens_cutoff, rec_list[youdens_cutoff_index], marker='*', color='green', label="Youden's Best Cutoff")
    # plt.title("f1 val by distance cutoff")
    plt.xlabel("y-hat distance cutoff")
    # plt.ylabel("f1 (cm)")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend()
    plt.show()

    print(f"Precision at youden's: {prec_list[youdens_cutoff_index]}")
    print(f"Recall at youden's: {rec_list[youdens_cutoff_index]}")

    
    print(f"Best cutoff index: {best_cutoff_index}")
    print(f"Best cutoff: {best_cutoff}")
    print(f"Youden's Best cutoff index: {youdens_cutoff_index}")
    print(f"Youden's Best cutoff: {youdens_cutoff}")

    # Precision Recall Curve
    precision_recall_curve(prec_list, rec_list, best_cutoff_index, youdens_cutoff_index)
    print(f"Precision: {prec_list[best_cutoff_index]}")
    print(f"Recall: {rec_list[best_cutoff_index]}")
    print(f"fpr at intersect cutoff: {fpr_list[best_cutoff_index]}")
    print(f"tpr at instersect cutoff: {tpr_list[best_cutoff_index]}")
    print(f"fpr at youdens cutoff: {fpr_list[youdens_cutoff_index]}")
    print(f"tpr at youdens cutoff: {tpr_list[youdens_cutoff_index]}")


    # ROC Curve
    plt.plot(fpr_list, tpr_list, marker='', alpha=0.8, label="ROC Curve")
    plt.plot([0,1], [0,1], color='black', linestyle='--', label="Random Guessing")
    plt.scatter(fpr_list[best_cutoff_index], tpr_list[best_cutoff_index], marker='*', color='red', label="Precision-Recall Intersection Cutoff")
    plt.scatter(fpr_list[youdens_cutoff_index], tpr_list[youdens_cutoff_index], marker='*', color='green', label="Youden's Best Cutoff")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend()
    plt.show()

    # ROC Scatter Plot
    plt.scatter(fpr_list, tpr_list, marker='.', alpha=0.8, label="ROC Curve")
    # plt.scatter(fpr_list[best_cutoff_index], tpr_list[best_cutoff_index], marker='*', color='red', label="Best Cutoff")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend()
    plt.show()

    # Caluclate the area under the curve using a function area_under_curve(fpr, tpr)
    auc2 = auc_rect(fpr_list, tpr_list)
    print(f"AUC (rectangle): {auc2}")
    auc = area_under_curve(fpr_list, tpr_list)
    print(f"AUC (trapezoid): {auc}")

if __name__ == "__main__":
    main()
