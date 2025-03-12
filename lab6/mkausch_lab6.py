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
    tpr = tp / (tp + fn)
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



def main():

    df = load_grape_data("PerforEva_grape_data.txt", False)

    min_val = df.iloc[:,0].min()
    max_val = df.iloc[:,0].max()
    print(min_val)
    print(max_val)

    d_list = []
    f_list = []
    acc_list = []
    prec_list = []
    rec_list = []

    # Tpr and Fpr lists
    tpr_list = []
    fpr_list = []

    d = 1
    delta = .1
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

    # Scatter Plot
    plt.plot(d_list, f_list, marker='', alpha=0.8, label="f1 vals")
    plt.plot(d_list, acc_list, marker='', alpha=0.8, label="accuracy")
    plt.plot(d_list, prec_list, marker='', alpha=0.8, label="precision")
    plt.plot(d_list, rec_list, marker='', alpha=0.8, label="recall")
    # plt.title("f1 val by distance cutoff")
    plt.xlabel("y-hat distance cutoff")
    # plt.ylabel("f1 (cm)")
    plt.legend()
    plt.show()

    # ROC Curve
    plt.plot(fpr_list, tpr_list, marker='', alpha=0.8, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

    # Caluclate the area under the curve using a function area_under_curve(fpr, tpr)
    auc = area_under_curve(fpr_list, tpr_list)
    print(f"AUC: {auc}")

if __name__ == "__main__":
    main()
