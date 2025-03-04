import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tqdm
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
    return f1, accuracy, precision, recall

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

    d = 1
    delta = .1
    while d < max_val:
        f1, acc, prec, rec = get_f_val(df, d)
        d_list.append(d)
        f_list.append(f1)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
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

if __name__ == "__main__":
    main()
