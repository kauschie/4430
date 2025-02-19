import numpy as np
import pandas as pd
import math
import csv
import matplotlib
matplotlib.use('TkAgg')

steam_purch_path = 'MeasurementsExamples_SteamPurchase.csv'
steam_list_path = 'MeasurementsExamples_SteamList.csv'
iris_path = 'MeasurementsExamples_iris.csv'
judge_path = 'MeasurementsExamples_Judges.csv'

def read_steamPurchase(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        p1, p2, p3, p4, p5 = [], [], [], [], []
        persons = [p1, p2, p3, p4, p5]
        headers = next(reader)  # Read header row
        for row in reader:
            for i in range(len(headers)):
                if i < len(row) and row[i] == "":  # Check if column exists and is empty
                    # print("EMPTY", end=", ")
                    continue
                else:
                    # print(f"{row[i]}", end=", ")  # Placeholder for non-empty values
                    persons[i].append(int(row[i]))
            # print()
        for i in range(len(headers)):
            data[headers[i]]=persons[i]
        # print(data)
        # print()
        
    return data
def read_steamList(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        app_id, name, genre = [], [], []
        headers = next(reader)  # Read header row
        for row in reader:
            app_id.append(row[0])
            name.append(row[1])
            genre.append(row[2])
        data = {headers[0]:app_id, headers[1]:name, headers[2]:genre}

        # print(data)
        # print()
    return data
# Dataset order: Sepal length, Sepal width, Petal length, Petal width, Species 
def read_iris(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        s_len, s_wid, p_len, p_wid, species = [], [], [], [], []
        g1, g2, g3 = {}, {}, {}

        prev_class = None
        g2_start = None
        g3_start = None

        count = 0
        headers = ["Sepal length", "Sepal width", "Petal length", "Petal width", "Species"]  # Read header row
        for row in reader:
            f1 = float(row[0])
            f2 = float(row[1])
            f3 = float(row[2])
            f4 = float(row[3])
            s = row[4]

            # init prev_class

            if prev_class == None:
                prev_class = s
            elif prev_class != s:
                if g2_start == None:
                    g2_start = count
                    prev_class = s
                elif g3_start == None:
                    g3_start = count
                    prev_class = s

            s_len.append(f1)
            s_wid.append(f2)
            p_len.append(f3)
            p_wid.append(f4)
            species.append(s)
            count += 1
                
        all = {headers[0]:s_len, 
                headers[1]:s_wid, 
                headers[2]:p_len,
                headers[3]:p_wid,
                headers[4]:species}
        g1 = {headers[0]:s_len[:g2_start], 
                headers[1]:s_wid[:g2_start], 
                headers[2]:p_len[:g2_start],
                headers[3]:p_wid[:g2_start]}
        g2 = {headers[0]:s_len[g2_start:g3_start], 
                headers[1]:s_wid[g2_start:g3_start], 
                headers[2]:p_len[g2_start:g3_start],
                headers[3]:p_wid[g2_start:g3_start]}
        g3 = {headers[0]:s_len[g3_start:], 
                headers[1]:s_wid[g3_start:], 
                headers[2]:p_len[g3_start:],
                headers[3]:p_wid[g3_start:],}
        
        

        data = {"all": all, "g1": g1, "g2":g2, "g3":g3}
        # print(data)
    return data

def read_judges(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [], [], [], [], [], [], [], [], [], []
        j1, j2, j3, j4, j5, j6 = [], [], [], [], [], []
        judges = [j1, j2, j3, j4, j5, j6]
        headers = next(reader)  # Read header row
        i = 0
        for row in reader:
            p1.append(float(row[0]))
            p2.append(float(row[1]))
            p3.append(float(row[2]))
            p4.append(float(row[3]))
            p5.append(float(row[4]))
            p6.append(float(row[5]))
            p7.append(float(row[6]))
            p8.append(float(row[7]))
            p9.append(float(row[8]))
            p10.append(float(row[9]))
            judges[i].append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])
            i += 1
        p_data = {headers[0]:p1, headers[1]:p2, headers[2]:p3,
                headers[3]:p4, headers[4]:p5, headers[0]:p6, 
                headers[1]:p7, headers[2]:p8, headers[3]:p9, 
                headers[4]:p10}
        j_data = {"Judge1":judges[0], "Judge2":judges[1], "Judge3":judges[2], "Judge4":judges[3], "Judge5":judges[4], "Judge6":judges[5]}
        data = {"Persons":p_data, "Judges":j_data}

        # print(data)
        # print()
    return data


import pandas as pd

def min_max_norm(list_of_vals):
    """
    Applies Min-Max Normalization to a list of values.
    """
    l_max = max(list_of_vals)
    l_min = min(list_of_vals)
    return [(v - l_min) / (l_max - l_min) for v in list_of_vals]

def remove_outliers_iqr(group):
    """
    Removes entire rows from a group if any feature is an outlier using the IQR method.
    Returns the cleaned group and the total number of removed rows.
    """
    df = pd.DataFrame(group)  # Convert dictionary to DataFrame
    original_size = len(df)  # Store the original number of rows

    # Compute IQR bounds for each feature (excluding the 'Species' column)
    lower_bounds, upper_bounds = {}, {}
    for col in df.columns[:-1]:  # Ignore 'Species' column
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile range
        lower_bounds[col] = Q1 - 1.5 * IQR
        upper_bounds[col] = Q3 + 1.5 * IQR

    # Create a mask that identifies entire rows to remove if ANY feature is an outlier
    mask = (df.iloc[:, :-1] < pd.Series(lower_bounds)) | (df.iloc[:, :-1] > pd.Series(upper_bounds))
    rows_to_remove = mask.any(axis=1)  # Identify rows where ANY feature is an outlier

    # Count total outliers removed
    removed_rows = rows_to_remove.sum()

    # Remove entire rows where any feature was an outlier
    df_cleaned = df[~rows_to_remove].reset_index(drop=True)

    return df_cleaned.to_dict(orient="list"), removed_rows  # Return cleaned data + count of removed rows

def preprocess_iris(iris_dict):
    """
    Removes outliers from each group and normalizes features using Min-Max Normalization.
    Also prints the number of outliers removed from each group.
    """
    print("before")
    for k,v in iris_dict.items():
        print(f"{k}")
        for k2,v2 in iris_dict[k].items():
            print(f"len({k2}): {len(v2)}")

    # Step 1: Extract groups (excluding "all")
    groups = {k: v for k, v in iris_dict.items() if k != "all"}

    # Step 2: Remove outliers for each group and store removal counts
    new_g_no_outliers = {}
    outlier_counts = {}

    for k, v in groups.items():
        cleaned_data, total_outliers = remove_outliers_iqr(v)
        new_g_no_outliers[k] = cleaned_data
        outlier_counts[k] = total_outliers

    # Print outlier summary
    print("\n📊 Outlier Removal Summary:")
    for group, count in outlier_counts.items():
        print(f"Group {group}: {count} rows removed")

    # Step 3: Compute feature-wise min/max across all cleaned groups
    all_data_no_outliers = pd.concat([pd.DataFrame(v) for v in new_g_no_outliers.values()])
    feature_min = all_data_no_outliers.iloc[:, :-1].min()
    feature_max = all_data_no_outliers.iloc[:, :-1].max()

    # Step 4: Normalize the data feature-wise
    normed_no_outlier_dict = {}
    for key, value in new_g_no_outliers.items():
        df = pd.DataFrame(value)  # Convert each group back to DataFrame
        for col in df.columns[:-1]:  # Normalize each feature
            df[col] = (df[col] - feature_min[col]) / (feature_max[col] - feature_min[col])
        normed_no_outlier_dict[key] = df.to_dict(orient="list")

    print()
    print("after")
    # print(normed_no_outlier_dict)
    for k,v in normed_no_outlier_dict.items():
        print(f"{k}")
        for k2,v2 in normed_no_outlier_dict[k].items():
            print(f"len({k2}): {len(v2)}")

    return normed_no_outlier_dict




def main():
    # need nxn matrix for all
    steamPurchaseData = read_steamPurchase(steam_purch_path)
    steamListData = read_steamList(steam_list_path)
    irisData = read_iris(iris_path)
    judgeData = read_judges(judge_path)

    normed_no_outlier_dict = preprocess_iris(irisData)




if __name__ == "__main__":
    main()