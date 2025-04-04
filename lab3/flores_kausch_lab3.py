import numpy as np
import pandas as pd
import csv
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
matplotlib.use('TkAgg')

steam_purch_path = 'MeasurementsExamples_SteamPurchase.csv'
steam_list_path = 'MeasurementsExamples_SteamList.csv'
iris_path = 'MeasurementsExamples_iris.csv'
judge_path = 'MeasurementsExamples_Judges.csv'


#########################
##  File Reading Funcs ##
#########################

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
                    persons[i].append(row[i])
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
def read_iris(filepath):
# Dataset order: Sepal length, Sepal width, Petal length, Petal width, Species 
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
        judges = []
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
            judges.append([float(row[0]), float(row[1]), 
                           float(row[2]), float(row[3]), 
                           float(row[4]), float(row[5]), 
                           float(row[6]), float(row[7]), 
                           float(row[8]), float(row[9])])
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


#########################
##  Iris Preprocessing ##
#########################

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
    # print(f"df: {df}")
    original_size = len(df)  # Store the original number of rows

    # Compute IQR bounds for each feature (excluding the 'Species' column)
    lower_bounds, upper_bounds = {}, {}
    for col in df.columns:  # Ignore 'Species' column
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile range
        lower_bounds[col] = Q1 - 1.5 * IQR
        upper_bounds[col] = Q3 + 1.5 * IQR

    # Create a mask that identifies entire rows to remove if ANY feature is an outlier
    mask = (df < pd.Series(lower_bounds)) | (df > pd.Series(upper_bounds))
    rows_to_remove = mask.any(axis=1)  # Identify rows where ANY feature is an outlier

    # Count total outliers removed
    removed_rows = rows_to_remove.sum()

    # Remove entire rows where any feature was an outlier
    df_cleaned = df[~rows_to_remove].reset_index(drop=True)

    # print data summary
    # print("Data Summary:")
    # data_summary = df_cleaned.agg(['mean', 'std'])
    # print(data_summary)

    return df_cleaned.to_dict(orient="list"), removed_rows  # Return cleaned data + count of removed rows

def preprocess_iris(iris_dict):
    """
    Removes outliers from each group and normalizes features using Min-Max Normalization.
    Also prints the number of outliers removed from each group.
    """
    # print("before")
    # for k,v in iris_dict.items():
    #     print(f"{k}")
    #     for k2,v2 in iris_dict[k].items():
    #         print(f"len({k2}): {len(v2)}")

    # Step 1: Extract groups (excluding "all")
    groups = {k: v for k, v in iris_dict.items() if k != "all"}

    # Step 2: Remove outliers for each group and store removal counts
    new_g_no_outliers = {}
    outlier_counts = {}

    for k, v in groups.items():
        cleaned_data, total_outliers = remove_outliers_iqr(v)
        new_g_no_outliers[k] = cleaned_data
        outlier_counts[k] = total_outliers

    # # Print outlier summary
    # print("\n📊 Outlier Removal Summary:")
    # for group, count in outlier_counts.items():
    #     print(f"Group {group}: {count} rows removed")

    # Step 3: Compute feature-wise min/max across all cleaned groups
    all_data_no_outliers = pd.concat([pd.DataFrame(v) for v in new_g_no_outliers.values()])
    feature_min = all_data_no_outliers.iloc[:, :].min()
    feature_max = all_data_no_outliers.iloc[:, :].max()

    # Step 4: Normalize the data feature-wise
    normed_no_outlier_dict = {}
    for key, value in new_g_no_outliers.items():
        df = pd.DataFrame(value)  # Convert each group back to DataFrame
        # print(f"\n\ndf{key}: {df}")
        for col in df.columns[:]:  # Normalize each feature
            df[col] = (df[col] - feature_min[col]) / (feature_max[col] - feature_min[col])
        normed_no_outlier_dict[key] = df.to_dict(orient="list")

    # print()
    # print("after")
    # print(normed_no_outlier_dict)
    # for k,v in normed_no_outlier_dict.items():
    #     print(f"{k}")
    #     for k2,v2 in normed_no_outlier_dict[k].items():
    #         print(f"len({k2}): {len(v2)}")

    return normed_no_outlier_dict

def reconstruct_all(normed_no_outlier_dict):
    """
    Reconstructs df['all'] by concatenating g1, g2, and g3.
    Appends a 'Species' column to differentiate groups.
    """
    # Convert each group dictionary back into a DataFrame and append species labels
    df_g1 = pd.DataFrame(normed_no_outlier_dict["g1"])
    df_g2 = pd.DataFrame(normed_no_outlier_dict["g2"])
    df_g3 = pd.DataFrame(normed_no_outlier_dict["g3"])

    # Append species labels (1, 2, 3)
    df_g1["Species"] = 1
    df_g2["Species"] = 2
    df_g3["Species"] = 3

    # Concatenate all groups into one DataFrame
    df_all = pd.concat([df_g1, df_g2, df_g3], ignore_index=True)

    # Convert back to dictionary format
    return df_all.to_dict(orient="list")


###############################
##  Iris Distance Functions  ##
###############################

def minkowski_distance(data, p=3):
    # Number of data points
    n = len(data['Sepal length'])
    
    # Create an array of shape (n, 4) where each row represents one flower's features
    features = np.column_stack((
        data['Sepal length'],
        data['Sepal width'],
        data['Petal length'],
        data['Petal width']
    ))
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise diff then use to calc minkowski
    for i in range(n):
        for j in range(i, n):
            diff = abs(features[i] - features[j])
            dist = np.sum(diff**p)**(1 / p)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def chebyshev_distance(data):
    # Number of data points
    n = len(data['Sepal length'])
    
    # Create an array of shape (n, 4) where each row represents one flower's features
    features = np.column_stack((
        data['Sepal length'],
        data['Sepal width'],
        data['Petal length'],
        data['Petal width']
    ))
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise difference then take max feature
    for i in range(n):
        for j in range(i, n):
            diff = abs(features[i] - features[j])
            dist = max(diff)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def manhatten_distance(data):
    # Number of data points
    n = len(data['Sepal length'])
    
    # Create an array of shape (n, 4) where each row represents one flower's features
    features = np.column_stack((
        data['Sepal length'],
        data['Sepal width'],
        data['Petal length'],
        data['Petal width']
    ))
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise Manhatten distances
    for i in range(n):
        for j in range(i, n):
            diff = features[i] - features[j]
            dist = np.sum(np.abs(diff))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def euclidean_distance(data):
    # Number of data points
    n = len(data['Sepal length'])
    
    # Create an array of shape (n, 4) where each row represents one flower's features
    features = np.column_stack((
        data['Sepal length'],
        data['Sepal width'],
        data['Petal length'],
        data['Petal width']
    ))
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise Euclidean distances
    for i in range(n):
        for j in range(i, n):
            diff = features[i] - features[j]
            dist = np.sqrt(np.sum(diff**2))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix


##########################
##  Steam Preprocessing ##
##########################

def create_purchased_genre(plist, steamList):
    glist = {}
    ids = steamList['AppID']
    name_list = steamList['Name']
    genre_list = steamList['Genres']
    for person, item_list in plist.items():
        p_genres = []
        # print(f"{person}")
        for appid in item_list:
            idx = ids.index(appid)
            # print(f"game {name_list[idx]} has genre {genre_list[idx]}")
            p_genres.append(genre_list[idx])
        glist[person] = p_genres
    return glist

def make_binary_matrix(arr2d):
    """ Makes Binary Matrx from frequency Matrix 
        to calculate hamming distance and jaccard distance """
    binary_mat = []
    for row in arr2d:
        new_list = []
        # print(f"before: {row}")
        for x in row:
            if x > 0:
                new_list.append(1)
            else:
                new_list.append(0)
        # print(f"after: {new_list}")
        binary_mat.append(new_list)
    return binary_mat

def get_genre_set(steamListdata):
    return sorted(set(steamListdata['Genres']))

def get_friend_genre_counts(genre_set, friend_genres):
    """ Returns Genre Frequencies """
    f_genre_count = {}

    for friend, glist in friend_genres.items():
        counts = [0]*(len(genre_set))
        for item in glist:
            idx = genre_set.index(item) # gets index of genre in genre set
            counts[idx] += 1
        f_genre_count[friend] = counts
    # print(f"genre set {genre_set}")
    # print(f"f_genre_count {f_genre_count}")
    return f_genre_count

def get_gc_2darr(fdc):
    """ Makes 2d Matrix of genre frequencies from the 
        Friend Dictionary of Counts from get_friend_genre_counts """
    outer = [fdc['Person1'], fdc['Person2'],fdc['Person3'],fdc['Person4'],fdc['Person5']]
    return outer

def preprocess_steam(steamPurchaseData, steamListData):
    """ Preprocesses Steam Data into 2d Matrix """
    friend_genres = create_purchased_genre(steamPurchaseData, steamListData)
    genre_set = get_genre_set(steamListData)
    # print(f"genre_set {genre_set} has {len(genre_set)} items:")
    friend_genre_count = get_friend_genre_counts(genre_set, friend_genres)
    # print(friend_genre_count)
    return get_gc_2darr(friend_genre_count)

###############################
##  Steam Distance Functions ##
###############################

def hamming_distance(person1, person2):
    sum = 0
    for i in range(len(person1)):
        sum += (abs(person1[i]-person2[i]))
    return sum

# TODO: Finish this function, gets called in calc distance
def cosine_similarity(person1, person2):
    # Convert to numpy array
    p1 = np.array(person1)
    p2 = np.array(person2)

    # Calculate the numerator
    numerator = np.sum(p1 * p2)
    # Calculate the denominator
    denominator = np.sqrt(np.sum(p1 ** 2)) * np.sqrt(np.sum(p2 ** 2))

    cos_similarity = round((numerator / denominator), 4)

    return (cos_similarity)

def jaccard_similarity(person1, person2):
    """measures similary 
    larger value means more dissimilar"""

    intersect, union = 0, 0
    for i in range(len(person1)):
        intersect += (1 if person1[i]==1 and person2[i]==1 else 0)
        union += (1 if person1[i]==1 or person2[i]==1 else 0)
    
    # avoid div by zero, if they both don't have games then 
    # i guess that means they're similar? 
    # shouldn't hit this case anyways in the dataset
    if union == 0:
        return 1
    # print(f"intersect: {intersect}")
    # print(f"union: {union}")
    return (round((intersect/union),2))

def calc_distances(arr, arr2):
    jacc_mat = []
    ham_mat = []
    cos_mat = []
    # Jaccard Similarity and Hamming Distance Matrices
    for person1 in arr:
        j_m = []
        h_m = []
        for person2 in arr:
            j_m.append(jaccard_similarity(person1, person2))
            h_m.append(hamming_distance(person1, person2))
        jacc_mat.append(j_m)
        ham_mat.append(h_m)
    # Cosine Similarity Matrix
    for person1 in arr2:
        c_m = []
        for person2 in arr2:
            c_m.append(cosine_similarity(person1, person2))
        cos_mat.append(c_m)

    # Print results
    # print(f"\n\nJaccard Distances")
    # print(f"------------------")
    # print_2d_matrix(jacc_mat)
    # print(f"\n\nHamming Distances")
    # print(f"------------------")
    # print_2d_matrix(ham_mat)
    # print(f"\n\nCosine Similarity")
    # print(f"------------------")
    # print_2d_matrix(cos_mat)
    return jacc_mat, ham_mat, cos_mat

#########################
## Judge Preprocessing ##
#########################


def rank_judge_matrix(j_mat):
    ranked_mat = []
    for judge in j_mat:
        # print(f"judge: {judge}")
                #   0  1   2   3   4
        # judge: [ 89, 60, 55, 76, 40 ]
        sorted_judge = sorted(judge) # descending
                #   0   1   2   3   4
        # sorted: [ 89, 76, 60, 55, 40 ]
        ranked_judge = []
        for score in judge:
            i = sorted_judge.index(score)
            ranked_judge.append(i)
        # print(f"ranked judge: {ranked_judge}")
        ranked_mat.append(ranked_judge)
    return ranked_mat


def make_judge_matrix(judgeData):
    j_mat = [judgeData["Judge1"], judgeData["Judge2"], 
                judgeData["Judge3"], judgeData["Judge4"],
                    judgeData["Judge5"], judgeData["Judge6"]]
    return j_mat

###############################
##  Judge Distance Functions ##
###############################

def pearson_correlation_coefficient(data, rowvar):
    # Convert the input to a NumPy array (if it isn't one already)
    matrix = np.array(data)
    # print(matrix)
    
    # np.corrcoef, with the default rowvar=True, computes the correlation among rows
    corr_matrix = np.corrcoef(matrix, rowvar=rowvar)
    
    # when rowvar == True, it's judge to judge
    if rowvar == True:
        names = [f"Judge{i+1}" for i in range(len(data))]
    else: # swimmers
        names = [f"Swimmer{i+1}" for i in range(len(data[0]))]

    # return corr_matrix
    return pd.DataFrame(corr_matrix, 
                        columns=names, 
                        index=names)


def compute_spearman_matrix(scores, axis="Judge"):
    """
    Computes the Spearman rank correlation matrix for an n × m score matrix.
    
    :param scores: A NumPy array of shape (n_judges, m_swimmers).
    :return: A DataFrame of Spearman correlations (n × n matrix).
    """
    n_judges = scores.shape[0]
    spearman_matrix = np.ones((n_judges, n_judges))  # Initialize with 1s (self-correlation)

    for i in range(n_judges):
        for j in range(i + 1, n_judges):  # Only compute upper triangle
            correlation, _ = spearmanr(scores[i], scores[j])  # Spearman correlation
            spearman_matrix[i, j] = correlation
            spearman_matrix[j, i] = correlation  # Symmetric matrix

    if axis == "Judge":
        names = [f"Judge{i+1}" for i in range(n_judges)]
    elif axis == "Swimmer":
        names = [f"Swimmer{i+1}" for i in range(n_judges)]
    else:
        print("error in nameing convention")

    return pd.DataFrame(spearman_matrix, 
                        columns=names, 
                        index=names)



##################################
##       Helper Functions       ##
##################################

def plot_distance_heatmap(data, title="Heatmap", cmap="Reds", vmin=None, vmax=None, titles=None, annot=False):
    # Set size
    plt.figure(figsize=(12,7))

    # Set up the heatmap using Seaborn
    ax = sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for readability

    # Add a title
    plt.title(title)

    # Display the heatmap
    plt.show()

def print_2d_matrix(mat):
    rows = len(mat)
    print(f"\n[{mat[0]}")
    for i in range(1,rows-1):
        print(f" {mat[i]}")
    print(f" {mat[rows-1]}]\n")


def main():

    #################################
    #######   Iris Stuff  ###########
    #################################
    irisData = read_iris(iris_path)
    normed_no_outlier_dict = preprocess_iris(irisData) # has 3 groups not together
    df_all_reconstructed = reconstruct_all(normed_no_outlier_dict) # merged back to 'all' equivalent

    ## Calculate Distances
    data = irisData['all']

    # Euclidean
    euc_dist = euclidean_distance(data)
    euc_dist2 = euclidean_distance(df_all_reconstructed)

    # Manhattan
    man_dist = manhatten_distance(data)
    man_dist2 = manhatten_distance(df_all_reconstructed)
    
    # Chebyshev
    cheb_dist = chebyshev_distance(data)
    cheb_dist2 = chebyshev_distance(df_all_reconstructed)

    # Minkowski
    minkow_dist = minkowski_distance(data, 3)
    minkow_dist2 = minkowski_distance(df_all_reconstructed, 3)
    minkow_dist3 = minkowski_distance(data, 4)
    minkow_dist4 = minkowski_distance(df_all_reconstructed, 4)

    #   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo
    #   oOo~~>     Plot Heatmaps       <~~oOo
    #   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo
    plot_distance_heatmap(euc_dist, "Iris Euclidean Distance Heatmap (non normalized)", "jet")
    plot_distance_heatmap(euc_dist2, "Iris Euclidean Distance Heatmap (normalized)", "jet")
    plot_distance_heatmap(man_dist, "Iris Manhattan Distance Heatmap (non normalized)", "jet")
    plot_distance_heatmap(man_dist2, "Iris Manhattan Distance Heatmap (normalized)", "jet")
    plot_distance_heatmap(cheb_dist, "Iris Chebyshev Distance Heatmap (non normalized)", "jet")
    plot_distance_heatmap(cheb_dist2, "Iris Chebyshev Distance Heatmap (normalized)", "jet")
    plot_distance_heatmap(minkow_dist, "Iris Minkowski Distance Heatmap (non normalized)", "jet")
    plot_distance_heatmap(minkow_dist2, "Iris Minkowski Distance Heatmap (normalized)", "jet")
    plot_distance_heatmap(minkow_dist3, "Iris Minkowski Distance Heatmap P=4 (non normalized)", "jet")
    plot_distance_heatmap(minkow_dist4, "Iris Minkowski Distance Heatmap P=4 (normalized)", "jet")

    ## Should we show data summary?

    #################################
    #######   Steam Stuff ###########
    #################################
    steamPurchaseData = read_steamPurchase(steam_purch_path)
    steamListData = read_steamList(steam_list_path)
    genre_2d = preprocess_steam(steamPurchaseData, steamListData)
    # print("\n-- Player Genre Count Matrix --")
    # print_2d_matrix(genre_2d)
    binary_2d = make_binary_matrix(genre_2d)
    # print("-- Player Genre Count Binary Matrix --")
    # print_2d_matrix(binary_2d)
    jac, ham, cos = calc_distances(binary_2d, genre_2d)


    ##   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo
    ##   oOo~~>     Plot Heatmaps       <~~oOo
    ##   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo

    plot_distance_heatmap(jac, "Jaccard Similarity Heatmap", "jet", 0, annot=True)
    plot_distance_heatmap(ham, "Hamming Distance Heatmap", "jet_r", 0, 6, annot=True)
    plot_distance_heatmap(cos, "Cosine Similarity Heatmap", "jet", 0, 1, annot=True)
    

    #########################################
    #######  Swimming Judge Stuff ###########
    #########################################
    judgeData = read_judges(judge_path)
    judge_mat = make_judge_matrix(judgeData["Judges"])
    # print(f" -- Judge Matrix -- ")
    # print_2d_matrix(judge_mat)

    pcc_mat_judges = pearson_correlation_coefficient(judge_mat, True)
    pcc_mat_swimmers = pearson_correlation_coefficient(judge_mat, False)

    judge_mat = np.array(judge_mat)
    swimmer_mat = judge_mat.T
    scc_mat_judges = compute_spearman_matrix(judge_mat)
    scc_mat_swimmers =  compute_spearman_matrix(swimmer_mat, axis="Swimmer")

    
    ##   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo
    ##   oOo~~>     Plot Heatmaps       <~~oOo
    ##   oOoOooOooOoOoOoOooOooOoOoOoOooOooOoOo
# 
    plot_distance_heatmap(pcc_mat_judges, "Judge v Judge PCC", "jet", -1, 1, annot=True)
    plot_distance_heatmap(scc_mat_judges, "Judge v Judge SCC", "jet", -1, 1, annot=True)
    plot_distance_heatmap(pcc_mat_swimmers, "Swimmer v Swimmer PCC", "jet", -1, 1, annot=True)
    plot_distance_heatmap(scc_mat_swimmers, "Swimmer v Swimmer SCC", "jet", -1, 1, annot=True)

if __name__ == "__main__":
    main()