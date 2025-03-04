import dcor
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm, mannwhitneyu
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

##################################
##   Load data from CSV Files   ##
##################################

def load_army_data(file_path):
    """Loads CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ["feature 1", "feature 2", "feature 3"]
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
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
    

def read_ds(filepath):
    with open(filepath, 'r', newline = '') as file:
        distances = []
        boxes = []
        i = 0
        for row in file:
            if i == 0:
                distances = row.split()
                i += 1
            elif i == 1:
                boxes = row.strip().split()
        # print(distances)
        # print(box)
        return [float(d) for d in distances], [int(b) for b in boxes]


##########################################
##   Statistical Dependency Functions   ##
##########################################

def compute_correlations(df):
    """Computes Pearson and Spearman correlation matrices."""
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")
    return pearson_corr, spearman_corr

def compute_distance_correlation(df):
    """Computes Distance Correlation between each pair of features."""

    # Convert the DataFrame to float64 explicitly
    df = df.astype(np.float64)

    num_features = df.shape[1]  # Number of columns
    distance_corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    for i in range(num_features):
        for j in range(num_features):
            x, y = df.iloc[:, i], df.iloc[:, j]
            distance_corr_matrix.iloc[i, j] = dcor.distance_correlation(x, y)

    return distance_corr_matrix.astype(float)


#################################
###    Stat Functions
##############################


def z_test(sample, pop_mean, pop_sigma):
    sample = np.array(sample)
    sample_mean = np.mean(sample)
    n = len(sample)

    z_stat = (sample_mean - pop_mean) / (pop_sigma / np.sqrt(n))
    p_value = 2 * (1-norm.cdf(abs(z_stat)))

    return z_stat, p_value


def single_observation_z_test(x, mu_0, sigma, alpha=0.05):
    """
    Check if a single observation x differs significantly
    from a hypothesized normal distribution (mean=mu_0, sd=sigma).

    Returns:
        z_value: float
        p_value: float
        significant: bool (True if p < alpha)
    """
    # Compute z
    z_value = (x - mu_0) / sigma
    
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z_value)))
    
    # Compare p-value with significance level alpha
    significant = (p_value < alpha)
    
    return z_value, p_value, significant



def confidence_interval(df: pd.DataFrame, column: str, confidence: float = 0.95):
    """
    Computes the confidence interval for a given column in a Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute the confidence interval.
        confidence (float): The confidence level (default is 95%).
        
    Returns:
        tuple: The lower and upper bounds of the confidence interval.
    """
    data = df[column].dropna().values  # Drop NaN values
    mean = np.mean(data)
    stdev = np.std(data)
    std_err = stats.sem(data)  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom
    
    # Use t-distribution for small sample sizes, normal distribution for large samples
    ci = stats.t.interval(confidence, df, loc=mean, scale=std_err)
    

    return {"mean":mean, "stdev":stdev, "ci":ci}

def confidence_interval2(data, confidence: float = 0.95):
    """
    Computes the confidence interval for a given column in a Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute the confidence interval.
        confidence (float): The confidence level (default is 95%).
        
    Returns:
        tuple: The lower and upper bounds of the confidence interval.
    """
    # data = df[column].dropna().values  # Drop NaN values
    mean = data.mean()
    stdev = data.std()
    std_err = stats.sem(data)  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom
    
    # Use t-distribution for small sample sizes, normal distribution for large samples
    ci = stats.t.interval(confidence, df, loc=mean, scale=std_err)
    

    return {"mean":mean, "stdev":stdev, "ci":ci}




#### Plotting Functions

def plot_3d_scatter(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c='b', marker='o')

    # Labels
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    ax.set_title('3D Scatter Plot of Normalized Features')

    plt.show()


def plot_2d_scatter(df, x_col, y_col):
    """
    Plots a 2D scatter plot for two selected columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The normalized dataset
    x_col (str): Name of the column for the x-axis
    y_col (str): Name of the column for the y-axis
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], c='b', alpha=0.6, edgecolors='k')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'2D Scatter Plot: {x_col} vs {y_col}')
    plt.grid(True)
    
    plt.show()

def plot_distance_heatmap(data, title="Heatmap", colorbar_title="Correlation", cmap="jet", vmin=None, vmax=None, titles=None, annot=False, labels=None):
    # Dynamically adjust figure size

    # Set up the heatmap using Seaborn
    if labels == None:
        plt.figure(figsize=(max(8, 1.5 * data.shape[1]), max(6, 1.5 * data.shape[0])))
        ax = sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': colorbar_title})
    else:
        plt.figure(figsize=(12,7))
        ax = sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)

    # Plot Title
    plt.title(title, fontsize=14, pad=20)

    # Optimize layout to prevent clipping
    plt.tight_layout()

    # Display the heatmap
    plt.show()

def plot_multiple_samples_vs_hypothesized_mean(samples, box_names, hypoth_mean):
    """
    Plot multiple samples with a line for the hypothesized mean.
    
    Parameters:
    -----------
    hypoth_mean : float
        The hypothesized mean (will be plotted as a horizontal line).
    """
    plt.figure(figsize=(7, 5))

    for i in range(len(samples)):
        data = np.array(samples[i])
        group_name = box_names[i]
        
        # Slight jitter on the x-axis so points don't overlap vertically
        x_vals = np.random.normal(loc=i + 1, scale=0.06, size=len(data))
        
        # Plot the raw data
        plt.scatter(x_vals, data, alpha=0.7, label=f"{group_name} data")
        
        # Plot the sample mean
        sample_mean = np.mean(data)
        plt.scatter(i + 1, sample_mean, color='red', s=100, marker='D',
                    edgecolor='black', zorder=3,
                    label=f"{group_name} mean = {sample_mean:.2f}")

    # Plot the hypothesized mean as a reference line
    plt.axhline(hypoth_mean, color='green', linestyle='--',
                label=f"Hypothesized mean = {hypoth_mean}")

    # Cosmetic adjustments for x-axis
    # plt.xlim(0.5, len(samples) + 0.5)
    # plt.xticks(range(1, len(samples) + 1), [s["name"] for s in samples])
    plt.ylabel("Value")
    plt.title("Box Data vs. Merlot Mean")

    # Place the legend outside the main plot area
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_z_test(sample, mu_0):

    sample_mean = np.mean(sample)
    n = len(sample)

    # Plot the raw data as a dot/strip
    plt.scatter(np.ones(n), sample, alpha=0.7, color='blue', label='Sample data')

    # Plot the sample mean
    plt.scatter(1, sample_mean, color='red', s=100, marker='D', zorder=3, 
                label=f'Sample mean = {sample_mean:.2f}')

    # Add a horizontal line for the hypothesized mean
    plt.axhline(mu_0, color='green', linestyle='--', label=f'Hypothesized mean = {mu_0}')

    # A little cosmetic offset so dots aren't hidden
    plt.xlim(0.8, 1.2)  
    plt.legend()
    plt.title("One-Sample Visualization vs. Hypothesized Mean")
    plt.ylabel("Value")
    plt.xticks([])
    plt.show()



##################################
##       Helper Functions       ##
##################################


def normalize_df(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def get_random_array(n:int, sample_size:int, replacement:bool):
    sample_indices = np.random.choice(n, size=sample_size, replace=replacement)
    return sample_indices

def get_index_freq(indices):
    # Get unique indices and their counts
    unique_indices, counts = np.unique(indices, return_counts=True)
    
    # filter where count > 1
    duplicate_indices = unique_indices[counts > 1]
    duplicate_counts = counts[counts > 1]

    dup_dict = dict(zip(duplicate_indices, duplicate_counts))
    return dup_dict

def do_sampling_stuff(df:pd.DataFrame, samp_rate:float):
    mult_repl_freq = {}
    num_runs = 100
    n = df["diameter"].size

    ## without replacement
    means_wo = []
    stdevs_wo = []
    print("\n------  Without Replacements ------")
    for _ in range(num_runs):
        indices = get_random_array(n, int(samp_rate*n), False) 
        rand_data = df.iloc[indices]['diameter']
        # print(rand_data)
        stats = confidence_interval2(rand_data, 0.95)
        mean = stats['mean']
        stdev = stats['stdev']
        # print(f"Mean: {mean}, STDEV: {stdev} Interval: {stats['ci'][0]} --> {stats['ci'][1]}")
        means_wo.append(mean)
        stdevs_wo.append(stdev)
    
    means_wo = np.array(means_wo)
    stdevs_wo = np.array(stdevs_wo)

    print(f"Mean of means: {means_wo.mean()}")
    print(f"Stdev of means: {means_wo.std()}")
    print(f"Mean stdev: {stdevs_wo.mean()}")
    print(f"Stdev of stdevs: {stdevs_wo.std()}")

    ## TODO: Box and Whisker From this Data

    ## with replacement

    print("\n\n-------  With Replacements -------")
    means_w = []
    stdevs_w = []
    freq_of_freqs = {}
    for i in range(num_runs):
        indices = get_random_array(n, int(samp_rate*n), True)
        count_dict = get_index_freq(indices)
        print(f"{i} - len: {len(count_dict)} - count_dict: {count_dict}")        
        rand_data = df.iloc[indices]['diameter']
        stats = confidence_interval2(rand_data, 0.95)
        mean = stats['mean']
        stdev = stats['stdev']
        means_w.append(mean)
        stdevs_w.append(stdev)
        f = freq_of_freqs.get(len(count_dict),0)
        freq_of_freqs[len(count_dict)] = f + 1

    means_w = np.array(means_w)
    stdevs_w = np.array(stdevs_w)

    print(f"Mean of means: {means_w.mean()}")
    print(f"Stdev of means: {means_w.std()}")
    print(f"Mean stdev: {stdevs_w.mean()}")
    print(f"Stdev of stdevs: {stdevs_w.std()}")
    
    # TODO: Box and Whisker from this data

    print(f"freq_of_freqs: {freq_of_freqs}")

    # TODO: Histogram from freq_of_freqs data


def run_question_one():
    path = './StatisticalTests_boxed_grape_data.txt'
    distances, boxes = read_ds(path)


    groups = []
    group = []
    old_group_num = None
    for i in range(len(distances)):
        if old_group_num != boxes[i]:
            old_group_num = boxes[i]
            print(f"working on box {old_group_num}")
            if len(group) > 0:
                print(f"appending group {boxes[i]-1}")
                groups.append(group.copy())
                group.clear()
        group.append(distances[i])
    groups.append(group)

    for g in groups:
        print(f"len(g): {len(g)}")

    pop_mean = 9.5
    pop_std = 3

    # t_test
    t_stats = []
    p_values = []
    for box1 in groups:
        t_row = []
        p_row = []
        for box2 in groups:
            t, p = stats.ttest_ind(box1, box2, equal_var=False)
            t_row.append(t)
            p_row.append(p)
        t_stats.append(t_row)
        p_values.append(p_row)



    box_names = [f"Box{i+1}" for i in range(len(groups))]
    print(box_names)
    # print(p_values)

    binary_p_values = [[1 if pval < 0.05 else 0 for pval in row] for row in p_values]

    # print(binary_p_values)
    # binary_p_values.reverse()
    # print(binary_p_values)

    tt_array = np.array(t_stats)
    b_p_array = np.array(binary_p_values)
    p_array = np.array(p_values)

    plot_distance_heatmap(tt_array, title="Grape T-Test T Val", cmap="coolwarm", vmin=-4, vmax=4, labels=box_names)
    plot_distance_heatmap(b_p_array, title="Grape T-Test Binary P Val", cmap="coolwarm", labels=box_names)
    plot_distance_heatmap(p_array, title="Grape T-Test P Val", cmap="coolwarm", labels=box_names)


    z_vals = []
    p_vals_from_ztest = []
    for box in groups:
        z, p_from_ztest = z_test(box, pop_mean, pop_std)
        z_vals.append(z)
        p_vals_from_ztest.append(p_from_ztest)

    print(f"z_vals {z_vals}")
    print(p_vals_from_ztest)

    # plot_z_test(z_vals, pop_mean)
    plot_multiple_samples_vs_hypothesized_mean(groups, box_names, pop_mean)
    binary_ztest_p_values = [1 if pval < 0.05 else 0 for pval in p_vals_from_ztest]
    print(f"significant from z-test: {binary_ztest_p_values}")

    # Scatter Plot
    plt.scatter([i for i in range(len(binary_ztest_p_values))], binary_ztest_p_values, marker='o', alpha=1, label="Box")
    plt.title("Grape Box Z-test P-val") 
    plt.xlabel("Box")
    plt.ylabel("1=significant 0=Not")
    # plt.legend()
    plt.show()




    single_z = []
    single_z_p = []
    sig = []
    for point in distances:
        z, p, significant = single_observation_z_test(point,pop_mean, pop_std)
        single_z.append(z)
        single_z_p.append(p)
        sig.append(significant)


    # Scatter Plot
    ## 
    plt.scatter([i for i in range(len(single_z))], single_z, marker='o', alpha=1, label="sample")
    plt.title("Grape individual z-test") 
    plt.xlabel("Grape Sample")
    plt.ylabel("z-val")
    # plt.legend()
    plt.show()


    # Scatter Plot
    single_z_p_bin = [1 if pval < 0.05 else 0 for pval in single_z_p]
    plt.scatter([i for i in range(len(single_z_p_bin))], single_z_p_bin, marker='o', alpha=1, label="sample")
    plt.title("Grape individual p-value") 
    plt.xlabel("Grape Sample")
    plt.ylabel("p-val")
    # plt.legend()
    plt.show()



    ## Mann Whitey U Test

    # stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    # t_test
    u_stats = []
    u_p_values = []
    for box1 in groups:
        s_row = []
        p_row = []
        for box2 in groups:
            s, p = mannwhitneyu(box1, box2, alternative='two-sided')
            s_row.append(s)
            p_row.append(p)
        u_stats.append(s_row)
        u_p_values.append(p_row)

    binary_u_p_values = [[1 if pval < 0.05 else 0 for pval in row] for row in u_p_values]

    u_stats = np.array(u_stats)
    binary_u_p_values = np.array(binary_u_p_values)
    u_p_values = np.array(u_p_values)

    plot_distance_heatmap(u_stats, title="Grape U-Test U Val", cmap="coolwarm", labels=box_names)
    plot_distance_heatmap(binary_u_p_values, title="Grape U-Test Binary P Val", cmap="coolwarm", labels=box_names)
    plot_distance_heatmap(u_p_values, title="Grape U-Test P Val", cmap="coolwarm", labels=box_names)



def main():
    
    ## Question 1 stuff
    run_question_one()




    ## Question 2 stuff
    army_attribute_path = "army_attribute_data.csv"
    df = load_army_data(army_attribute_path)

    if df is not None:
        pcc_matrix, src_matrix = compute_correlations(df)
        distance_corr_matrix = compute_distance_correlation(df)

        print("Pearson Correlation:\n", pcc_matrix)
        print("\nSpearman Correlation:\n", src_matrix)
        print("\nDistance Correlation:\n", distance_corr_matrix)

        # Heatmaps
        plot_distance_heatmap(pcc_matrix, title="Army Features PCC Heatmap", vmin=-1, vmax=1, titles=df.columns, annot=True)
        plot_distance_heatmap(src_matrix, title="Army Features SRC Heatmap", vmin=-1, vmax=1, titles=df.columns, annot=True)
        plot_distance_heatmap(distance_corr_matrix, title="Distance Correlation Heatmap", vmin=0, vmax=1, titles=df.columns, annot=True)

    else:
        print("Failed to load data.")


    # min/max normalization on data set
    df_norm = normalize_df(df)
    print(df_norm)
    plot_3d_scatter(df_norm)        # viewed all 3 together 3d
    plot_2d_scatter(df, "feature 1", "feature 2") # can see positive trend in data
    plot_2d_scatter(df, "feature 1", "feature 3") # can see their linear relationship creates a line
    plot_2d_scatter(df, "feature 2", "feature 3") # can see positive trend in data



    ##########
    ## confidence interval
    ##########

    # Lab - stitistical tools dataset 
    df = load_grape_data("./StatisticalTools_grape_data.txt")
    stats = confidence_interval(df, "diameter")
    print(f"Mean: {stats['mean']}")
    print(f"Stdev: {stats['stdev']}")
    print(f"confidence interval: {stats['ci'][0]} --> {stats['ci'][1]}")

if __name__ == "__main__":
    main()