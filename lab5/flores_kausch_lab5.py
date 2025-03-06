import dcor
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
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


def plot_multiple_box_strip_subplots(data_arrays, labels=None, title="Box and Strip Plots", ylabel="Values"):
    """
    Plots multiple vertical box and whisker plots with overlaid strip plots using subplots.

    Parameters:
    - data_arrays (list of np.array): A list of NumPy arrays containing numerical data.
    - labels (list of str): Labels for each dataset (default: None).
    - title (str): Title of the overall figure (default: "Box and Strip Plots").
    - ylabel (str): Label for the y-axis (default: "Values").
    """
    num_datasets = len(data_arrays)
    
    # Default labels if none provided
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(num_datasets)]
    
    fig, axes = plt.subplots(ncols=num_datasets, figsize=(num_datasets * 3, 6), sharey=True)

    for i, (data, label) in enumerate(zip(data_arrays, labels)):
        ax = axes[i] if num_datasets > 1 else axes  # Handle single subplot case

        # Boxplot
        sns.boxplot(y=data, ax=ax, color="lightblue", width=0.5)

        # Stripplot
        sns.stripplot(y=data, ax=ax, color="red", alpha=0.6, jitter=True, size=6)

        # Labels
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Overall title
    plt.suptitle(title)
    plt.ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit title
    plt.show()


##################################
##       Helper Functions       ##
##################################

def plot_distance_heatmap(data, title="Heatmap", colorbar_title="Correlation", cmap="jet", vmin=None, vmax=None, titles=None, annot=False):
    # Dynamically adjust figure size
    plt.figure(figsize=(max(8, 1.5 * data.shape[1]), max(6, 1.5 * data.shape[0])))

    # Set up the heatmap using Seaborn
    ax = sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': colorbar_title})
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)

    # Plot Title
    plt.title(title, fontsize=14, pad=20)

    # Optimize layout to prevent clipping
    plt.tight_layout()

    # Display the heatmap
    plt.show()

def plot_frequency_histogram(freq_dict, title="Frequency Histogram", xlabel="Unique Sample Sizes", ylabel="Frequency"):
    """
    Plots a histogram from a dictionary of frequencies.

    Parameters:
    - freq_dict (dict): A dictionary where keys represent unique values, and values represent their frequency.
    - title (str): The title of the histogram (default: "Histogram of Frequency of Frequencies").
    - xlabel (str): Label for the x-axis (default: "Unique Sample Sizes").
    - ylabel (str): Label for the y-axis (default: "Frequency").
    """
    # Get the min and max values from the keys
    min_val = min(freq_dict.keys())
    max_val = max(freq_dict.keys())

    # Generate a continuous range of integers from min to max
    full_x_values = list(range(min_val, max_val + 1))

    # Fill missing values with zero
    frequencies = [freq_dict.get(x, 0) for x in full_x_values]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.bar(full_x_values, frequencies, color="skyblue", edgecolor="black", alpha=0.7)

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Set x-axis ticks to be only the unique sample sizes
    plt.xticks(full_x_values)

    # Display grid for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Show the plot
    plt.show()

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

    print(f"freq_of_freqs: {freq_of_freqs}")

    # Visualizations

    # Boxplots
    mean_title = "With vs Without Replacement Boxplot of Means (Sample rate: " + str(samp_rate) + ")"
    stdevs_title = "With vs Without Replacement Boxplot of Stdevs (Sample rate: " + str(samp_rate) + ")"
    means = [means_wo, means_w]
    stdevs = [stdevs_wo, stdevs_w]
    mean_labels = ["Without Replacement", "With Replacement"]
    stdevs_labels = ["Without Replacement", "With Replacement"]

    plot_multiple_box_strip_subplots(means, mean_labels, mean_title, "Mean Diameter (mm)")
    plot_multiple_box_strip_subplots(stdevs, stdevs_labels, stdevs_title, "Standard Deviation (mm)")

    # Histogram
    frequency_plot_title = "Sampling Duplicates Distribution (Sample rate: " + str(samp_rate) + ")"
    plot_frequency_histogram(freq_of_freqs, title=frequency_plot_title, xlabel="Duplicate Samples", ylabel="Count")

def stratified_sampling(df, sample_rate, stratify_cols=None):
    """
    Performs stratified sampling on a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The dataset to sample from.
    - sample_rate (float): The fraction of the dataset to sample (between 0 and 1).
    - stratify_cols (list of str, optional): Column(s) to stratify by (SQL-like selection).
    
    Returns:
    - pd.DataFrame: A sampled DataFrame based on the given stratification.
    """
    if not (0 < sample_rate <= 1):
        raise ValueError("Sample rate must be between 0 and 1.")

    if stratify_cols:
        # Ensure valid column names
        for col in stratify_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Perform stratified sampling
        sampled_df = df.groupby(stratify_cols, group_keys=False)\
            .apply(lambda x: x.sample(n=min(len(x), max(1, int(len(x) * sample_rate)))), include_groups=False)
    else:
        # If no stratification columns, do simple random sampling
        sampled_df = df.sample(frac=sample_rate, random_state=42)

    return sampled_df.reset_index(drop=True)

# TODO Strategic Sampling
def strategic_sampling(df:pd.DataFrame, sample_rate:float, offset:int):
    """
    Performs strategic sampling by selecting evenly spaced samples from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The dataset to sample from.
    - sample_rate (float): The fraction of the dataset to sample (between 0 and 1).

    Returns:
    - pd.DataFrame: A sampled DataFrame using strategic selection.
    """
    if not (0 < sample_rate <= 1):
        raise ValueError("Argument sample_rate must be between 0 and 1.")
    
    if not (0 == offset or 1 == offset):
        raise ValueError("Argument offset must be either 0 or 1")
    
    N = len(df)  # Total number of rows
    num_samples = max(1, int(N * sample_rate))  # Ensure at least 1 sample
    step_size = max(1, N // num_samples)  # Compute step size dynamically

    # Select every `step_size`-th row, starting at `offset`
    sampled_df = df.iloc[offset::step_size].reset_index(drop=True)

    return sampled_df


def main():
    """Main execution function."""

    df_sampling = load_grape_data("./DataSampling_label_grape_data.txt", use_space=True)

    sample_rate = 0.02

    do_sampling_stuff(df_sampling, 0.1)
    do_sampling_stuff(df_sampling, 0.02)



    # Stratified Sampling

    # Create empty list of means and stdevs
    strat_means  = np.array([])
    strat_stdevs = np.array([])

    for _ in range(100):
        # Get stratified samples
        stratified_samples_df = stratified_sampling(df_sampling, sample_rate, ["type"])
        # print(stratified_samples_df)

        # Find the mean and standard deviation of the samples
        strat_mean = stratified_samples_df["diameter"].mean()
        strat_stdev  = stratified_samples_df["diameter"].std()

        # print(strat_mean, strat_stdev)

        strat_means = np.append(strat_means, strat_mean)
        strat_stdevs = np.append(strat_stdevs, strat_stdev)

    # Stratified Samples Mean of Means and Mean of Stdev
    print(f"Mean of means: {np.mean(strat_means)}")
    print(f"Stdev of means: {np.std(strat_means)}")
    print(f"Mean stdev: {np.mean(strat_stdevs)}")
    print(f"Stdev of stdevs: {np.std(strat_stdevs)}")

    # Boxplot Visualization
    mean_title   = "Stratified Sampling Boxplot of Means (Sample Rate: " + str(sample_rate) + ")"
    stdev_title  = "Stratified Sampling Boxplot of Stdevs (Sample Rate: " + str(sample_rate) + ")"

    plot_multiple_box_strip_subplots([strat_means], title=mean_title, labels=[""], ylabel="Mean Diameter (mm)")
    plot_multiple_box_strip_subplots([strat_stdevs], title=stdev_title, labels=[""], ylabel="Standard Deviation (mm)")




    # Strategic Sampling

    # Sampling with no offset (starting at the first index)
    strategic_samples_df = strategic_sampling(df_sampling, 0.02, 0)
    print(f'\nMean of Strategic Samples (offset = 0): {strategic_samples_df["diameter"].mean()}')

    # Sampling with offset (starting at the second index)
    strategic_samples_df = strategic_sampling(df_sampling, 0.02, 1)
    print(f'Mean of Strategic Samples (offset = 1): {strategic_samples_df["diameter"].mean()}\n')

if __name__ == "__main__":
    main()