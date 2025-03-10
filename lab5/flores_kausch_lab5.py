import dcor
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


##################################
##       Helper Functions       ##
##################################

def update_epoch_f(epoch_f:dict, indices):
    
    for idx in indices:
        val = epoch_f.get(idx,0)
        epoch_f[idx] = val+1

    return epoch_f

def get_num_blank(epoch_f:dict, num_samples:int):
    return (len(epoch_f) - num_samples)

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

def do_striated_sampling(df:pd.DataFrame, samp_rate:float):
    pass

def do_random_sampling(df:pd.DataFrame, samp_rate:float):
    num_runs = 100
    n = df["diameter"].size

    ## without replacement
    means_wo = []
    stdevs_wo = []
    epoch_f_wo = {}
    print("\n------  Without Replacements ------")
    for _ in range(num_runs):
        indices = get_random_array(n, int(samp_rate*n), False) 
        epoch_f_wo = update_epoch_f(epoch_f_wo, indices)
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
    num_not_sampled = n - len(epoch_f_wo)

    print(f"Mean of means: {means_wo.mean()}")
    print(f"Stdev of means: {means_wo.std()}")
    print(f"Mean stdev: {stdevs_wo.mean()}")
    print(f"Stdev of stdevs: {stdevs_wo.std()}")
    print(f"Number of samples not-sampled (sr: {samp_rate*100}%): {num_not_sampled}")

    ## TODO: Box and Whisker From this Data

    ## with replacement

    print("\n\n-------  With Replacements -------")
    means_w = []
    stdevs_w = []
    freq_of_freqs = {}
    epoch_f_w = {}
    for i in range(num_runs):
        indices = get_random_array(n, int(samp_rate*n), True)
        epoch_f_w = update_epoch_f(epoch_f_w, indices)
        count_dict = get_index_freq(indices)
        # print(f"{i} - len: {len(count_dict)} - count_dict: {count_dict}")        
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
    num_not_sampled = n - len(epoch_f_w)

    print(f"Mean of means: {means_w.mean()}")
    print(f"Stdev of means: {means_w.std()}")
    print(f"Mean stdev: {stdevs_w.mean()}")
    print(f"Stdev of stdevs: {stdevs_w.std()}")
    print(f"Number of samples not-sampled (sr: {samp_rate*100}%): {num_not_sampled}")
    
    # TODO: Box and Whisker from this data

    print(f"freq_of_freqs: {freq_of_freqs}")

    # TODO: Histogram from freq_of_freqs data



def main():
    """Main execution function."""

    ## Question 1: Random Sample 10% of the Data, how many are sampled twice?
    df_sampling = load_grape_data("./DataSampling_label_grape_data.txt", use_space=True)
    do_random_sampling(df_sampling, 0.1)


    ## Question 2: Random Sample 2% of the Data, 
    do_random_sampling(df_sampling, 0.02)

    # how many ARE NOT sampled at all? # done for both 10% and 2%
    

    ## Question 3: Mean and Stdev by


    ## Part A: Striated Sampling - Use Box Number and compare them?


    ## Part B: Systematic Sampling - sample every N rows, test a few different numbers and compare them?

if __name__ == "__main__":
    main()