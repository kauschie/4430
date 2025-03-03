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
    
def load_grape_data(file_path):
    try:
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

def normalize_df(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def random_without_replacement():
    pass

def random_with_replacement():
    pass

def main():
    """Main execution function."""
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
    print(f"confidence interval: {stats['ci'][0]}-{stats['ci'][1]}")

if __name__ == "__main__":
    main()