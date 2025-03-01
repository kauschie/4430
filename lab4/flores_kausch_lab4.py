import dcor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    

if __name__ == "__main__":
    main()