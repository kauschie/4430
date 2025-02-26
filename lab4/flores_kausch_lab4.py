import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def load_army_data(file_path):
    """Loads CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ["feature 1", "feature 2", "feature 3"]
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def compute_correlations(df):
    """Computes Pearson and Spearman correlation matrices."""
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")
    return pearson_corr, spearman_corr

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
        print("Pearson Correlation:\n", pcc_matrix)
        print("\nSpearman Correlation:\n", src_matrix)

        # Heatmaps
        plot_distance_heatmap(pcc_matrix, title="Army Features PCC Heatmap", vmin=-1, vmax=1, titles=df.columns, annot=True)
        plot_distance_heatmap(src_matrix, title="Army Features SRC Heatmap", vmin=-1, vmax=1, titles=df.columns, annot=True)
    else:
        print("Failed to load data.")

    

if __name__ == "__main__":
    main()