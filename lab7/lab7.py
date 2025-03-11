import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

training_path = "/home/mkausch/dev/4430/lab7/LR_armyTraining.csv"
testing_path = "/home/mkausch/dev/4430/lab7/LR_armyTesting.csv"
results_path = "/home/mkausch/dev/4430/lab7/LR_armyResults.csv"

# Given Data
x_vals = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
y_vals = [2.1, 4.7, 4.8, 6.6, 8.5, 9.9, 10.1, 10.9, 11.7, 13.1]

def read_training(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x_vals = [float(line.split(',')[0]) for line in lines]
        y_vals = [float(line.split(',')[1]) for line in lines]
    return x_vals, y_vals

def read_validation(ftesting, fresults):
    x_vals = []
    y_vals = []
    with open(ftesting, 'r') as f:
        lines = f.readlines()
        x_vals = [int(line[0]) for line in lines]
    with open(fresults, 'r') as f:
        lines = f.readlines()
        y_vals = [int(line[0]) for line in lines]
    return x_vals, y_vals

# Function Definitions
def y(x, a, b):
    return a * x + b

def find_slope_range(x_vals, y_vals, percentile=95):
    slopes = [
        (y_vals[i+1] - y_vals[i]) / (x_vals[i+1] - x_vals[i])
        for i in range(len(x_vals) - 1) if x_vals[i+1] != x_vals[i]
    ]
    
    if not slopes:
        return (0, 0)  # If no valid slopes, return 0

    # Remove extreme outliers by using percentiles
    m_min = np.percentile(slopes, 100 - percentile)  # 5th percentile (lower bound)
    m_max = np.percentile(slopes, percentile)  # 95th percentile (upper bound)

    return m_min, m_max


def find_intercept_range(x_vals, y_vals, m_min, m_max):
    y_min, y_max = min(y_vals), max(y_vals)
    x_min, x_max = min(x_vals), max(x_vals)

    b_min = y_min - m_max * x_min  # Using max slope
    b_max = y_max - m_min * x_max  # Using min slope

    return b_min, b_max



def sse(x_vals, y_vals, m, b):
    return sum((y_vals[i] - y(x_vals[i], m, b))**2 for i in range(len(x_vals)))

def sae(x_vals, y_vals, m, b):
    return sum(abs(y_vals[i] - y(x_vals[i], m, b)) for i in range(len(x_vals)))


def heatmap(data, x_labels, y_labels, title="Heatmap", colorbar_title="Error", cmap="inferno", vmin=None, vmax=None, global_min = None):
    plt.figure(figsize=(12, 7))

    # Determine the number of ticks to display
    num_x_ticks = 10  # Reduce the number of x-axis labels
    num_y_ticks = 10  # Reduce the number of y-axis labels

        # Find minimum value and its index
    min_index = np.unravel_index(np.argmin(data), data.shape)
    # min_m, min_b = x_vals[min_index[1]], y_vals[min_index[0]]  # Convert indices to (m, b)

    x_tick_positions = np.linspace(0, len(x_labels) - 1, num_x_ticks, dtype=int)
    y_tick_positions = np.linspace(0, len(y_labels) - 1, num_y_ticks, dtype=int)

    x_tick_labels = [f"{x_labels[i]:.2f}" for i in x_tick_positions]
    y_tick_labels = [f"{y_labels[i]:.2f}" for i in y_tick_positions]

    ax = sns.heatmap(data, annot=False, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                     cbar_kws={'label': colorbar_title})

    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)

    ax.scatter(min_index[1], min_index[0], color='red', marker='x', s=50, label=f"Min Error: (m={min_index[1]:.3f}, b={min_index[0]:.3f})")


    plt.xlabel("Slope (m)")
    plt.ylabel("Intercept (b)")
    plt.title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()

def plot_surface(M, B, Z, title="Surface Plot", xlabel="X", ylabel="Y", zlabel="Z"):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, B, Z, cmap='jet', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()


def main():
    # Read training data
    # x_vals, y_vals = read_training(training_path)
    print(f"Training Data: {x_vals}, {y_vals}")

    # Compute min/max slope with outlier filtering
    min_a, max_a = find_slope_range(x_vals, y_vals)

    # Compute min/max intercept based on the dataset
    min_b, max_b = find_intercept_range(x_vals, y_vals, min_a, max_a)

    # Define precision dynamically based on range size
    a_precision = (max_a - min_a) / 1000
    b_precision = (max_b - min_b) / 1000

    print(f"Precision: {a_precision:.6f}, {b_precision:.6f}")
    print(f"Min a: {min_a:.6f}, Max a: {max_a:.6f}")
    print(f"Min b: {min_b:.6f}, Max b: {max_b:.6f}")

    # Define grid of (m, b) values
    m_range = np.linspace(min_a, max_a, int((max_a - min_a) / a_precision))
    b_range = np.linspace(min_b, max_b, int((max_b - min_b) / b_precision))
    
    M, B = np.meshgrid(m_range, b_range)  # Create mesh grid for surface plot
    SSE = np.zeros(M.shape)  # Store SSE values
    SAE = np.zeros(M.shape)  # Store SAE values

    min_sae = (None, None, float('inf'))
    min_sse = (None, None, float('inf'))

    progress = tqdm(total=M.shape[0] * M.shape[1], desc="Computing Errors", dynamic_ncols=True)

    # Compute Error for each (m, b) pair
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):  # FIX: Use M.shape[1] instead of B.shape[1]
            m = M[i, j]
            b = B[i, j]
            mae = sae(x_vals, y_vals, m, b)
            mse = sse(x_vals, y_vals, m, b)

            SSE[i, j] = mse
            SAE[i, j] = mae

            if mae < min_sae[2]:
                min_sae = (m, b, mae)
            if mse < min_sse[2]:
                min_sse = (m, b, mse)
            progress.update(1)

        progress.set_description(
            f"SAE({min_sae[0]:.3f}, {min_sae[1]:.3f})={min_sae[2]:.6f} | "
            f"SSE({min_sse[0]:.3f}, {min_sse[1]:.3f})={min_sse[2]:.6f}"
        )

    progress.close()

    print("\nFinal Results:")
    print(f"Min SAE: m = {min_sae[0]:.3f}, b = {min_sae[1]:.3f}, error = {min_sae[2]:.6f}")
    print(f"Min SSE: m = {min_sse[0]:.3f}, b = {min_sse[1]:.3f}, error = {min_sse[2]:.6f}")

    # Generate heatmaps
    heatmap(SAE, m_range, b_range, title="SAE Heatmap", colorbar_title="Sum of Absolute Errors", cmap="inferno", global_min=min_sae)
    heatmap(SSE, m_range, b_range, title="SSE Heatmap", colorbar_title="Sum of Squared Errors", cmap="inferno", global_min=min_sse)

    # Plot SAE Surface
    plot_surface(M, B, SAE, title="Sum of Absolute Errors (SAE) Surface Plot", 
                 xlabel="Slope (a)", ylabel="Intercept (b)", zlabel="SAE")

    # Plot SSE Surface
    plot_surface(M, B, SSE, title="Sum of Squared Errors (SSE) Surface Plot",
                 xlabel="Slope (a)", ylabel="Intercept (b)", zlabel="SSE")


    num_decimals = 5
    # Read Validation Data
    x_valid, y_valid = read_validation(testing_path, results_path)
    valid_sae = sae(x_valid, y_valid, min_sae[0], min_sae[1])
    valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    print(f"\nValidation Results:")
    print(f"Min SAE: m = {min_sae[0]:.{num_decimals}f}, b = {min_sae[1]:.{num_decimals}f}, error = {valid_sae:.6f}")
    print(f"Min SSE: m = {min_sse[0]:.{num_decimals}f}, b = {min_sse[1]:.{num_decimals}f}, error = {valid_sse:.6f}")

if __name__ == "__main__":

    main()
