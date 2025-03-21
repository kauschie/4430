from decimal import Decimal, getcontext
# Set the precision of Decimal arithmetic
getcontext().prec = 75

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import seaborn as sns
import matplotlib
import random
import sys
matplotlib.use('TkAgg')

training_path = "/home/mkausch/dev/4430/lab7/LR_armyTraining.csv"
testing_path = "/home/mkausch/dev/4430/lab7/LR_armyTesting.csv"
results_path = "/home/mkausch/dev/4430/lab7/LR_armyResults.csv"


def read_training(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x_vals = [float(line.split(',')[0]) for line in lines]
        y_vals = [float(line.split(',')[1]) for line in lines]
    return x_vals, y_vals

# Given Data
x_vals = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
y_vals = [2.1, 4.7, 4.8, 6.6, 8.5, 9.9, 10.1, 10.9, 11.7, 13.1]
# Read training data
# x_vals, y_vals = read_training(training_path)

x_vals = [Decimal(x) for x in x_vals]
y_vals = [Decimal(y) for y in y_vals]

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

    m_min = min(slopes)
    m_max = max(slopes)

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
    
    safe_M = np.vectorize(float)(M)
    safe_B = np.vectorize(float)(B)
    safe_Z = np.vectorize(float)(Z)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(safe_M, safe_B, safe_Z, cmap='jet', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

# Min SSE: m = -1.155, b = 13.440, error = 3.849169
                                      #  : 3.8497296

def grid_search(min_m, max_m, min_b, max_b, num_iterations=0, evaluation="SSE"):
    global x_vals, y_vals

    if num_iterations >= 300:  # Stop at max depth
        # Return the midpoint of the current range
        return (min_m + max_m) / Decimal(2), (min_b + max_b) / Decimal(2), min_m, max_m, min_b, max_b
    
    m_range = max_m - min_m
    b_range = max_b - min_b

    # 5x5 grid
    m_step = m_range / Decimal(10)
    b_step = b_range / Decimal(10)

    # Define grid boundaries (6 boundaries for 5 cells)
    m_bounds = [min_m + Decimal(2) * i * m_step for i in range(6)]
    b_bounds = [min_b + Decimal(2) * i * b_step for i in range(6)]

    # Compute evaluation points as midpoints of each boundary segment
    m_vals = [(m_bounds[i] + m_bounds[i+1]) / Decimal(2) for i in range(5)]
    b_vals = [(b_bounds[i] + b_bounds[i+1]) / Decimal(2) for i in range(5)]

    # print(f"[Iteration {num_iterations}] Grid Search: m_range={m_range}, b_range={b_range}")
    # print(f"Incoming Range: m=({min_m}, {max_m}), b=({min_b}, {max_b})")
    # print(f"m_bounds: {m_bounds}")
    # print(f"b_bounds: {b_bounds}")
    # print(f"m_vals: {m_vals}")
    # print(f"b_vals: {b_vals}")

    best_m = None
    best_b = None
    best_eval = None

    # Evaluate the 5x5 grid cells
    for i in range(5):
        for j in range(5):
            m_candidate = m_vals[i]
            b_candidate = b_vals[j]
            if evaluation == "SSE":
                eval = sse(x_vals, y_vals, m_candidate, b_candidate)
            elif evaluation == "SAE":
                eval = sae(x_vals, y_vals, m_candidate, b_candidate)
            # print(f"m: {m_candidate} b: {b_candidate} eval: {eval}")
            if best_eval is None or eval < best_eval:
                best_eval = eval
                best_m_index = i
                best_b_index = j
                best_m = m_candidate
                best_b = b_candidate

    # print(f"Best m: {best_m}, b: {best_b}, error: {best_eval}")
    # print(f"best_m_index: {best_m_index}, best_b_index: {best_b_index}")

    # Dynamically adjust the new range around the best point
    new_min_m = best_m - Decimal(3) * m_step
    new_max_m = best_m + Decimal(3) * m_step
    new_min_b = best_b - Decimal(3) * b_step
    new_max_b = best_b + Decimal(3) * b_step

    # print(f"New Range: m=({new_min_m}, {new_max_m}), b=({new_min_b}, {new_max_b})\n")
    # print(f"Iteration: {num_iterations}", end="\r")
    
    return grid_search(new_min_m, new_max_m, new_min_b, new_max_b, num_iterations + 1)


def greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=None, start_b=None, evaluation="SSE"):
    # Choose evaluation function
    if evaluation == "SSE":
        evaluation_fn = sse  # sse should be defined elsewhere
    elif evaluation == "SAE":
        evaluation_fn = sae  # sae should be defined elsewhere
    else:
        raise ValueError("Invalid evaluation function")
    
    # Convert bounds to Decimal for precision arithmetic
    min_m, max_m, min_b, max_b = map(Decimal, (min_m, max_m, min_b, max_b))
    max_precision = Decimal("1e-14")

    # Initialize starting point
    if start_m is None:
        best_m = Decimal(str(random.uniform(float(min_m), float(max_m))))
    else:
        best_m = Decimal(str(start_m))
    if start_b is None:
        best_b = Decimal(str(random.uniform(float(min_b), float(max_b))))
    else:
        best_b = Decimal(str(start_b))

    best_error = evaluation_fn(x_vals, y_vals, best_m, best_b)

    # Start with a base precision (e.g., 1.0)
    dp = Decimal("1.0")
    total_iterations = 0

    # Outer loop: continue until the smallest step size is reached
    while dp > max_precision:
        # Loop over multipliers from 9 down to 1 for current precision
        # for counter in range(9, 0, -1):
        improved = True
        print(f"\nTrying dp: {dp}")
        # Keep searching at this dp until no further improvements are found
        curr_iterations = 0
        while improved:
            improved = False
            total_iterations += 1
            curr_iterations += 1
            # Define the 3x3 grid of neighbors using the current dp
            neighbors = [
                (best_m - dp, best_b + dp), (best_m, best_b + dp), (best_m + dp, best_b + dp),
                (best_m - dp, best_b),     (best_m, best_b),     (best_m + dp, best_b),
                (best_m - dp, best_b - dp), (best_m, best_b - dp), (best_m + dp, best_b - dp)
            ]
            
            for m, b in neighbors:
                error = evaluation_fn(x_vals, y_vals, m, b)
                if error < best_error:
                    best_error = error
                    best_m, best_b = m, b
                    improved = True
                    print(f"\rtotal iterations: {total_iterations} | current iterations: {curr_iterations} | error: {best_error:.6f} | m: {best_m:.6f}, b: {best_b:.6f}", end="")  
                    break  # restart search from the new best point
        # After testing all multipliers at this precision, reduce precision for a finer search
        dp /= Decimal(2)
    
    data = [best_m, best_b, best_error]
    return data, total_iterations


def plot_best_fit(m, b, x_vals, y_vals):
    x_vals = [Decimal(x) for x in x_vals]
    y_vals = [Decimal(y) for y in y_vals]
    
    # Create a new list of x values with the best fit line

    plt.plot([min(x_vals),max(x_vals)], [y(min(x_vals), m, b), y(max(x_vals), m, b)], color='black', linestyle='--', label="Best Fit")
    # Plot the best fit line
    plt.scatter(x_vals, y_vals, marker='.', color='red')
    # plt.plot(x_vals, y_fit, label="Best Fit Line")
    plt.legend()
    plt.show()

def iterative_search(min_m, max_m, min_b, max_b, evaluation="SSE"):
    # Iterative search algorithm
    # Compute min/max slope with outlier filtering

    if evaluation == "SSE":
        eval = sse
    elif evaluation == "SAE":
        eval = sae
    else:
        raise ValueError("Invalid evaluation function")

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
    error_mat = np.zeros(M.shape)  # Store error values

    min_error = (None, None, float('inf'))

    progress = tqdm(total=M.shape[0] * M.shape[1], desc="Computing Errors", dynamic_ncols=True)
    iterations = 0

    # Compute Error for each (m, b) pair
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):  # FIX: Use M.shape[1] instead of B.shape[1]
            iterations += 1
            m = M[i, j]
            b = B[i, j]
            error = eval(x_vals, y_vals, m, b)

            error_mat[i, j] = error

            if error < min_error[2]:
                min_error = (m, b, error)
            progress.update(1)

        progress.set_description(
            f"{evaluation}({min_error[0]:.3f}, {min_error[1]:.3f})={min_error[2]:.6f}"
        )

    progress.close()

    print("\nFinal Results:")
    print(f"Min {evaluation}: m = {min_error[0]}, b = {min_error[1]}, error = {min_error[2]}")

    # Generate heatmaps
    heatmap(error_mat, m_range, b_range, title="SAE Heatmap", colorbar_title="Sum of Absolute Errors", cmap="inferno", global_min=min_error)

    # Plot SAE Surface
    plot_surface(M, B, error_mat, title="Sum of Absolute Errors (SAE) Surface Plot", 
                 xlabel="Slope (a)", ylabel="Intercept (b)", zlabel=f"{evaluation}")

    return min_error, iterations

def main():

    print(f"Training Data: {x_vals}, {y_vals}")

    # data for real data
    # min_m = Decimal(5.5488870)
    # max_m = Decimal(5.5488880)
    # min_b = Decimal(-200.6969915)
    # max_b = Decimal(-200.6969925)
    min_m = Decimal(5.2)
    max_m = Decimal(5.4)
    min_b = Decimal(-187)
    max_b = Decimal(-186)
    # min_m = Decimal(-1000)
    # max_m = Decimal(1000)
    # min_b = Decimal(-1000)
    # max_b = Decimal(1000)


    # Read Validation Data
    x_valid, y_valid = read_validation(testing_path, results_path)
    
    print("\n\n------  Iterative Search ------\n")
    # min_sae = iterative_search(min_m, max_m, min_b, max_b, evaluation="SAE")
    min_sse = iterative_search(min_m, max_m, min_b, max_b, evaluation="SSE")

    # valid_sae = sae(x_valid, y_valid, min_sae[0], min_sae[1])
    valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    print(f"\nValidation Results:")
    # print(f"Min SAE: m = {min_sae[0]}, b = {min_sae[1]}, error = {valid_sae}")
    print(f"Min SSE: m = {min_sse[0]}, b = {min_sse[1]}, error = {valid_sse}")

    x_valid = [Decimal(x) for x in x_valid]
    y_valid = [Decimal(y) for y in y_valid]


    # print(f"\n------  Grid Search ------")
    # print("------  SSE  ------\n")
    # best_m, best_b, range_m_min, range_m_max, range_b_min, range_b_max = grid_search(min_m, max_m, min_b, max_b, num_iterations=0)
    # print(f"\nBest m: {best_m}, best b: {best_b}")
    # print(f"Best training grid sse: {sse(x_vals, y_vals, best_m, best_b)}")
    # print(f"Last iteration range: m=({range_m_min}, {range_m_max}), b=({range_b_min}, {range_b_max})")


    # valid_sse = sse(x_valid, y_valid, best_m, best_b)
    # print(f"\nGrid Validation Results:")
    # print(f"Min SSE: m = {best_m}, b = {best_b}, error = {valid_sse}")

    # plot_best_fit(best_m, best_b, x_vals, y_vals)


    ## Grid Search using SAE
    # min_m = Decimal(3.701)
    # max_m = Decimal(3.706)
    # min_b = Decimal(-70.5590)
    # max_b = Decimal(-70.5600)

    # print(f"\n\n------  Grid Search ------")
    # print("------  SAE  ------\n")

    # best_m, best_b, range_m_min, range_m_max, range_b_min, range_b_max = grid_search(min_m, max_m, min_b, max_b, num_iterations=0, evaluation="SAE")
    # print(f"\nBest m: {best_m}, best b: {best_b}")
    # print(f"Best training grid sse: {sae(x_vals, y_vals, best_m, best_b)}")
    # print(f"Last iteration range: m=({range_m_min}, {range_m_max}), b=({range_b_min}, {range_b_max})")

    # valid_sae = sae(x_valid, y_valid, best_m, best_b)
    # print(f"\nGrid Validation Results:") 
    # print(f"Min SAE: m = {best_m}, b = {best_b}, error = {valid_sae}")


    # plot_best_fit(best_m, best_b, x_vals, y_vals)


    print("\n\n------  Greedy Search ------\n")

    min_m, max_m = find_slope_range(x_vals, y_vals)
    min_b, max_b = find_intercept_range(x_vals, y_vals, min_m, max_m)

    # sm = Decimal(3.7)
    # sb = Decimal(-70)
    sm = None
    sb = None
    # min_sae, it1 = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=sm, start_b = sb, evaluation="SAE")
    # print(f"\nSAE Iterations: {it1}, error: {min_sae[2]}")      

    # sm = Decimal(5.3)
    # sb = Decimal(-186.5)
    min_sse, it2 = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=sm, start_b = sb, evaluation="SSE")
    print(f"\nSSE Iterations: {it2}, error: {min_sse[2]}")

    # valid_sae = sae(x_valid, y_valid, min_sae[0], min_sae[1])
    valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    print(f"\nValidation Results:")
    # print(f"Min SAE: m = {min_sae[0]}, b = {min_sae[1]}, error = {valid_sae}")
    print(f"Min SSE: m = {min_sse[0]}, b = {min_sse[1]}, error = {valid_sse}")


if __name__ == "__main__":

    main()
