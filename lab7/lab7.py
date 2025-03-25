
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import seaborn as sns
import matplotlib
import random
import sys
import math
matplotlib.use('TkAgg')

training_path = "/home/mkausch/dev/4430/lab7/LR_armyTraining.csv"
testing_path = "/home/mkausch/dev/4430/lab7/LR_armyTesting.csv"
results_path = "/home/mkausch/dev/4430/lab7/LR_armyResults.csv"
random_path = "/home/mkausch/dev/4430/lab7/rand_vals.txt"

min_m = -50
max_m = 50
min_b = -500
max_b = 500


def read_training(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x_vals = [float(line.split(',')[0]) for line in lines]
        y_vals = [float(line.split(',')[1]) for line in lines]
    return x_vals, y_vals

def read_random(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            x_vals.append(float(line[0]))
            y_vals.append(float(line[1]))
    return x_vals, y_vals

# Given Data
# x_vals = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# y_vals = [2.1, 4.7, 4.8, 6.6, 8.5, 9.9, 10.1, 10.9, 11.7, 13.1]
# Read training data
x_vals, y_vals = read_training(training_path)

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


def terminator(x, y):
    radicant = x**2 + y**2
    rad = math.sqrt(radicant)
    return math.sin(rad) * math.cos(x) * math.sin(y)


def find_slope_range(x_vals, y_vals):
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

def get_vals_steps(min_m, max_m, min_b, max_b, num_cells=5):
    m_range = max_m - min_m
    b_range = max_b - min_b

    # 5x5 grid
    m_step = m_range / (num_cells*2)
    b_step = b_range / (num_cells*2)

    # Define grid boundaries (6 boundaries for 5 cells)
    m_bounds = [min_m + 2 * i * m_step for i in range(num_cells+1)]
    b_bounds = [min_b + 2 * i * b_step for i in range(num_cells+1)]

    # Compute centroids as midpoints of each boundary segment
    m_vals = [(m_bounds[i] + m_bounds[i+1]) / 2 for i in range(num_cells)]
    b_vals = [(b_bounds[i] + b_bounds[i+1]) / 2 for i in range(num_cells)]

    return m_vals, b_vals, m_step, b_step

evaluation = "term"
max_iterations = 500
epsilon = 1E-4

def grid_search(min_m, max_m, min_b, max_b, start_m=None, start_b=None, num_cells=5, grid_2_pass=True):
    m_path = [0]
    b_path = [0]
    eval_path = [0]
    global x_vals, y_vals, evaluation
    num_iterations = 0
    
    if evaluation == "SSE":
        eval = sse
    elif evaluation == "SAE":
        eval = sae
    elif evaluation == "term":
        eval = terminator
    else:
        raise ValueError("Invalid evaluation function")

    # m_bounds, b_bounds, m_vals, b_vals, m_range, b_range, m_step, b_step = get_5x5_centroid_bounds(min_m, max_m, min_b, max_b)
    m_vals, b_vals, m_step, b_step = get_vals_steps(min_m, max_m, min_b, max_b, num_cells=num_cells)
    # print(f"{m_vals}, {b_vals}")
    # print(f"m_step: {m_step}, b_step: {b_step}")

    best_m = None
    best_b = None
    if eval != terminator:
        best_eval = sys.maxsize
        prev_best_e = sys.maxsize - 1
    else:
        best_eval = -sys.maxsize - 1
        prev_best_e = best_eval + 1

    # go until theres no improvement
    # while num_iterations < max_iterations:
    while abs(best_eval - prev_best_e) > epsilon:
        # Evaluate the 5x5 grid cells - First Pass
        # print(f"iteration {num_iterations}")
        prev_best_e = best_eval
        for i in range(num_cells):
            for j in range(num_cells):
                m_candidate = m_vals[i]
                b_candidate = b_vals[j]
                if eval != terminator:
                    e = eval(x_vals, y_vals, m_candidate, b_candidate)
                    if  best_eval is None or e < best_eval:
                        best_eval = e
                        best_m = m_candidate
                        best_b = b_candidate
                else:
                    e = eval(m_candidate, b_candidate)
                    if  best_eval is None or e > best_eval:
                        best_eval = e
                        best_m = m_candidate
                        best_b = b_candidate
                # num_iterations += 1
                
        # m_path.append(best_m)
        # b_path.append(best_b)
        # eval_path.append(best_eval)

        if grid_2_pass:
            # new min/max based on the best point
            # for the second pass
            new_min_m = best_m - (num_cells*m_step)
            new_max_m = best_m + (num_cells*m_step)
            new_min_b = best_b - (num_cells*b_step)
            new_max_b = best_b + (num_cells*b_step)

            # center the same size grid around the best point
            m_vals, b_vals, m_step, b_step = get_vals_steps(new_min_m, new_max_m, new_min_b, new_max_b, num_cells=num_cells)


            # Evaluate the 5x5 grid cells - Second Pass
            for i in range(num_cells):
                for j in range(num_cells):
                    m_candidate = m_vals[i]
                    b_candidate = b_vals[j]
                    if eval != terminator:
                        e = eval(x_vals, y_vals, m_candidate, b_candidate)
                        if  best_eval is None or e < best_eval:
                            best_eval = e
                            best_m = m_candidate
                            best_b = b_candidate
                    else:
                        e = eval(m_candidate, b_candidate)
                        if  best_eval is None or e > best_eval:
                            best_eval = e
                            best_m = m_candidate
                            best_b = b_candidate

        m_path.append(best_m)
        b_path.append(best_b)
        eval_path.append(best_eval)

        # Dynamically adjust the new range 
        # in a 3x3 around the final best point

        # +---+---+---+---+---+
        # |   | * | * | * | * | *
        # +---+---+---+---+---+
        # |   | * | x | x | x | * 
        # +---+---+---+---+---+
        # |   | * | x | o | x | *
        # +---+---+---+---+---+
        # |   | * | x | x | x | *
        # +---+---+---+---+---+
        # |   | * | * | * | * | * 
        # +---+---+---+---+---+

        # * is the 5x5 grid second pass
        #   -- could overlap outside of bounds in order to center it
        # o is the best spot
        # x is the 3x3 grid


        new_min_m = best_m - 3 * m_step
        new_max_m = best_m + 3 * m_step
        new_min_b = best_b - 3 * b_step
        new_max_b = best_b + 3 * b_step
        m_vals, b_vals, m_step, b_step = get_vals_steps(new_min_m, new_max_m, new_min_b, new_max_b, num_cells=num_cells)
        num_iterations += 1
    
    return best_m, best_b, min_m, max_m, min_b, max_b, m_path, b_path, eval_path, num_iterations

    # return (min_m + max_m) / 2, (min_b + max_b) / 2, min_m, max_m, min_b, max_b


# def greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=None, start_b=None):
#     # Choose evaluation function
#     global evaluation

#     if evaluation == "SSE":
#         evaluation_fn = sse  # sse should be defined elsewhere
#     elif evaluation == "SAE":
#         evaluation_fn = sae  # sae should be defined elsewhere
#     else:
#         raise ValueError("Invalid evaluation function")
    
#     # Convert bounds to Decimal for precision arithmetic
#     max_precision = .001

#     # Initialize starting point
#     if start_m is None:
#         best_m = random.uniform(float(min_m), float(max_m))
#     else:
#         best_m = start_m
#     if start_b is None:
#         best_b = random.uniform(float(min_b), float(max_b))
#     else:
#         best_b = start_b

#     best_error = evaluation_fn(x_vals, y_vals, best_m, best_b)

#     # Start with a base precision (e.g., 1.0)
#     dp = 1.0
#     total_iterations = 0
#     prev_error = sys.maxsize

#     # Outer loop: continue until the smallest step size is reached
#     # while dp > max_precision:
#     while abs(prev_error - best_error) > epsilon:
#         prev_error = best_error
#         # Loop over multipliers from 9 down to 1 for current precision
#         # for counter in range(9, 0, -1):
#         improved = True
#         print(f"\nTrying dp: {dp}")
#         # Keep searching at this dp until no further improvements are found
#         curr_iterations = 0
#         prev_prec_error = sys.maxsize

#         while improved:
#         # while abs(prev_prec_error - best_error) > epsilon:
#             prev_prec_error = best_error
#             improved = False
#             total_iterations += 1
#             curr_iterations += 1
#             # Define the 3x3 grid of neighbors using the current dp
#             neighbors = [
#                 (best_m - dp, best_b + dp), (best_m, best_b + dp), (best_m + dp, best_b + dp),
#                 (best_m - dp, best_b),     (best_m, best_b),     (best_m + dp, best_b),
#                 (best_m - dp, best_b - dp), (best_m, best_b - dp), (best_m + dp, best_b - dp)
#             ]
            
#             for m, b in neighbors:
#                 error = evaluation_fn(x_vals, y_vals, m, b)
#                 if error < best_error:
#                     best_error = error
#                     best_m, best_b = m, b
#                     improved = True
#                     print(f"\rtotal iterations: {total_iterations} | current iterations: {curr_iterations} | error: {best_error:.6f} | m: {best_m:.6f}, b: {best_b:.6f}", end="")  
#                     break  # restart search from the new best point
#         # After testing all multipliers at this precision, reduce precision for a finer search
#         dp /= 10
    
#     data = [best_m, best_b, best_error]
#     return data, total_iterations


import random

def greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=None, start_b=None, epsilon = 1E-4, max_precision = 1E-3):
    # Choose evaluation function from global 'evaluation'
    global evaluation

    if evaluation == "SSE":
        evaluation_fn = sse  # sse should be defined elsewhere
    elif evaluation == "SAE":
        evaluation_fn = sae  # sae should be defined elsewhere
    elif evaluation == "term":
        evaluation_fn = terminator
    else:
        raise ValueError("Invalid evaluation function")
    
    # Maximum precision for dp
    # max_precision = 0.001

    # Initialize starting point
    if start_m is None:
        best_m = random.uniform(float(min_m), float(max_m))
    else:
        best_m = start_m
    if start_b is None:
        best_b = random.uniform(float(min_b), float(max_b))
    else:
        best_b = start_b

    best_error = evaluation_fn(x_vals, y_vals, best_m, best_b)
    
    # Initial step size (dp)
    dp = max_precision * 1E4
    total_iterations = 0

    # Parameters for the sliding window termination criterion
    window_size = 10   # Number of iterations to accumulate improvements
    # Outer loop: refine dp until it's too small

    m_path, b_path, eval_path = [best_m], [best_b], [best_error]
    dp_m_path, dp_b_path, db_eval_path = [best_m], [best_b], [best_error]
    while dp > max_precision:
        print(f"\nTrying dp: {dp}")
        curr_iterations = 0
        improvement_history = []  # Reset improvement history for this precision level
        improved = True
        dp_m_path.append(best_m)
        dp_b_path.append(best_b)
        db_eval_path.append(best_error)
        
        # Inner loop: keep searching at the current dp until improvements are insignificant
        while improved:
            improved = False
            total_iterations += 1
            curr_iterations += 1
            
            # Define the 3x3 grid of neighbors around the current best point
            neighbors = [
                (best_m - dp, best_b + dp), (best_m, best_b + dp), (best_m + dp, best_b + dp),
                (best_m - dp, best_b),      (best_m, best_b),      (best_m + dp, best_b),
                (best_m - dp, best_b - dp), (best_m, best_b - dp), (best_m + dp, best_b - dp)
            ]
            
            best_local_improvement = 0.0
            best_candidate = None
            candidate_error = None
            
            # Evaluate all neighbors to find the best improvement
            for m_candidate, b_candidate in neighbors:
                error = evaluation_fn(x_vals, y_vals, m_candidate, b_candidate)
                improvement = best_error - error  # Positive improvement means error is reduced
                if improvement > best_local_improvement:
                    best_local_improvement = improvement
                    best_candidate = (m_candidate, b_candidate)
                    candidate_error = error
            
            # If an improvement was found, update best parameters and record the improvement
            if best_candidate is not None and best_local_improvement > 0:
                best_m, best_b = best_candidate
                best_error = candidate_error
                improvement_history.append(best_local_improvement)
                improved = True
                print(f"\rtotal iterations: {total_iterations} | current iterations: {curr_iterations} | error: {best_error:.6f} | m: {best_m:.6f}, b: {best_b:.6f}", end="")
            else:
                # Even if no improvement is found, record a zero improvement to maintain window size
                improvement_history.append(0)
            
            # Ensure the history only contains the last 'window_size' improvements
            if len(improvement_history) > window_size:
                improvement_history.pop(0)
            
            b_path.append(best_b)
            m_path.append(best_m)
            eval_path.append(best_error)

            # If we've collected enough iterations, check if the accumulated improvement is below the threshold
            if len(improvement_history) == window_size and sum(improvement_history) < epsilon:
                print("\nTotal improvement over the last iterations is negligible, moving to finer search.")
                break
        
        # Reduce the step size for a finer search
        dp /= 2
    
    data = [best_m, best_b, best_error]
    dp_path_data = [dp_m_path, dp_b_path, db_eval_path]
    path_data = [m_path, b_path, eval_path]
    return data, dp_path_data, path_data, total_iterations



def plot_path(m_path, b_path, e_path, title="Path", xlabel="m", ylabel="b", extra_data = None, eval=None):
    plt.figure(figsize=(8, 6))
    
    # Plot the path as a line
    plt.plot(m_path, b_path, marker='', alpha=0.8, linewidth=2, label="Path")
    plt.scatter(m_path, b_path, marker='.', color='black',s=20, label="a/b path")
    
    # Plot each point with a color corresponding to its SSE value
    sc = None
    # sc = plt.scatter(m_path, b_path, c=e_path, cmap='jet', marker='.', s=20, label="a/b path")
    

    if extra_data is not None:
        dp_m = extra_data[0]
        dp_b = extra_data[1]
        dp_e = extra_data[2]
        sc = plt.scatter(dp_m, dp_b, c=dp_e, cmap='jet', marker='o', s=50, label="dp begin") 

    # Highlight the final best point with a distinct marker
    plt.scatter(m_path[-1], b_path[-1], marker='*', color='red', s=150, label="Best")
    

    min_m = min(m_path)
    max_m = max(m_path)
    min_b = min(b_path)
    max_b = max(b_path)
    m_range = max_m - min_m
    b_range = max_b - min_b
    plt.xlim(min_m-(.1*m_range), max_m+(.1*m_range))
    plt.ylim(min_b-(.1*b_range), max_b+(.1*b_range))
    # elif eval == "terminator":
    #     min_m = -6
    #     max_m = 6
    #     min_b = -6
    #     max_b = 6
    #     plt.xlim(min_m, max_m)
    #     plt.ylim(min_b, max_b)
        

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if sc:
        plt.colorbar(sc, label="SSE")
    plt.show()

def plot_best_fit(m, b, x_vals, y_vals):
    # x_vals = [x for x in x_vals]
    # y_vals = [y for y in y_vals]
    
    # Create a new list of x values with the best fit line

    plt.plot([min(x_vals),max(x_vals)], [y(min(x_vals), m, b), y(max(x_vals), m, b)], color='black', linestyle='--', label="Best Fit")
    # Plot the best fit line
    plt.scatter(x_vals, y_vals, marker='.', color='red')
    # plt.plot(x_vals, y_fit, label="Best Fit Line")
    plt.legend()
    plt.show()

def plot_all_paths(vec_of_mpaths, vec_of_bpaths, vec_of_evals):
    # print(f"Plotting {len(vec_of_mpaths)} paths")
    # print(f"{vec_of_mpaths}")
    plt.figure(figsize=(8, 6))
    min_x = sys.maxsize
    max_x = -sys.maxsize
    min_y = sys.maxsize
    max_y = -sys.maxsize

    for i in range(len(vec_of_mpaths)):
        for v in vec_of_mpaths[i]:
            min_x = min(min_x, v)
            max_x = max(max_x, v)
        for v in vec_of_bpaths[i]:
            min_y = min(min_y, v)
            max_y = max(max_y, v)
    
    plt.xlim(min_x-(.1*(max_x-min_x)), max_x+(.1*(max_x-min_x)))
    plt.ylim(min_y-(.1*(max_y-min_y)), max_y+(.1*(max_y-min_y)))
    # plt.xlim(-1000, 1000)
    # plt.ylim(-1000, 1000)

    for i in range(len(vec_of_mpaths)):
        plt.plot(vec_of_mpaths[i], vec_of_bpaths[i], marker='', alpha=0.2, linewidth=2, label=f"Path {i}")
        plt.scatter(vec_of_mpaths[i][0], vec_of_bpaths[i][0], marker='o', color='red', alpha=.8, s=30, label=f"a/b path {i}")
        plt.scatter(vec_of_mpaths[i][-1], vec_of_bpaths[i][-1], marker='*', color='black', alpha=.8,s=30, label=f"a/b path {i}")
    plt.title("All Paths")
    plt.xlabel("m")
    plt.ylabel("b")

    # plt.legend()
    plt.show()

def boxplot(df, col, y_label, x_label, title='Random Data Box Plot', chart_type="box_and_strip"):
    plt.figure(figsize=(8, 6))

    sns.stripplot(y=col, data=df, color='gray', size=4, jitter=True, alpha=0.4)
    if chart_type == "box_and_strip":
        sns.boxplot(y=col, data=df)

    # Customize Labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
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

    # print(f"Training Data: {x_vals}, {y_vals}")
    global evaluation
    evaluation = "SSE"
    global min_m, max_m, min_b, max_b


    # # Read Validation Data
    x_valid, y_valid = read_validation(testing_path, results_path)
    
    # # print("\n\n------  Iterative Search ------\n")
    # # # min_sae = iterative_search(min_m, max_m, min_b, max_b, evaluation="SAE")
    # # min_sse = iterative_search(min_m, max_m, min_b, max_b, evaluation="SSE")

    # # # valid_sae = sae(x_valid, y_valid, min_sae[0], min_sae[1])
    # # valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    # # print(f"\nValidation Results:")
    # # # print(f"Min SAE: m = {min_sae[0]}, b = {min_sae[1]}, error = {valid_sae}")
    # # print(f"Min SSE: m = {min_sse[0]}, b = {min_sse[1]}, error = {valid_sse}")


    # print(f"\n------  Grid Search ------")
    # print("------  SSE  ------\n")
    # grid_search_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # for cell_nums in grid_search_vals:
    #     best_m, best_b, range_m_min, range_m_max, range_b_min, range_b_max, m_path, b_path, e_path, n_iter = grid_search(min_m, max_m, min_b, max_b, num_cells = cell_nums)
    #     print(f"\nBest m: {best_m}, best b: {best_b}")
    #     print(f"Num Iterations: {n_iter}")
    #     print(f"Best training grid sse: {e_path[-1]}")
    #     print(f"Last iteration range: m=({range_m_min}, {range_m_max}), b=({range_b_min}, {range_b_max})")
    #     plot_path(m_path, b_path, e_path, title=f"Grid Search Path: {cell_nums} Cells, iterations {n_iter}, sse {e_path[-1]}", xlabel=f"m, best m: {best_m}", ylabel=f"b, best b: {best_b}")

 
    #     valid_sse = sse(x_valid, y_valid, best_m, best_b)
    #     print(f"\nGrid Validation Results:")
    #     print(f"Min SSE: m = {best_m}, b = {best_b}, error = {valid_sse}")

    #     plot_best_fit(best_m, best_b, x_vals, y_vals)



    # print("\n\n------  Greedy Search ------\n")

    # # min_m, max_m = find_slope_range(x_vals, y_vals)
    # # min_b, max_b = find_intercept_range(x_vals, y_vals, min_m, max_m)


    # print(f"Min m: {min_m}, max m: {max_m}")
    # print(f"Min b: {min_b}, max b: {max_b}")

    # sm = None
    # sb = None

    # min_sse, dp_path_data, path_data, it2 = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=sm, start_b = sb)
    # print(f"\nSSE Iterations: {it2}, error: {min_sse[2]}")
    # plot_path(path_data[0], path_data[1], path_data[2], title="Greedy SearchPath", xlabel="m", ylabel="b", extra_data=dp_path_data)


    # valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    # print(f"\Greedy Validation Results:")
    # print(f"Min SSE: m = {best_m}, b = {best_b}, error = {valid_sse}")

    # plot_best_fit(best_m, best_b, x_vals, y_vals)


    # valid_sse = sse(x_valid, y_valid, min_sse[0], min_sse[1])
    # print(f"\nValidation Results:")
    # print(f"Min SSE: m = {min_sse[0]}, b = {min_sse[1]}, error = {valid_sse}")


    rand_m, rand_b = read_random(random_path)
    eval_scores = []
    greedy_iterations = []
    greedy_ms = []
    greedy_bs = []
    best_greedy_m = None
    best_greedy_b = None
    best_greedy_eval = None
    best_greedy_index = None
    vec_of_mpaths = []
    vec_of_bpaths = []
    vec_of_evals = []
    worst_greedy_index = None
    worst_greedy_eval = None

    plt.figure(figsize=(8, 6))
    plt.scatter(rand_m, rand_b, marker='o', color='red', s=20, label="Random Dataset")
    plt.title("Random Dataset")
    plt.xlabel("m")
    plt.ylabel("b")
    plt.legend()
    plt.show()


    for i in range(len(rand_m)):
        g, g_dp_path, g_path, iters = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=rand_m[i], start_b=rand_b[i])
        # print(f"g_path: {g_path}")
        
        eval_scores.append(g[2])
        greedy_ms.append(g[0])
        greedy_bs.append(g[1])
        greedy_iterations.append(iters)
        vec_of_mpaths.append(g_path[0])
        vec_of_bpaths.append(g_path[1])
        vec_of_evals.append(g[2])
        if best_greedy_eval is None or g[2] < best_greedy_eval:
            best_greedy_eval = g[2]
            best_greedy_m = g[0]
            best_greedy_b = g[1]
            best_greedy_index = i
        if worst_greedy_eval is None or g[2] > worst_greedy_eval:
            worst_greedy_eval = g[2]
            worst_greedy_index = i

    
    overall_best_greedy_m = best_greedy_m
    overall_best_greedy_b = best_greedy_b
    overall_best_greedy_eval = best_greedy_eval
    plot_all_paths(vec_of_mpaths, vec_of_bpaths, vec_of_evals)

    print(f"Mean Greedy Iterations: {np.mean(greedy_iterations)}")
    print(f"Std Dev of Greedy Iterations: {np.std(greedy_iterations)}")
    
    df2 = {'eval_scores':eval_scores, 'greedy_iterations':greedy_iterations}
    boxplot(df2, 'eval_scores', y_label='eval_scores', x_label='', title="Random eval_scores")
    boxplot(df2, 'greedy_iterations', y_label='greedy_iterations', x_label='', title="Random greedy_iterations")

    df3 = {'greedy_ms':greedy_ms, 'greedy_bs':greedy_bs}
    boxplot(df3, 'greedy_ms', y_label='greedy_ms', x_label='', title="Random greedy_ms")
    boxplot(df3, 'greedy_bs', y_label='greedy_bs', x_label='', title="Random greedy_bs")

    print("Difference between best and worst greedy m:")
    print(f"Delta SSE: {best_greedy_eval - worst_greedy_eval}")
    print(f"Delta Iterations: {greedy_iterations[best_greedy_index] - greedy_iterations[worst_greedy_index]}")
    print(f"Best greedy m: {best_greedy_m}, best greedy b: {best_greedy_b}, best eval: {best_greedy_eval}, index: {best_greedy_index}")
    print(f"Worst greedy m: {greedy_ms[best_greedy_index]}, worst greedy b: {greedy_bs[best_greedy_index]}, worst eval: {worst_greedy_eval}, index: {worst_greedy_index}")
    

    print("\n------  Valididation Results ------")
    valid_sse = sse(x_valid, y_valid, best_greedy_m, best_greedy_b)
    print(f"SSE for Validation Data = {valid_sse}")

    index = best_greedy_index

    greedy_ms = []
    greedy_bs = []
    vec_of_evals = []
    vec_of_iters = []
    vec_of_mpaths = []
    vec_of_bpaths = []
    best_greedy_m = None
    best_greedy_b = None
    best_greedy_eval = None
    best_greedy_index = None


    print("\n------  Testing Affect of Epsilon on Greedy Search ------")
    print("\n------ Using Best M/B from Previous Search ------")
    e_list = [1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3, 1E4, 1E5]
    for e in e_list:
        print(f"epsilon: {e}")
        g, g_dp_path, g_path, iters = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=rand_m[index], start_b=rand_b[index], epsilon=e)
        vec_of_evals.append(g[2])
        greedy_ms.append(g[0])
        greedy_bs.append(g[1])
        vec_of_iters.append(iters)
        vec_of_mpaths.append(g_path[0])
        vec_of_bpaths.append(g_path[1])
        if best_greedy_eval is None or g[2] < best_greedy_eval:
            best_greedy_eval = g[2]
            best_greedy_m = g[0]
            best_greedy_b = g[1]
            best_greedy_index = i
        if worst_greedy_eval is None or g[2] > worst_greedy_eval:
            worst_greedy_eval = g[2]
            worst_greedy_index = i
   
   
    plot_all_paths(vec_of_mpaths, vec_of_bpaths, vec_of_evals)

    # Plot of iterations vs SSE for each epsilon
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel("epsilon")
    ax1.set_ylabel("SSE", color=color1)
    ax1.plot(e_list, vec_of_evals, color=color1, marker='', linewidth=2, label="SSE")
    ax1.scatter(e_list, vec_of_evals, color=color1, marker='o', s=10, label="SSE")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')


    # Second y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Iterations", color=color2)
    ax2.plot(e_list, vec_of_iters, color=color2, marker='', linewidth=2, label="Iterations")
    ax2.scatter(e_list, vec_of_iters, color=color2, marker='x', s=10, label="Iterations")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Epsilon vs SSE and Iterations")
    fig.tight_layout()
    plt.show()
   

    fin = open("epsilon_results.txt", "w")
    fin.write("index, epsilon, final_m, final_b, sse, delta best,iterations, \n")
    for i in range(len(e_list)):
        fin.write(f"{i}, {e_list[i]}, {greedy_ms[i]}, {greedy_bs[i]}, {vec_of_evals[i]}, {vec_of_evals[i] - overall_best_greedy_eval}, {vec_of_iters[i]}, \n")
    fin.write(f"Maximum SSE, {str(max(vec_of_evals))}\n")
    fin.write(f"Minimum SSE, {str(min(vec_of_evals))}\n")
    fin.write(f"SSE Range, {str(max(vec_of_evals) - min(vec_of_evals))}\n")
    fin.write(f"Maximum Iterations, {str(max(vec_of_iters))}\n")
    fin.write(f"Minimum Iterations, {str(min(vec_of_iters))}\n")
    fin.write(f"Iterations Range, {str(max(vec_of_iters) - min(vec_of_iters))}\n")    
    fin.write("\n")
    
    fin.close()


    ##########################################################
    ###    Testing Afffect of Precision on Greedy Search   ##
    ##########################################################

    print("\n------  Testing Affect of Precision on Greedy Search ------")
    print("\n------ Using Best M/B from Previous Search ------")
    print("\n ---- Using Best Epsilon from Previous Search ----")


    greedy_ms = []
    greedy_bs = []
    vec_of_evals = []
    vec_of_iters = []
    vec_of_mpaths = []
    vec_of_bpaths = []
    best_greedy_m = None
    best_greedy_b = None
    best_greedy_eval = None
    best_greedy_index = None


    p_list = [1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3, 1E4, 1E5]
    for p in p_list:
        print(f"precision: {p}")
        g, g_dp_path, g_path, iters = greedy_search(x_vals, y_vals, min_m, max_m, min_b, max_b, start_m=rand_m[index], start_b=rand_b[index], epsilon=100, max_precision=p)
        vec_of_evals.append(g[2])
        greedy_ms.append(g[0])
        greedy_bs.append(g[1])
        vec_of_iters.append(iters)
        vec_of_mpaths.append(g_path[0])
        vec_of_bpaths.append(g_path[1])
        if best_greedy_eval is None or g[2] < best_greedy_eval:
            best_greedy_eval = g[2]
            best_greedy_m = g[0]
            best_greedy_b = g[1]
            best_greedy_index = i
        if worst_greedy_eval is None or g[2] > worst_greedy_eval:
            worst_greedy_eval = g[2]
            worst_greedy_index = i
   
   
    plot_all_paths(vec_of_mpaths, vec_of_bpaths, vec_of_evals)

    # Plot of iterations vs SSE for each epsilon
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel("precision")
    ax1.set_ylabel("SSE", color=color1)
    ax1.plot(p_list, vec_of_evals, color=color1, marker='', linewidth=2, label="SSE")
    ax1.scatter(p_list, vec_of_evals, color=color1, marker='o', s=10, label="SSE")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')


    # Second y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Iterations", color=color2)
    ax2.plot(p_list, vec_of_iters, color=color2, marker='', linewidth=2, label="Iterations")
    ax2.scatter(p_list, vec_of_iters, color=color2, marker='x', s=10, label="Iterations")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Precision vs SSE and Iterations")
    fig.tight_layout()
    plt.show()
   
    overall_best_greedy_m
    overall_best_greedy_b
    overall_best_greedy_eval

    fin = open("precision_results.txt", "w")
    fin.write("index, precision, final_m, final_b, sse, delta best,iterations, \n")
    for i in range(len(p_list)):
        fin.write(f"{i}, {p_list[i]}, {greedy_ms[i]}, {greedy_bs[i]}, {vec_of_evals[i]}, {vec_of_evals[i] - overall_best_greedy_eval}, {vec_of_iters[i]}, \n")
    fin.write(f"Maximum SSE, {str(max(vec_of_evals))}\n")
    fin.write(f"Minimum SSE, {str(min(vec_of_evals))}\n")
    fin.write(f"SSE Range, {str(max(vec_of_evals) - min(vec_of_evals))}\n")
    fin.write(f"Maximum Iterations, {str(max(vec_of_iters))}\n")
    fin.write(f"Minimum Iterations, {str(min(vec_of_iters))}\n")
    fin.write(f"Iterations Range, {str(max(vec_of_iters) - min(vec_of_iters))}\n")    
    fin.write("\n")
    
    fin.close()






    # evaluation = "term"
    # best_term_x = None
    # best_term_y = None
    # best_term_eval = None


    # print("\n\n------  Terminator Search ------")
    # for cell_nums in grid_search_vals:
    #     print(f"cells: {cell_nums}")
    #     best_m, best_b, range_m_min, range_m_max, range_b_min, range_b_max, m_path, b_path, e_path, n_iter = grid_search(min_m, max_m, min_b, max_b, num_cells = cell_nums)
    #     print(f"\nBest x: {best_m}, best y: {best_b}")
    #     print(f"Num Iterations: {n_iter}")
    #     print(f"Best training grid sse: {e_path[-1]}")
    #     print(f"Last iteration range: m=({range_m_min}, {range_m_max}), b=({range_b_min}, {range_b_max})")
    #     plot_path(m_path, b_path, e_path, title=f"Grid Search Path: {cell_nums} Cells, iterations {n_iter}, max {e_path[-1]}", xlabel=f"x, best x: {best_m}", ylabel=f"y, best y: {best_b}", eval="terminator")
    #     if best_term_eval is None or e_path[-1] > best_term_eval:
    #         best_term_eval = e_path[-1]
    #         best_term_x = best_m
    #         best_term_y = best_b

    # print(f"Best terminator x: {best_term_x}, best terminator y: {best_term_y}, best eval: {best_term_eval}")



if __name__ == "__main__":

    main()
