
from math import pi
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import imageio
import os


def terminator(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

# Create meshgrid for contour plot
X = np.linspace(-5 * pi, 5 * pi, 400)
Y = np.linspace(-5 * pi, 5 * pi, 400)
X_grid, Y_grid = np.meshgrid(X, Y)
Z = terminator(X_grid, Y_grid)

def read_random(filename):
    x_vals = []
    y_vals = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            x_vals.append(float(line[0]))
            y_vals.append(float(line[1]))
    return np.array(list(zip(x_vals, y_vals)))

def generate_iteration_plot(tracking_data, iteration, filename):
    # Unpack tracking data
    pos = tracking_data["pos"][iteration]
    pbest = tracking_data["pbest"][iteration]
    gbest = pbest[np.argmax(tracking_data["pbest_val"][iteration])]

    # Plot contour
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(label='Function Value')

    # Scatter current pbest locations
    plt.scatter(pos[:, 0], pos[:, 1], c='gray', alpha=0.6, s=20, edgecolors='black', label='Particles')
    plt.scatter(pbest[:, 0], pbest[:, 1], c='white', s=40, edgecolors='black', label='PBest')

    # Highlight global best
    plt.scatter(gbest[0], gbest[1], c='red', s=100, marker='*', label='GBest')

    # Static axis bounds
    plt.xlim(-5 * pi, 5 * pi)
    plt.ylim(-5 * pi, 5 * pi)

    plt.title(f"Iteration {iteration}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')

    # Save plot to file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Example usage
# Assuming the user will call this inside a loop with generated tracking_data
# generate_iteration_plot(tracking_data, iteration=0, filename="frame_000.png")


def pso_step(positions: np.ndarray, velocities: np.ndarray, pbests: np.ndarray, 
             gbest: np.ndarray, w: float, c1: float, c2: float):
    """
    Update the velocities and positions of particles in a PSO algorithm.

    Args:
        positions (np.ndarray): Current positions of particles.
        velocities (np.ndarray): Current velocities of particles.
        pbests (np.ndarray): Personal best positions of particles.
        gbest (np.ndarray): Global best position.
        w (float): Inertia weight - how much the previous velocity influences the current velocity.
        c1 (float): Cognitive coefficient - how much the particle is influenced by its own best position.
        c2 (float): Social coefficient - how much the particle is influenced by the global best position.

    Returns:
        np.ndarray: Updated positions of particles.
    """
    N, D = positions.shape
    max_vel = 1.1

    # r1 = np.random.rand(N, D) # random samples from [0, 1] over N,D array
    # r2 = np.random.rand(N, D)

    r1 = np.random.uniform(0.4, 0.9, size=(N, D))
    r2 = np.random.uniform(0.4, 0.9, size=(N, D))


    
    new_velocities = (
        w * velocities
        + c1 * r1 * (pbests - positions)
        + c2 * r2 * (gbest - positions)
    )

    # new_velocities = np.clip(new_velocities, -max_vel, max_vel)  # Limit velocity to [-0.2, 0.2]
    new_positions = positions + new_velocities

    return new_positions, new_velocities

def run_pso(objective_function, N, D, w=0.1, c1=0.2, c2=1.2, max_iter=100, epsilon=1e-6, v_max=None, start_positions=None):

    # terminate based on stagnation over changing epsilon
    patience = 3
    none_improved_counter = 0

    c1_start = .8
    c1_end   = .2
    c2_start = .2
    c2_end   = .8
    # c1 = 1.0
    # c2 = 3.1


    # Initialize starting values
    # start pos in the range (symmetrical around it)
    pos_tracker = []
    # positions = np.random.uniform(-2*pi, 2*pi, (N, D))
    if start_positions is None:
        rand_path = "./rand_vals_terminator.txt"
        positions = read_random(rand_path)
    else:
        positions = start_positions

    pos_tracker = [positions.copy()]
    
    # find max x,y and min x,y for the velocity 
    max_pos = np.max(positions, axis=0) # max position in each dimension
    # print(f"Max Position: {max_pos}")
    # print(f"Min Position: {np.min(positions, axis=0)}")
    # print(f"Positions: {positions}")
    min_pos = np.min(positions, axis=0) # min position in each dimension
    if v_max == None:
        v_max = (max_pos - min_pos) / 2 # max velocity in each dimension
    # v_max = 0
    v_max = 0.1
    velocities = np.random.uniform(-v_max, v_max, (N, D)) # random velocities in the range of max velocity
    
    # Init personal best as current location
    pbests = positions.copy() # just use current position as their best at this point
    pbest_tracker = [pbests.copy()] # track the personal best positions over time
    pbest_vals = objective_function(positions[:, 0], positions[:, 1]) # calculate the objective function for each particle
    pbest_val_tracker = [pbest_vals.copy()] # track the personal best values over time

    # Init global best as the best personal best
    best_index = np.argmax(pbest_vals) # find the index of the best personal best
    gbest = pbests[best_index] # find the best personal best
    gbest_val = pbest_vals[best_index] # find the value of the best personal best
    gbest_val_tracker = [gbest_val] # track the best value over time

    for i in range(max_iter):

        # Update inertia weight and cognitive/social coefficients
        # if i < max_iter / 2:
        #     c1 = c1_start + (c1_end - c1_start) * (i / (max_iter / 2))
        #     c2 = c2_start + (c2_end - c2_start) * (i / (max_iter / 2))
        # print(f"Iteration {i}: c1 = {c1:.2f}, c2 = {c2:.2f}")

        # Update velocities and positions
        positions, velocities = pso_step(positions, velocities, pbests, gbest, w, c1, c2)
        pos_tracker.append(positions.copy())

        # Update personal bests
        new_vals = objective_function(positions[:, 0], positions[:, 1])

        improved = new_vals > pbest_vals # create mask of values that are better than the current personal best
        any_improved = np.any(improved) # check if any values improved
        
        pbests[improved] = positions[improved] # update personal bests based on mask passed in
        pbest_vals[improved] = new_vals[improved] # update the personal best values based on the mask
        pbest_tracker.append(pbests.copy()) # track the personal best positions over time
        pbest_val_tracker.append(pbest_vals.copy())

        # Update global best
        best_index = np.argmax(pbest_vals)
        gbest = pbests[best_index]

        previous_gbest_val = gbest_val
        gbest_val = pbest_vals[best_index]
        gbest_val_tracker.append(gbest_val)



        # # check for stagnation among particles
        # if any_improved:
        #     none_improved_counter = 0
        #     # print(f"Improved: {num_improved} particles improved their personal bests.")
        # else:
        #     none_improved_counter += 1

        # check if no pbest improved
        # if not any_improved:
        #     print(f"Terminating after {i} iterations due to no pbest change.")
        #     break
            

        # # check for stagnation among global best
        if abs(previous_gbest_val - gbest_val) < epsilon:
            none_improved_counter += 1
        else:
            none_improved_counter = 0

        if none_improved_counter >= patience:
            # print(f"Terminating after {i} iterations due to stagnation.")
            # print(f"Converged: {gbest_val} with epsilon {epsilon}")
            break

        # if none_improved_counter >= patience:
        #     print(f"Terminating after {i} iterations due to stagnation.")
        #     break


        # print(f"Iteration {i}: Best Value = {gbest_val}")
        # Print the current best value

    # print(f"Iteration {i}: Best Value = {gbest_val}")

    tracking_data = {"pos": pos_tracker,
                     "pbest": pbest_tracker,
                     "pbest_val": pbest_val_tracker,
                     "gbest_val": gbest_val_tracker}

    return gbest, gbest_val, tracking_data


def make_gif(img_dir, output_path):
    with imageio.get_writer(output_path, mode='I', duration=0.02, loop=0) as writer:
        for filename in sorted(os.listdir(img_dir)):
            if filename.endswith(".png") and filename.startswith("frame_"):
                image = imageio.imread(os.path.join(img_dir, filename))
                writer.append_data(image)
    # delete the images after creating the gif
    for filename in os.listdir(img_dir):
        if filename.endswith(".png") and filename.startswith("frame_"):
            os.remove(os.path.join(img_dir, filename))



def boxplot(df, col, y_label, x_label, title='PSO Box Plot', chart_type="box_and_strip"):
    plt.figure(figsize=(7, 7))
    sns.stripplot(data=df, x='Omega', y=col, color='gray', size=4, jitter=True, alpha=0.4)
    if chart_type == "box_and_strip":
        sns.boxplot(data=df, x='Omega', y=col)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
    

def confidence_interval(data, confidence: float = 0.95):
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

def plot_distance_heatmap(data, title="Heatmap", colorbar_title="Correlation", cmap="jet", vmin=None, vmax=None, titles=None, annot=False):
    # Dynamically adjust figure size
    plt.figure(figsize=(max(8, 1.5 * data.shape[1]), max(8, 1.5 * data.shape[0])))

    # Set up the heatmap using Seaborn
    ax = sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': colorbar_title})
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)

    # Plot Title
    plt.title(title, fontsize=14, pad=20)

    # Optimize layout to prevent clipping
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    # Display the heatmap
    plt.show()


if __name__ == "__main__":
    # Parameters
    N = 64  # Number of particles
    D = 2   # Number of dimensions
    max_iter = 100  # Maximum number of iterations
    trials = 1000
    omegas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    omega_labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    # Add a sanitized version for filename usage
    v_safe_labels = [label.replace("/", "_").replace("*", "x") for label in omega_labels]

    captured_example = False
    results = []

    # Run PSO
    best_position, best_value, tracking_data = run_pso(terminator, N, D, max_iter=max_iter)

    # Print final results
    print(f"Best Position: {best_position}")
    print(f"Best Value: {best_value}")
    if (abs(best_position[0]) > 5*pi or abs(best_position[1]) > 5*pi):
        print("Best position is out of bounds. Skipping image generation.")
    else:
        for i in range(len(tracking_data["pos"])):
            generate_iteration_plot(tracking_data, i, f"./img/frame_{i:03d}.png")
            print(f"\rGenerating frame {i+1}/{len(tracking_data['pos'])}", end='')
        make_gif("./img", f"./img/animation_test_swarm.gif")
    
    
    
    fout = open("pso_results.txt", "w")
    fout.write("Trial, Omega, Best Position, Best Value, num iterations\n")

    for v in range(len(omegas)):
        captured_example = False
        # captured_example = True

        for i in range(trials):
            best_position, best_value, tracking_data = run_pso(terminator, N, D, max_iter=max_iter, w=omegas[v])
            fout.write(f"{i+1}, {omega_labels[v]}, {best_position}, {best_value}, {len(tracking_data['pos'])}\n")
            # print(f"Trial {i+1}, Omega {omega_labels[v]}: Best Position: {best_position}, Best Value: {best_value}")

            results.append({
                "Trial": i+1,
                "Omega": omega_labels[v],
                "Best Value": best_value,
                "Iterations": len(tracking_data['pos']),
                "Best X": best_position[0],
                "Best Y": best_position[1],
            })

            # capture the first good example
            if not captured_example:
                if (abs(best_position[0]) > 5*pi or abs(best_position[1]) > 5*pi):
                    print("Best position is out of bounds. Skipping image generation.")
                else:
                    for j in range(len(tracking_data["pos"])):
                        generate_iteration_plot(tracking_data, j, f"./img/frame_{j:03d}.png")
                        print(f"\rGenerating frame {j+1}/{len(tracking_data['pos'])}", end='')
                    
                    make_gif("./img", f"./img/animation_{v_safe_labels[v]}_trial_{i+1}.gif")
                    captured_example = True
                    print(f"\nGIF created: animation_{v}_trial_{i+1}.gif")
                    # remove the images after creating the gif
                    for filename in os.listdir("./img"):
                        if filename.endswith(".png") and filename.startswith("frame_"):
                            os.remove(os.path.join("./img", filename))
                    
    # Create DataFrame
    df = pd.DataFrame(results)

    # Plot best value vs omega
    boxplot(df, col="Best Value", y_label="Best Value", x_label="Omega (w)", title="Final Best Values by Omega (w)")
    boxplot(df, col="Iterations", y_label="Iterations", x_label="Omega (w)", title="Total Iterations by Omega (w)")

        # Define metrics to analyze
    metrics = ["Best Value", "Iterations"]

    for metric in metrics:
        fout.write(f"\n=== {metric} Analysis ===\n")

        grouped = df.groupby("Omega")[metric]

        # Confidence intervals
        fout.write("Confidence Intervals:\n")
        ci_results = {}
        for name, group in grouped:
            stats_result = confidence_interval(group)
            fout.write(f"{name}: Mean = {stats_result['mean']:.3f}, StdDev = {stats_result['stdev']:.3f}, CI = ({stats_result['ci'][0]:.3f}, {stats_result['ci'][1]:.3f})\n")
            ci_results[name] = stats_result

        # Prepare matrices for t-tests
        omegas = list(grouped.groups.keys())
        t_vals = np.zeros((len(omegas), len(omegas)))
        p_vals = np.zeros((len(omegas), len(omegas)))

        for i in range(len(omegas)):
            for j in range(len(omegas)):
                if i == j:
                    t_vals[i, j] = np.nan
                    p_vals[i, j] = np.nan
                else:
                    t_stat, p_val = ttest_ind(grouped.get_group(omegas[i]),
                                              grouped.get_group(omegas[j]),
                                              equal_var=False)
                    t_vals[i, j] = t_stat
                    p_vals[i, j] = p_val

        # Convert to DataFrames
        t_df = pd.DataFrame(t_vals, index=omegas, columns=omegas)
        p_df = pd.DataFrame(p_vals, index=omegas, columns=omegas)

        # Plot heatmaps
        plot_distance_heatmap(t_df, title=f"T-Test Heatmap (t-values) - {metric}", colorbar_title="t-statistic", cmap="coolwarm", annot=True)
        plot_distance_heatmap(p_df, title=f"T-Test Heatmap (p-values) - {metric}", colorbar_title="p-value", cmap="viridis", annot=True)

    fout.close()
    
    """
    # chose 100 samples to align with CEO testing
    # used stop crieria 3 where pbest didn't change for any particle
    # used linear c1/c2 adjust
    # used pi for v_max
    # w = 0.72984 # inertia weight

    # test to compare to CEO

    
    fout = open("pso_results_100.txt", "w")
    fout.write("Trial, Omega, Best Position, Best Value, num iterations\n")

    results = []
    positions = read_random("rand_vals_terminator_100.data")
    for i in range(1000):
        best_position, best_value, tracking_data = run_pso(terminator, 100, 2, max_iter=max_iter, v_max=pi, start_positions=positions)
        fout.write(f"{i+1}, {best_position}, {best_value}, {len(tracking_data['pos'])}\n")

        results.append({
            "Trial": i+1,
            "Best Value": best_value,
            "Iterations": len(tracking_data['pos']),
            "Best X": best_position[0],
            "Best Y": best_position[1],
        })


        # Create DataFrame
    df = pd.DataFrame(results)

    stats_result = confidence_interval(df["Best Value"])
    fout.write(f"Best Value: {stats_result['mean']:.3f}, StdDev = {stats_result['stdev']:.3f}, CI = ({stats_result['ci'][0]:.3f}, {stats_result['ci'][1]:.3f})\n")
    stats_result = confidence_interval(df["Iterations"])
    fout.write(f"Iterations: {stats_result['mean']:.3f}, StdDev = {stats_result['stdev']:.3f}, CI = ({stats_result['ci'][0]:.3f}, {stats_result['ci'][1]:.3f})\n")
    fout.close()
    print(f"Results saved to pso_results_100.txt")
    """