import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from math import pi
import tqdm
epsilon = 1e-2

def read_random(filename):
    x_vals = []
    y_vals = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            x_vals.append(float(line[0]))
            y_vals.append(float(line[1]))
    return x_vals, y_vals

def objective_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

import matplotlib.pyplot as plt

def plot_ce_iteration(track_data, iteration, objective_fn=None, save_path=None):
    """
    Plot a single iteration of the Cross-Entropy optimization process.

    Parameters:
    - track_data: dict from `cross_entropy_optimization(track=True)`
    - iteration: int, index of the iteration to plot
    - objective_fn: optional function to use for drawing a background contour plot
    - save_path: if provided, will save the figure to this path
    """
    samples_x = track_data["samples_x"][iteration]
    samples_y = track_data["samples_y"][iteration]
    elite_x = track_data["elite_x"][iteration]
    elite_y = track_data["elite_y"][iteration]
    mean_x = track_data["mean_x"][iteration]
    mean_y = track_data["mean_y"][iteration]
    std_x = track_data["std_x"][iteration]
    std_y = track_data["std_y"][iteration]

    plt.figure(figsize=(8, 6))

    # Optional: Draw contour of the objective function
    if objective_fn is not None:
        # x_range = np.linspace(min(samples_x) - 1, max(samples_x) + 1, 200)
        # y_range = np.linspace(min(samples_y) - 1, max(samples_y) + 1, 200)
        x_range = np.linspace(-2 * pi, 2 * pi, 200)
        y_range = np.linspace(-2 * pi, 2 * pi, 200)

        X, Y = np.meshgrid(x_range, y_range)
        Z = objective_fn(X, Y)
        plt.contourf(X, Y, Z, levels=50, alpha=0.4, cmap='viridis')

    # Plot all samples
    plt.scatter(samples_x, samples_y, color='black', label="Samples", alpha=0.6)

    # Plot elite samples
    plt.scatter(elite_x, elite_y, color='red', label="Elite Samples", edgecolor='k', zorder=3)

    # Plot mean
    plt.scatter([mean_x], [mean_y], color='blue', label="Mean", zorder=4)

    # Horizontal std_x line (x-direction)
    plt.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y],
            color='blue', linestyle='--', alpha=0.8, label="±σₓ")

    # Vertical std_y line (y-direction)
    plt.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y],
            color='magenta', linestyle='--', alpha=0.8, label="±σᵧ")


    plt.title(f"Cross-Entropy Iteration {iteration}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.xlim(-2 * pi, 2 * pi)
    plt.ylim(-2 * pi, 2 * pi)
    plt.gca().set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def cross_entropy_optimization(mean=0.0, std=3.14, n_samples=100, elite_frac=0.1, max_iters=100, track=False):
    epsilon = 1e-6
    if type(mean) != tuple:
        mean_x, std_x = mean, std
        mean_y, std_y = mean, std
    else:
        mean_x, mean_y = mean
        std_x = std
        std_y = std

    elite_size = int(n_samples * elite_frac)
    data = {}
    iterations = 0
    track_data = {}
    track_data["samples_x"] = []
    track_data["samples_y"] = []
    track_data["mean_x"] = []
    track_data["mean_y"] = []
    track_data["std_x"] = []
    track_data["std_y"] = []
    track_data["elite_x"] = []
    track_data["elite_y"] = []

    for i in range(max_iters):
        iterations += 1
        samples_x = np.random.normal(mean_x, std_x, n_samples)
        samples_y = np.random.normal(mean_y, std_y, n_samples)
        scores = np.array([objective_function(x, y) for x, y in zip(samples_x, samples_y)])
        elite_indices = scores.argsort()[-elite_size:]

        elite_x = samples_x[elite_indices]
        elite_y = samples_y[elite_indices]

        if track:
            track_data["samples_x"].append(samples_x)
            track_data["samples_y"].append(samples_y)
            track_data["mean_x"].append(mean_x)
            track_data["mean_y"].append(mean_y)
            track_data["std_x"].append(std_x)
            track_data["std_y"].append(std_y)
            track_data["elite_x"].append(elite_x)
            track_data["elite_y"].append(elite_y)

        mean_x, std_x = np.mean(elite_x), np.std(elite_x)
        mean_y, std_y = np.mean(elite_y), np.std(elite_y)


        if std_x < epsilon and std_y < epsilon:
            break

    # Store best result
    best_idx = elite_indices[-1]
    data["x"] = samples_x[best_idx]
    data["y"] = samples_y[best_idx]
    data["score"] = scores[best_idx]
    data["iter"] = iterations
    data["std"] = std

    if track:
        return mean_x, mean_y, data, track_data

    return mean_x, mean_y, data

def main():

    # Set number of runs
    n_runs = 100


    # Test for the best starting standard deviation
    max_val = 1
    max_list = []
    max_freq = {}

    # Collect results
    # results = {label: [] for label in stdev_labels}
    # progress = tqdm.tqdm(total=total_runs, desc="Sigma Testing Progress")


    for run in range(n_runs):
        # _, _, data = cross_entropy_optimization(mean=(0,-pi), std=std)
        _, _, data = cross_entropy_optimization(mean=(0,0), std=pi, n_samples=100, elite_frac=0.1, max_iters=100)

        if max_val - data["score"] <= epsilon:
            round_x = round(data["x"],3)
            round_y = round(data["y"],3)
            if abs(round_x - 0) < 1e-1:
                round_x = 0
            if abs(round_y - 0) < 1e-1:
                round_y = 0
            max_freq[(round_x, round_y)] = max_freq.get((round_x, round_y), 0) + 1

            if (round_x, round_y) not in max_list:
                max_list.append((round_x, round_y))



        # Step 1: Get sorted (label, count) pairs
    sorted_items = sorted(max_freq.items(), key=lambda item: item[1], reverse=True)

    # Step 2: Split them into x and y for plotting
    labels = [f"({x:.3f}, {y:.3f})" for (x, y), _ in sorted_items]
    counts = [count for _, count in sorted_items]


    # plot frequency of max peaks
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.title("Number of Times Global Maximum Was Found")
    plt.xlabel("Peak Location (x, y)")
    plt.ylabel("Global Max Hits (score ≥ 1.0)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("global_max_hits.png")
    plt.show()



    exit()
    # track one run
    # _, _, data, track_data = cross_entropy_optimization(mean=0.0, std=pi, n_samples=100, elite_frac=0.1, max_iters=100, track=True)
    # print(f"Num Iterations: {len(track_data['samples_x'])}")

    # for i in range(len(track_data["samples_x"])):
    #     plot_ce_iteration(track_data, i, objective_fn=objective_function)
    #     # plot_ce_iteration(track_data, i, objective_fn=objective_function, save_path=f"ce_iter_{i}.png")





    # exit()



    _, _, data = cross_entropy_optimization(mean=0.0, std=pi, n_samples=100, elite_frac=0.1, max_iters=100)
    print(f"Starting Score: {data['score']}")
    print(f"equal one? {1- data['score'] <= 1e-12}")
    print(f"Starting Iterations: {data['iter']}")
    print(f"x: {data['x']}, y: {data['y']}")
    print(f"x: {data['x']:.5f}, y: {data['y']:.5f}")
    # exit()


    # Test for the best starting standard deviation
    stdev_list = [pi/3, 2*pi/3, pi, (4*pi)/3, (5*pi/3), 2*pi]
    stdev_labels = ["π/3", "2π/3", "π", "4π/3", "5π/3", "2π"]
    total_runs = len(stdev_list) * n_runs
    max_val = 1
    max_list = []
    max_freq = {}

    # Collect results
    results = {label: [] for label in stdev_labels}
    progress = tqdm.tqdm(total=total_runs, desc="Sigma Testing Progress")

    for std, label in zip(stdev_list, stdev_labels):
        for run in range(n_runs):
            # _, _, data = cross_entropy_optimization(mean=(0,-pi), std=std)
            _, _, data = cross_entropy_optimization(mean=(0,0), std=std)
            results[label].append(data)
            
            # Overwrites instead of printing new line
            progress.set_postfix({
                "σ": label,
                "Run": run + 1,
                "Score": f"{data['score']:.4f}",
                "Iter": data["iter"]
            })
            progress.update(1)
            # if max_val == None or data["score"] >= max_val:
            if max_val - data["score"] <= epsilon:
                round_x = round(data["x"],3)
                round_y = round(data["y"],3)
                if abs(round_x - 0) < 1e-2:
                    round_x = 0
                if abs(round_y - 0) < 1e-2:
                    round_y = 0
                max_freq[(round_x, round_y)] = max_freq.get((round_x, round_y), 0) + 1

                if (round_x, round_y) not in max_list:
                    max_list.append((round_x, round_y))


    # Extract scores and iterations for plotting
    score_data = [ [run['score'] for run in results[label]] for label in stdev_labels ]
    iter_data = [ [run['iter'] for run in results[label]] for label in stdev_labels ]

    # Plot boxplot for scores
    plt.figure(figsize=(10, 5))
    plt.boxplot(score_data, tick_labels=stdev_labels)
    plt.title("Final Objective Scores vs. Initial σ")
    plt.xlabel("Initial σ Value")
    plt.ylabel("Best Score Found")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("score_boxplot.png")
    plt.show()

    # Plot boxplot for iterations to converge
    plt.figure(figsize=(10, 5))
    plt.boxplot(iter_data, tick_labels=stdev_labels)
    plt.title("Number of Iterations vs. Initial σ")
    plt.xlabel("Initial σ Value")
    plt.ylabel("Iterations to Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("iterations_boxplot.png")
    plt.show()

    print("\n------------ Best Results ------------")
    print(f"Maximum Score: {max_val}")
    print(f"Number of peaks found: {len(max_list)}")

    for i in range(len(max_list)):
        print(f"{max_list[i][0]:.5f}, {max_list[i][1]:.5f}")


    exit()

    # # draw all global peaks
    # plt.figure(figsize=(10,5))
    # plt.scatter(max_x_list, max_y_list, marker='.', color='red', s=20, label="a/b path")
    # plt.title("All global peaks")
    # plt.xlim(-2*pi - .5, 2*pi + .5)
    # plt.ylim(-2*pi - .5, 2*pi + .5)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # # plt.legend()
    # plt.show()


    # ###############################################################################################
    # ######################## Test for the best mean location based on stev ########################
    # ###############################################################################################

    # # === Updated test values ===
    # n_runs = 100
    # stdev_list = [pi/6, pi/3, pi, 3*pi/2, 2*pi]
    # stdev_labels = ["π/6", "π/3", "π", "3π/2", "2π"]
    # # stdev_labels = ["π/2", "π", "3π/2", "2π"]
    # # stdev_labels = ["π/3", "2π/3", "π"]
    # mean_list = [-pi, 0, pi]
    # mean_labels = ["-π", "0", "π"]

    # mean_tuples = [(-pi,0), (-pi,pi), (pi,0), (pi, pi), (0, -pi)]
    # mean_tuple_labels = [("-π", "0"), ("-π", "π"), ("π", "0"), ("π", "π"), ("0", "-π")]

    # # Create mean tuples and corresponding labels
    # # mean_tuples = [(m1, m2) for m1 in mean_list for m2 in mean_list]
    # # mean_tuple_labels = [(mean_labels[i], mean_labels[j]) for i in range(len(mean_labels)) for j in range(len(mean_labels))]

    # # Initialize results
    # results = {(s_label, m_label): [] for s_label in stdev_labels for m_label in mean_tuple_labels}
    # total_runs = len(stdev_labels) * len(mean_tuple_labels) * n_runs
    # progress = tqdm.tqdm(total=total_runs, desc="Mean/Sigma Testing Progress")

    # max_val = 1.0  # Known theoretical global max
    # max_vals_found = {(s_label, m_label): 0 for s_label in stdev_labels for m_label in mean_tuple_labels}


    # # Run optimization
    # for std, s_label in zip(stdev_list, stdev_labels):
    #     for (mean, m_label) in zip(mean_tuples, mean_tuple_labels):
    #         for run in range(n_runs):
    #             _, _, data = cross_entropy_optimization(mean=mean, std=std)
    #             results[(s_label, m_label)].append(data)

    #             progress.set_postfix({
    #                 "σ": s_label,
    #                 "μ": f"({m_label[0]}, {m_label[1]})",
    #                 "Run": run + 1,
    #                 "Score": f"{data['score']:.4f}"
    #             })
    #             progress.update(1)

    #             if data["score"] == max_val:
    #                 max_vals_found[(s_label, m_label)] += 1
                    

    # # Extract score data
    # score_data = [[run['score'] for run in results[(s_label, m_label)]]
    #             for s_label in stdev_labels
    #             for m_label in mean_tuple_labels]
    # iter_data = [[run['iter'] for run in results[(s_label, m_label)]]
    #             for s_label in stdev_labels
    #             for m_label in mean_tuple_labels]
    # # Extract values in the same order as other plots
    # global_hit_counts = [max_vals_found[(s_label, m_label)]
    #                  for s_label in stdev_labels
    #                  for m_label in mean_tuple_labels]
    

    # # Create labels
    # labels = [f"{s_label}\n({m_label[0]}, {m_label[1]})"
    #         for s_label in stdev_labels
    #         for m_label in mean_tuple_labels]

    # # Plot
    # plt.figure(figsize=(18, 6))
    # plt.boxplot(score_data, tick_labels=labels)
    # plt.title("Final Objective Scores for All μ and σ Combinations")
    # plt.xlabel("σ and μ Combinations")
    # plt.ylabel("Best Score Found")
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("full_score_boxplot.png")
    # plt.show()

    # plt.figure(figsize=(18, 6))
    # plt.boxplot(iter_data, tick_labels=labels)  # reuse the same labels from the score plot
    # plt.title("Number of Iterations to Convergence")
    # plt.xlabel("σ and μ Combinations")
    # plt.ylabel("Iterations")
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("iterations2_boxplot.png")
    # plt.show()

    # # Bar plot of how many times each pair reached the global max
    # plt.figure(figsize=(18, 6))
    # plt.bar(labels, global_hit_counts)
    # plt.title("Number of Times Global Maximum Was Found")
    # plt.xlabel("σ and μ Combinations")
    # plt.ylabel("Global Max Hits (score ≥ 1.0)")
    # plt.xticks(rotation=45)
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.savefig("global_max_hits.png")
    # plt.show()


    # # Print summary
    # print("\n------------ Best Results ------------")
    # print(f"Maximum Score: {max_val}")
    # print(f"Number of unique X peaks: {len(max_x_list)}")
    # print(f"Number of unique Y peaks: {len(max_y_list)}")


    #####################################################################################
    ######################## Test Randomly Sampled Means ########################
    #####################################################################################
    # Mean = (0, 0), std = π
    std = pi
    e_frac = 0.1

    rand_path = "./rand_vals_terminator.data"
    rm_x, rm_y = read_random(rand_path)
    # print(f"Random Means: {random_means}")
    # show the distribution of the random means
    plt.figure(figsize=(10, 5))
    plt.scatter(rm_x, rm_y, marker='.', color='red', s=20, label="Random Dataset")
    plt.title("Random Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("random_means.png")
    plt.show()



    stdev_list = [pi/3, 2*pi/3, pi, (4*pi)/3, (5*pi/3), 2*pi]
    stdev_labels = ["π/3", "2π/3", "π", "4π/3", "5π/3", "2π"]
    total_runs = len(rm_x) * len(stdev_list)
    max_val = 1.0

    # Collect results
    results = {label: [] for label in stdev_labels}
    max_vals = {label: 0 for label in stdev_labels}
    progress = tqdm.tqdm(total=total_runs, desc="Sigman Rand Means Testing Progress")

    for std, label in zip(stdev_list, stdev_labels):
        for m_x, m_y in zip(rm_x, rm_y):
            m = (m_x, m_y)
            _, _, data = cross_entropy_optimization(mean=m, std=std, n_samples=100, elite_frac=e_frac, max_iters=100)
            # print(f"m: {m}, score: {data['score']}, iter: {data['iter']}")
            results[label].append(data)
            if max_val - data["score"] <= epsilon: 
                max_vals[label] += 1
            progress.set_postfix({
                "σ": label,
                "Run": run + 1,
                "Score": f"{data['score']:.4f}",
                "Iter": data["iter"]
            })
            progress.update(1)
        
    # plot max hits results
    plt.figure(figsize=(10, 5))
    plt.bar(stdev_labels, max_vals.values())
    plt.title("Number of Times Global Maximum Was Found")
    plt.xlabel("σ Combinations")
    plt.ylabel("Global Max Hits (score ≥ 1.0)")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("rand_data_global_max_hits.png")
    plt.show()
    

    exit()

    n_runs = 100
    #####################################################################################
    ######################## Test for the best number of samples ########################
    #####################################################################################
    # Mean = (0, 0), std = π

    n_samples = [10, 50, 100, 200, 500, 1000]
    n_samples_labels = ["10", "50", "100", "200", "500", "1000"]

    max_val = None
    max_x_list = []
    max_y_list = []

    # Collect results
    n_samples_results = {label: [] for label in n_samples_labels}
    progress = tqdm.tqdm(total=len(n_samples) * n_runs, desc="Sample Testing Progress")

    for n_sample, label in zip(n_samples, n_samples_labels):
        for run in range(n_runs):
            _, _, data = cross_entropy_optimization(mean=0.0, std=pi, n_samples=n_sample)
            n_samples_results[label].append(data)

            # Overwrites instead of printing new line
            progress.set_postfix({
                "Samples": label,
                "Run": run + 1,
                "Score": f"{data['score']:.4f}",
                "Iter": data["iter"]
            })
            progress.update(1)
            if max_val == None or data["score"] >= max_val:
                max_val = data["score"]
                if data["x"] not in max_x_list:
                    max_x_list.append(data["x"])
                if data["y"] not in max_y_list:
                    max_y_list.append(data["y"])

    # Extract scores and iterations for plotting
    n_samples_score_data = [ [run['score'] for run in n_samples_results[label]] for label in n_samples_labels ]
    n_samples_iter_data = [ [run['iter'] for run in n_samples_results[label]] for label in n_samples_labels ]

    # Plot boxplot for scores
    plt.figure(figsize=(10, 5))
    plt.boxplot(n_samples_score_data, tick_labels=n_samples_labels)
    plt.title("Final Objective Scores vs. Number of Samples [Parameters: start=(0,0), σ=π]")
    plt.xlabel("Number of Samples")
    plt.ylabel("Best Score Found")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("n_samples_score_boxplot.png")
    plt.show()

    # Plot boxplot for iterations to converge
    plt.figure(figsize=(10, 5))
    plt.boxplot(n_samples_iter_data, tick_labels=n_samples_labels)
    plt.title("Number of Iterations vs. Number of Samples [Parameters: start=(0,0), σ=π]")
    plt.xlabel("Number of Samples")
    plt.ylabel("Iterations to Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("n_samples_iterations_boxplot.png")
    plt.show()

    print("\n------------ Best Results ------------")
    print(f"Maximum Score: {max_val}")
    print(f"Number of peaks found: {len(max_x_list)}")

    # draw all global peaks
    # plt.figure(figsize=(10,5))
    # plt.scatter(max_x_list, max_y_list, marker='.', color='red', s=20, label="a/b path")
    # plt.title("All global peaks")
    # plt.xlim(-2*pi - .5, 2*pi + .5)
    # plt.ylim(-2*pi - .5, 2*pi + .5)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # # plt.legend()
    # plt.show()

    #####################################################################################
    ######################## Test for the best elite fraction ###########################
    #####################################################################################
    # Mean = (0, 0), std = π

    elite_fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    elite_fractions_labels = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]

    max_val = None
    max_x_list = []
    max_y_list = []

    # Collect results
    elite_fractions_results = {label: [] for label in elite_fractions_labels}
    progress = tqdm.tqdm(total=len(elite_fractions) * n_runs, desc="Elite Fraction Testing Progress")

    for elite_frac, label in zip(elite_fractions, elite_fractions_labels):
        for run in range(n_runs):
            _, _, data = cross_entropy_optimization(mean=0.0, std=pi, elite_frac=elite_frac)
            elite_fractions_results[label].append(data)

            # Overwrites instead of printing new line
            progress.set_postfix({
                "Elite Fraction": label,
                "Run": run + 1,
                "Score": f"{data['score']:.4f}",
                "Iter": data["iter"]
            })
            progress.update(1)
            if max_val == None or data["score"] >= max_val:
                max_val = data["score"]
                if data["x"] not in max_x_list:
                    max_x_list.append(data["x"])
                if data["y"] not in max_y_list:
                    max_y_list.append(data["y"])
    
    # Extract scores and iterations for plotting
    elite_fractions_score_data = [ [run['score'] for run in elite_fractions_results[label]] for label in elite_fractions_labels ]
    elite_fractions_iter_data = [ [run['iter'] for run in elite_fractions_results[label]] for label in elite_fractions_labels ]

    # Plot boxplot for scores
    plt.figure(figsize=(10, 5))
    plt.boxplot(elite_fractions_score_data, tick_labels=elite_fractions_labels)
    plt.title("Final Objective Scores vs. Elite Fraction [Parameters -> start=(0,0), σ=π]")
    plt.xlabel("Elite Fraction")
    plt.ylabel("Best Score Found")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elite_fractions_score_boxplot.png")
    plt.show()

    # Plot boxplot for iterations to converge
    plt.figure(figsize=(10, 5))
    plt.boxplot(elite_fractions_iter_data, tick_labels=elite_fractions_labels)
    plt.title("Number of Iterations vs. Elite Fraction [Parameters -> start=(0,0), σ=π]")
    plt.xlabel("Elite Fraction")
    plt.ylabel("Iterations to Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elite_fractions_iterations_boxplot.png")
    plt.show()

    print("\n------------ Best Results ------------")
    print(f"Maximum Score: {max_val}")
    print(f"Number of peaks found: {len(max_x_list)}")


if __name__ == "__main__":
    main()
