import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from math import pi
import tqdm
epsilon = 1e-7

def objective_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

def cross_entropy_optimization(mean=0.0, std=3.14, n_samples=100, elite_frac=0.1, max_iters=100):
    
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

    for i in range(max_iters):
        iterations += 1
        samples_x = np.random.normal(mean_x, std_x, n_samples)
        samples_y = np.random.normal(mean_y, std_y, n_samples)
        scores = np.array([objective_function(x, y) for x, y in zip(samples_x, samples_y)])
        elite_indices = scores.argsort()[-elite_size:]

        elite_x = samples_x[elite_indices]
        elite_y = samples_y[elite_indices]

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

    return mean_x, mean_y, data

def main():
    # Set number of runs
    n_runs = 100

    # Test for the best starting standard deviation
    stdev_list = [pi/3, 2*pi/3, pi, (4*pi)/3, (5*pi/3), 2*pi]
    stdev_labels = ["π/3", "2π/3", "π", "4π/3", "5π/3", "2π"]
    total_runs = len(stdev_list) * n_runs
    max_val = None
    max_x_list = []
    max_y_list = []

    # Collect results
    results = {label: [] for label in stdev_labels}
    progress = tqdm.tqdm(total=total_runs, desc="Sigma Testing Progress")

    for std, label in zip(stdev_list, stdev_labels):
        for run in range(n_runs):
            _, _, data = cross_entropy_optimization(mean=0.0, std=std)
            results[label].append(data)
            
            # Overwrites instead of printing new line
            progress.set_postfix({
                "σ": label,
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
    print(f"Number of peaks found: {len(max_x_list)}")
    # print(f"Maximum x: {max_x_list}")
    # print(f"Maximum y: {max_y_list}")

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


    ###############################################################################################
    ######################## Test for the best mean location based on stev ########################
    ###############################################################################################

    # === Updated test values ===
    n_runs = 100
    stdev_list = [pi/6, pi/3, pi, 3*pi/2, 2*pi]
    stdev_labels = ["π/6", "π/3", "π", "3π/2", "2π"]
    # stdev_labels = ["π/2", "π", "3π/2", "2π"]
    # stdev_labels = ["π/3", "2π/3", "π"]
    mean_list = [-pi, 0, pi]
    mean_labels = ["-π", "0", "π"]

    mean_tuples = [(-pi,0), (-pi,pi), (pi,0), (pi, pi), (0, -pi)]
    mean_tuple_labels = [("-π", "0"), ("-π", "π"), ("π", "0"), ("π", "π"), ("0", "-π")]

    # Create mean tuples and corresponding labels
    # mean_tuples = [(m1, m2) for m1 in mean_list for m2 in mean_list]
    # mean_tuple_labels = [(mean_labels[i], mean_labels[j]) for i in range(len(mean_labels)) for j in range(len(mean_labels))]

    # Initialize results
    results = {(s_label, m_label): [] for s_label in stdev_labels for m_label in mean_tuple_labels}
    total_runs = len(stdev_labels) * len(mean_tuple_labels) * n_runs
    progress = tqdm.tqdm(total=total_runs, desc="Mean/Sigma Testing Progress")

    max_val = 1.0  # Known theoretical global max
    max_vals_found = {(s_label, m_label): 0 for s_label in stdev_labels for m_label in mean_tuple_labels}


    # Run optimization
    for std, s_label in zip(stdev_list, stdev_labels):
        for (mean, m_label) in zip(mean_tuples, mean_tuple_labels):
            for run in range(n_runs):
                _, _, data = cross_entropy_optimization(mean=mean, std=std)
                results[(s_label, m_label)].append(data)

                progress.set_postfix({
                    "σ": s_label,
                    "μ": f"({m_label[0]}, {m_label[1]})",
                    "Run": run + 1,
                    "Score": f"{data['score']:.4f}"
                })
                progress.update(1)

                if data["score"] == max_val:
                    max_vals_found[(s_label, m_label)] += 1
                    

    # Extract score data
    score_data = [[run['score'] for run in results[(s_label, m_label)]]
                for s_label in stdev_labels
                for m_label in mean_tuple_labels]
    iter_data = [[run['iter'] for run in results[(s_label, m_label)]]
                for s_label in stdev_labels
                for m_label in mean_tuple_labels]
    # Extract values in the same order as other plots
    global_hit_counts = [max_vals_found[(s_label, m_label)]
                     for s_label in stdev_labels
                     for m_label in mean_tuple_labels]
    

    # Create labels
    labels = [f"{s_label}\n({m_label[0]}, {m_label[1]})"
            for s_label in stdev_labels
            for m_label in mean_tuple_labels]

    # Plot
    plt.figure(figsize=(18, 6))
    plt.boxplot(score_data, tick_labels=labels)
    plt.title("Final Objective Scores for All μ and σ Combinations")
    plt.xlabel("σ and μ Combinations")
    plt.ylabel("Best Score Found")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("full_score_boxplot.png")
    plt.show()

    plt.figure(figsize=(18, 6))
    plt.boxplot(iter_data, tick_labels=labels)  # reuse the same labels from the score plot
    plt.title("Number of Iterations to Convergence")
    plt.xlabel("σ and μ Combinations")
    plt.ylabel("Iterations")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("iterations_boxplot.png")
    plt.show()

    # Bar plot of how many times each pair reached the global max
    plt.figure(figsize=(18, 6))
    plt.bar(labels, global_hit_counts)
    plt.title("Number of Times Global Maximum Was Found")
    plt.xlabel("σ and μ Combinations")
    plt.ylabel("Global Max Hits (score ≥ 1.0)")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("global_max_hits.png")
    plt.show()


    # Print summary
    print("\n------------ Best Results ------------")
    print(f"Maximum Score: {max_val}")
    print(f"Number of unique X peaks: {len(max_x_list)}")
    print(f"Number of unique Y peaks: {len(max_y_list)}")

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
