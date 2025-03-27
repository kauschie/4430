import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from math import pi
import tqdm

def objective_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

def cross_entropy_optimization(mean=0.0, std=3.14, n_samples=100, elite_frac=0.1, max_iters=100):
    mean_x, std_x = mean, std
    mean_y, std_y = mean, std
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

        if std_x < 1e-7 and std_y < 1e-7:
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
    stdev_list = [pi/3, pi/2, 2*pi/3, pi, (4*pi)/3, (5*pi/3), 2*pi]
    stdev_labels = ["π/3", "π/2", "2π/3", "π", "4π/3", "5π/3", "2π"]
    n_runs = 100
    total_runs = len(stdev_list) * 100
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

    # draw all global peaks
    plt.figure(figsize=(10,5))
    plt.scatter(max_x_list, max_y_list, marker='.', color='red', s=20, label="a/b path")
    plt.title("All global peaks")
    plt.xlim(-2*pi - .5, 2*pi + .5)
    plt.ylim(-2*pi - .5, 2*pi + .5)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
