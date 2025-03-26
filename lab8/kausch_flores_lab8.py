import numpy as np

def objective_function(x, y):
    """Computes the objective function value given x and y."""
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

def cross_entropy_optimization(mean=0.0, std=3.14, n_samples=100, elite_frac=0.1, max_iters=100):
    """Optimizes the objective function using the Cross-Entropy Method in 2D."""
    
    # Initialize mean and variance for x and y
    mean_x, std_x = mean, std
    mean_y, std_y = mean, std
    
    elite_size = int(n_samples * elite_frac)  # Number of elite samples

    for i in range(max_iters):
        # Step 2: Sample candidate solutions
        samples_x = np.random.normal(mean_x, std_x, n_samples)
        samples_y = np.random.normal(mean_y, std_y, n_samples)

        # Step 3: Evaluate solutions
        scores = np.array([objective_function(x, y) for x, y in zip(samples_x, samples_y)])

        # Step 4: Select top elite solutions
        elite_indices = scores.argsort()[-elite_size:]  # Maximization: take top values
        elite_x = samples_x[elite_indices]
        elite_y = samples_y[elite_indices]

        # Step 5: Update distribution
        mean_x, std_x = np.mean(elite_x), np.std(elite_x)
        mean_y, std_y = np.mean(elite_y), np.std(elite_y)

        # Print progress
        print(f"Iteration {i+1}: Mean_x = {mean_x:.4f}, Std_x = {std_x:.4f}, Mean_y = {mean_y:.4f}, Std_y = {std_y:.4f}")

        # Convergence check (stop if the variance is small enough)
        if std_x < 1e-7 and std_y < 1e-7:
            break

    return mean_x, mean_y  # Return the optimal solution found

def main():
    # Run the cross-entropy optimization
    optimal_x, optimal_y = cross_entropy_optimization(mean=0.0, std=3.14)
    print(f"Optimal solution found: x = {optimal_x:.4f}, y = {optimal_y:.4f}")
    print(f"Objective function value: {objective_function(optimal_x, optimal_y):.4f}")

if __name__ == "__main__":
    main()