import numpy as np
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from math import pi

def terminator(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) * np.cos(x) * np.sin(y)

# Create meshgrid for contour plot
X = np.linspace(-5 * pi, 5 * pi, 400)
Y = np.linspace(-5 * pi, 5 * pi, 400)
X_grid, Y_grid = np.meshgrid(X, Y)
Z = terminator(X_grid, Y_grid)

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

    r1 = np.random.rand(N, D) # random samples from [0, 1] over N,D array
    r2 = np.random.rand(N, D)

    new_velocities = (
        w * velocities
        + c1 * r1 * (pbests - positions)
        + c2 * r2 * (gbest - positions)
    )
    new_positions = positions + new_velocities

    return new_positions, new_velocities

def run_pso(objective_function, N, D, w=.72984, c1=2.05, c2=2.05, max_iter=100, epsilon=1e-6):

    # terminate based on stagnation over changing epsilon
    patience = 10
    none_improved_counter = 0

    # c1_start = 2.5
    # c1_end   = 0.1
    # c2_start = 0.1
    # c2_end   = 2.5
    # c1 = 2.0
    # c2 = 2.0


    # Initialize starting values
    # start pos in the range (symmetrical around it)
    pos_tracker = []
    positions = np.random.uniform(-2*pi, 2*pi, (N, D))
    pos_tracker = [positions.copy()]
    
    # find max x,y and min x,y for the velocity 
    max_pos = np.max(positions, axis=0) # max position in each dimension
    # print(f"Max Position: {max_pos}")
    # print(f"Min Position: {np.min(positions, axis=0)}")
    # print(f"Positions: {positions}")
    min_pos = np.min(positions, axis=0) # min position in each dimension
    v_max = (max_pos - min_pos) / 2 # max velocity in each dimension
    # v_max = 0
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
        # c1 = c1_start - (c1_start - c1_end) * (i / max_iter)
        # c2 = c2_start + (c2_end - c2_start) * (i / max_iter)


        # Update velocities and positions
        positions, velocities = pso_step(positions, velocities, pbests, gbest, w, c1, c2)
        pos_tracker.append(positions.copy())

        # Update personal bests
        new_vals = objective_function(positions[:, 0], positions[:, 1])

        improved = pbest_vals < new_vals # create mask of values that are better than the current personal best
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



        # check for stagnation among particles
        # if any_improved:
        #     none_improved_counter = 0
        #     # print(f"Improved: {num_improved} particles improved their personal bests.")
        # else:
        #     none_improved_counter += 1

        # check if no pbest improved
        if not any_improved:
            print(f"Terminating after {i} iterations due to no pbest change.")
            break
            

        # check for stagnation among global best
        # if abs(previous_gbest_val - gbest_val) < epsilon:
        #     none_improved_counter += 1
        # else:
        #     none_improved_counter = 0

        # if none_improved_counter >= patience:
        #     print(f"Terminating after {i} iterations due to stagnation.")
        #     print(f"Converged: {gbest_val} with epsilon {epsilon}")
        #     break

        # if none_improved_counter >= patience:
        #     # print(f"Terminating after {i} iterations due to stagnation.")
        #     break


        # print(f"Iteration {i}: Best Value = {gbest_val}")
        # Print the current best value

    # print(f"Iteration {i}: Best Value = {gbest_val}")

    tracking_data = {"pos": pos_tracker,
                     "pbest": pbest_tracker,
                     "pbest_val": pbest_val_tracker,
                     "gbest_val": gbest_val_tracker}

    return gbest, gbest_val, tracking_data

if __name__ == "__main__":
    # Parameters
    N = 30  # Number of particles
    D = 2   # Number of dimensions
    max_iter = 1000  # Maximum number of iterations

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

