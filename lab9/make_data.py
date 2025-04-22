import random
from math import pi
import numpy as np


num_vals = 100

fin = open("rand_vals_terminator_100.txt", "w")
positions = np.random.uniform(-2*pi, 2*pi, (num_vals, 2))

for i in range(num_vals):
    fin.write(f"{positions[i, 0]:.5f}, {positions[i, 1]:.5f}\n")

fin.close()