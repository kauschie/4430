import random
from math import pi
import numpy as np


num_vals = 64
DIM = 2

fin = open("rand_vals_terminator.txt", "w")
positions = np.random.uniform(-2*pi, 2*pi, (num_vals, DIM))

for i in range(num_vals):
    fin.write(f"{positions[i, 0]:.5f}, {positions[i, 1]:.5f}\n")

fin.close()