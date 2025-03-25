import random
import math

# min_m = -2*math.pi
# max_m = 2*math.pi
# min_b = -2*math.pi
# max_b = 2*math.pi

min_m = -1000
max_m = 1000
min_b = -1000
max_b = 1000

num_vals = 30

fin = open("rand_vals.txt", "w")
m_vals = [random.uniform(min_m,max_m) for _ in range(num_vals)]
b_vals = [random.uniform(min_b,max_b) for _ in range(num_vals)]

for i in range(num_vals):
    fin.write(f"{m_vals[i]:.5f}, {b_vals[i]:.5f}\n")

fin.close()