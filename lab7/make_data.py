import random

fin = open("rand_vals.txt","w")
m_vals = [random.uniform(-1000,1000) for _ in range(100)]
b_vals = [random.uniform(-1000,1000) for _ in range(100)]

for i in range(100):
    fin.write(f"{m_vals[i]:.5f}, {b_vals[i]:.5f}\n")

fin.close()