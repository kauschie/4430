import numpy as np
from scipy import stats
from scipy.stats import norm, mannwhitneyu
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def plot_distance_heatmap(data, title="Heatmap", cmap="Reds", vmin=None, vmax=None, labels=None):
    # Set size
    plt.figure(figsize=(12,7))
    # Set up the heatmap using Seaborn
    ax = sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for readability

    # Add a title
    plt.title(title, pad=20)

    # Display the heatmap
    plt.show()


def read_ds(filepath):
    with open(filepath, 'r', newline = '') as file:
        distances = []
        boxes = []
        i = 0
        for row in file:
            if i == 0:
                distances = row.split()
                i += 1
            elif i == 1:
                boxes = row.strip().split()
        # print(distances)
        # print(box)
        return [float(d) for d in distances], [int(b) for b in boxes]

def z_test(sample, pop_mean, pop_sigma):
    sample = np.array(sample)
    sample_mean = np.mean(sample)
    n = len(sample)

    z_stat = (sample_mean - pop_mean) / (pop_sigma / np.sqrt(n))
    p_value = 2 * (1-norm.cdf(abs(z_stat)))

    return z_stat, p_value


def plot_multiple_samples_vs_hypothesized_mean(samples, box_names, hypoth_mean):
    """
    Plot multiple samples with a line for the hypothesized mean.
    
    Parameters:
    -----------
    hypoth_mean : float
        The hypothesized mean (will be plotted as a horizontal line).
    """
    plt.figure(figsize=(7, 5))

    for i in range(len(samples)):
        data = np.array(samples[i])
        group_name = box_names[i]
        
        # Slight jitter on the x-axis so points don't overlap vertically
        x_vals = np.random.normal(loc=i + 1, scale=0.06, size=len(data))
        
        # Plot the raw data
        plt.scatter(x_vals, data, alpha=0.7, label=f"{group_name} data")
        
        # Plot the sample mean
        sample_mean = np.mean(data)
        plt.scatter(i + 1, sample_mean, color='red', s=100, marker='D',
                    edgecolor='black', zorder=3,
                    label=f"{group_name} mean = {sample_mean:.2f}")

    # Plot the hypothesized mean as a reference line
    plt.axhline(hypoth_mean, color='green', linestyle='--',
                label=f"Hypothesized mean = {hypoth_mean}")

    # Cosmetic adjustments for x-axis
    # plt.xlim(0.5, len(samples) + 0.5)
    # plt.xticks(range(1, len(samples) + 1), [s["name"] for s in samples])
    plt.ylabel("Value")
    plt.title("Box Data vs. Merlot Mean")

    # Place the legend outside the main plot area
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_z_test(sample, mu_0):

    sample_mean = np.mean(sample)
    n = len(sample)

    # Plot the raw data as a dot/strip
    plt.scatter(np.ones(n), sample, alpha=0.7, color='blue', label='Sample data')

    # Plot the sample mean
    plt.scatter(1, sample_mean, color='red', s=100, marker='D', zorder=3, 
                label=f'Sample mean = {sample_mean:.2f}')

    # Add a horizontal line for the hypothesized mean
    plt.axhline(mu_0, color='green', linestyle='--', label=f'Hypothesized mean = {mu_0}')

    # A little cosmetic offset so dots aren't hidden
    plt.xlim(0.8, 1.2)  
    plt.legend()
    plt.title("One-Sample Visualization vs. Hypothesized Mean")
    plt.ylabel("Value")
    plt.xticks([])
    plt.show()


path = './StatisticalTests_boxed_grape_data.txt'
distances, boxes = read_ds(path)


groups = []
group = []
old_group_num = None
for i in range(len(distances)):
    if old_group_num != boxes[i]:
        old_group_num = boxes[i]
        print(f"working on box {old_group_num}")
        if len(group) > 0:
            print(f"appending group {boxes[i]-1}")
            groups.append(group.copy())
            group.clear()
    group.append(distances[i])
groups.append(group)

for g in groups:
    print(f"len(g): {len(g)}")

pop_mean = 9.5
pop_std = 3

# t_test
t_stats = []
p_values = []
for box1 in groups:
    t_row = []
    p_row = []
    for box2 in groups:
        t, p = stats.ttest_ind(box1, box2, equal_var=False)
        t_row.append(t)
        p_row.append(p)
    t_stats.append(t_row)
    p_values.append(p_row)



box_names = [f"Box{i+1}" for i in range(len(groups))]
print(box_names)
# print(p_values)

binary_p_values = [[1 if pval < 0.05 else 0 for pval in row] for row in p_values]

# print(binary_p_values)
# binary_p_values.reverse()
# print(binary_p_values)

tt_array = np.array(t_stats)
b_p_array = np.array(binary_p_values)
p_array = np.array(p_values)

plot_distance_heatmap(tt_array, title="Grape T-Test T Val", cmap="coolwarm", vmin=-4, vmax=4, labels=box_names)
plot_distance_heatmap(b_p_array, title="Grape T-Test Binary P Val", cmap="coolwarm", labels=box_names)
plot_distance_heatmap(p_array, title="Grape T-Test P Val", cmap="coolwarm", labels=box_names)


z_vals = []
p_vals_from_ztest = []
for box in groups:
    z, p_from_ztest = z_test(box, pop_mean, pop_std)
    z_vals.append(z)
    p_vals_from_ztest.append(p_from_ztest)

print(f"z_vals {z_vals}")
print(p_vals_from_ztest)

# plot_z_test(z_vals, pop_mean)
plot_multiple_samples_vs_hypothesized_mean(groups, box_names, pop_mean)
binary_ztest_p_values = [1 if pval < 0.05 else 0 for pval in p_vals_from_ztest]
print(f"significant from z-test: {binary_ztest_p_values}")

# Scatter Plot
plt.scatter([i for i in range(len(binary_ztest_p_values))], binary_ztest_p_values, marker='o', alpha=1, label="Box")
plt.title("Grape Box Z-test P-val") 
plt.xlabel("Box")
plt.ylabel("1=significant 0=Not")
# plt.legend()
plt.show()



# individual z-tests

def single_observation_z_test(x, mu_0, sigma, alpha=0.05):
    """
    Check if a single observation x differs significantly
    from a hypothesized normal distribution (mean=mu_0, sd=sigma).

    Returns:
        z_value: float
        p_value: float
        significant: bool (True if p < alpha)
    """
    # Compute z
    z_value = (x - mu_0) / sigma
    
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z_value)))
    
    # Compare p-value with significance level alpha
    significant = (p_value < alpha)
    
    return z_value, p_value, significant

single_z = []
single_z_p = []
sig = []
for point in distances:
    z, p, significant = single_observation_z_test(point,pop_mean, pop_std)
    single_z.append(z)
    single_z_p.append(p)
    sig.append(significant)


# Scatter Plot
## 
plt.scatter([i for i in range(len(single_z))], single_z, marker='o', alpha=1, label="sample")
plt.title("Grape individual z-test") 
plt.xlabel("Grape Sample")
plt.ylabel("z-val")
# plt.legend()
plt.show()


# Scatter Plot
single_z_p_bin = [1 if pval < 0.05 else 0 for pval in single_z_p]
plt.scatter([i for i in range(len(single_z_p_bin))], single_z_p_bin, marker='o', alpha=1, label="sample")
plt.title("Grape individual p-value") 
plt.xlabel("Grape Sample")
plt.ylabel("p-val")
# plt.legend()
plt.show()



## Mann Whitey U Test

# stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

# t_test
u_stats = []
u_p_values = []
for box1 in groups:
    s_row = []
    p_row = []
    for box2 in groups:
        s, p = mannwhitneyu(box1, box2, alternative='two-sided')
        s_row.append(s)
        p_row.append(p)
    u_stats.append(s_row)
    u_p_values.append(p_row)

binary_u_p_values = [[1 if pval < 0.05 else 0 for pval in row] for row in u_p_values]

u_stats = np.array(u_stats)
binary_u_p_values = np.array(binary_u_p_values)
u_p_values = np.array(u_p_values)

plot_distance_heatmap(u_stats, title="Grape U-Test U Val", cmap="coolwarm", labels=box_names)
plot_distance_heatmap(binary_u_p_values, title="Grape U-Test Binary P Val", cmap="coolwarm", labels=box_names)
plot_distance_heatmap(u_p_values, title="Grape U-Test P Val", cmap="coolwarm", labels=box_names)