import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import csv
import os
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
from collections import Counter
# from mpl_toolkits.mplot3d import Axes3D


# plots to do
dataset1 = "KNN_armyTraining1.csv"
dataset2 = "DataCentralTendency_grape_data.txt"


def read_ds1(filepath):
    with open(filepath, 'r', newline = '')  as file:
        all_heights = []
        all_weights = []
        all_classifications = []

        m_heights = []
        m_weights = []

        f_heights = []
        f_weights = []

        reader = csv.reader(file)
        for row in reader:
            if len(row) == 3:
                height = int(row[0])
                weight = int(row[1])
                classification = row[2].lower()
                all_heights.append(height)
                all_weights.append(weight)
                all_classifications.append(classification)

                if classification == 'm':
                    m_heights.append(height)
                    m_weights.append(weight)
                elif classification == 'f':
                    f_heights.append(height)
                    f_weights.append(weight)

            else:
                print("poorly formated csv file for job, exiting")
                os._exit(1)
        all_data = {'h': all_heights, 'w': all_weights, 'c': all_classifications}
        m_data = {'h': m_heights, 'w': m_weights}
        f_data = {'h': f_heights, 'w': f_weights}
        data = {'all': all_data, 'm': m_data, 'f': f_data}
    
    return data

def read_ds2(filepath):
    with open(filepath, 'r', newline = '') as file:
        distances = []
        classifications = []
        i = 0
        for row in file:
            if i == 0:
                distances = row.split(',')
                i += 1
            elif i == 1:
                classifications = row.strip().split(',')
        # print(len(distances))
        # print(len(classifications))
        data = {'d':[float(d) for d in distances], 'c':classifications}
        return data


def get_cnt_sum(x_arr, y_arr):
    data = {}
    for i in range(len(x_arr)):
        sum, cnt = data.get(x_arr[i],(0,0))
        data[x_arr[i]] = ((sum + y_arr[i]), cnt+1)
    return data

def avg_data(x_arr, y_arr):
    # print(f"x_arr: {x_arr}")
    # print(f"y_arr: {y_arr}")
    data = get_cnt_sum(x_arr, y_arr)

    sorted_x = sorted(data.keys())
    sorted_y = [data[x][0] / data[x][1] for x in sorted_x]

    # print(f"sorted_x: {sorted_x}")
    # print(f"sorted_y: {sorted_y}")

    return sorted_x, sorted_y


def get_height_weight_avg(heights, weights, max_h, min_h, max_w, min_w):
    count_h = len(heights)
    count_w = len(weights)
    if count_h != count_w:
        print(f"count_h{count_h} != count_w{count_w}")
        exit(1)

    height_avg = sum(heights) / count_h
    weight_avg = sum(weights) / count_w


    heights_n = [((h-min_h)/(max_h-min_h)) for h in heights]
    weights_n = [((w-min_w)/(max_w-min_w)) for w in weights]
    
    height_avg = sum(heights_n) / len(heights_n)
    weight_avg = sum(weights_n) / len(weights_n)
        

    return height_avg, weight_avg

def get_avg_bmi(heights_m, weights_m, heights_f, weights_f):
    bmis_m = []
    bmis_f = []
    for i in range(len(heights_m)):
        bmis_m.append((weights_m[i]/(heights_m[i]**2))*703)
    for i in range(len(heights_f)):
        bmis_f.append((weights_f[i]/(heights_f[i]**2))*703)

    max_bmi = max(max(bmis_m), max(bmis_f))
    min_bmi = min(min(bmis_m), min(bmis_f))
    

    bmis_m = [((b-min_bmi)/(max_bmi-min_bmi)) for b in bmis_m]
    bmis_f = [((b-min_bmi)/(max_bmi-min_bmi)) for b in bmis_f]

    return (sum(bmis_m)/len(bmis_m), sum(bmis_f)/len(bmis_f))

def smooth_plot(x, y, label):
    x_smooth = np.linspace(min(x), max(x), 200)
    # spline = make_interp_spline(x, y, k=1)
    pchip = PchipInterpolator(x, y)
    y_smooth = pchip(x_smooth)

    plt.plot(x_smooth, y_smooth, label=label, linestyle="--" )


def make_pictogram_baby(groups, freq):
    # Define pictogram settings
    icon_size = 100  # Number of units each icon represents
    num_rows = 10    # Max rows in pictogram
    # Define marker types (emoji symbols)
    markers = ['o', 's']
    colors = ['royalblue', 'darkred']
    g = ['100 Not Merlot Grapes', '100 Merlot Grapes']

    # Create Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate pictogram
    x_spacing = 7  # Horizontal spacing between groups
    y_spacing = 1  # Vertical spacing

    for i, (label, count) in enumerate(zip(groups, freq)):
        num_icons = int(count / icon_size)  # Scale frequency to number of icons
        x_pos = []  # X-coordinates
        y_pos = []  # Y-coordinates

        for j in range(num_icons):
            row = j // num_rows  # Row index
            col = j % num_rows   # Column index
            x = i * x_spacing + col * 0.7  # Adjust X position for each icon
            y = -row * y_spacing  # Adjust Y position to stack
            
            # **Use `text()` instead of scatter to place emojis**
            ax.scatter(x, y, s=300, color=colors[i], marker=markers[i], alpha=0.8)

    # Customize appearance
    ax.set_xlim(-1, x_spacing * len(groups))  # Adjust x-axis limits
    ax.set_ylim(-1, 1)  # Adjust height to fit icons
    ax.set_xticks([])
    # ax.set_xticklabels(groups, fontsize=12, fontweight='bold')
    ax.set_yticks([])  # Hide y-axis numbers
    ax.set_title("Pictogram of Grape Count", fontsize=14, fontweight='bold')

    legend_handles = [mlines.Line2D([], [], color='w', marker=markers[i], markerfacecolor=colors[i], markersize=10, label=g[i]) for i in range(len(groups))]
    ax.legend(handles=legend_handles, loc="upper right", title="Groups")
    plt.show()


def make_bubble_graph(ds1, num_bins=20, scale_factor=5):


    all_heights = ds1['m']['h'] + ds1['f']['h']
    all_weights = ds1['m']['w'] + ds1['f']['w']

    # Use the same bins for both M and F
    # height_bins = np.linspace(min(all_heights), max(all_heights), num_bins)
    # weight_bins = np.linspace(min(all_weights), max(all_weights), num_bins)

    height_bins = np.arange(min(all_heights), max(all_heights), 2)
    weight_bins = np.arange(min(all_weights), max(all_weights), int((max(all_weights)-min(all_weights))/20))
    

    # # 1. Define bins for males
    # height_bins_m = np.linspace(min(ds1['m']['h']), max(ds1['m']['h']), num_bins)
    # weight_bins_m = np.linspace(min(ds1['m']['w']), max(ds1['m']['w']), num_bins)

    # # 2. Define bins for females
    # height_bins_f = np.linspace(min(ds1['f']['h']), max(ds1['f']['h']), num_bins)
    # weight_bins_f = np.linspace(min(ds1['f']['w']), max(ds1['f']['w']), num_bins)

    # 3. Create 2D histograms
    H_m, xedges_m, yedges_m = np.histogram2d(ds1['m']['h'], ds1['m']['w'], 
                                             bins=[height_bins, weight_bins])
    H_f, xedges_f, yedges_f = np.histogram2d(ds1['f']['h'], ds1['f']['w'], 
                                             bins=[height_bins, weight_bins])

    H_m = H_m.T
    H_f = H_f.T

    # 4. Compute bin centers for males
    x_centers_m = 0.5 * (xedges_m[:-1] + xedges_m[1:])
    y_centers_m = 0.5 * (yedges_m[:-1] + yedges_m[1:])
    X_m, Y_m = np.meshgrid(x_centers_m, y_centers_m)

    # Flatten for use in plt.scatter()
    X_m_flat = X_m.ravel()
    Y_m_flat = Y_m.ravel()
    H_m_flat = H_m.ravel()

    # 5. Compute bin centers for females
    x_centers_f = 0.5 * (xedges_f[:-1] + xedges_f[1:])
    y_centers_f = 0.5 * (yedges_f[:-1] + yedges_f[1:])
    X_f, Y_f = np.meshgrid(x_centers_f, y_centers_f)

    X_f_flat = X_f.ravel()
    Y_f_flat = Y_f.ravel()
    H_f_flat = H_f.ravel()

    # 6. Create the bubble plot
    plt.figure(figsize=(8, 6))

    # Males
    plt.scatter(
        X_m_flat, Y_m_flat,
        s = H_m_flat * scale_factor,  # Bubble size ~ frequency
        alpha = 0.3,
        color = 'blue',
        label = 'Male'
    )
    # print(H_m_flat * scale_factor)

    # Females
    plt.scatter(
        X_f_flat, Y_f_flat,
        s = H_f_flat * scale_factor,
        alpha = 0.3,
        color = 'red',
        label = 'Female'
    )

    # plt.xticks(x_centers_m)
    # plt.yticks(y_centers_m)
    plt.xlabel('Height Bin')
    plt.ylabel('Weight Bin')
    ax = plt.gca()
    ax.set_xlim(left=55)
    plt.locator_params(axis='x', nbins=20)  # tries to create up to 10 x-ticks
    plt.locator_params(axis='y', nbins=20)  # tries to create up to 10 y-ticks
    plt.title('Bubble Plot: Height vs. Weight Frequency')
    plt.legend()
    plt.show()



# Function to hide Pie chart percentage
def my_autopct(pct):
    return '{:.0f}%'.format(pct) if pct > 0 else ''

# Normalize data using Min-Max Scaling
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    ds1 = read_ds1(dataset1) # army
    # print((ds1['all']['h']))
    # print(len(ds1['m']['h']))
    # print(len(ds1['f']['h']))
    ds2 = read_ds2(dataset2) # grape
    # print(len(ds2['d']))
    # print(len(ds2['c']))
    
    
    # Line Graph
    # get average weights at each height, plot by group

    sorted_avg_x_male, sorted_avg_y_male = avg_data(ds1['m']['h'], ds1['m']['w'])
    sorted_avg_x_female, sorted_avg_y_female = avg_data(ds1['f']['h'], ds1['f']['w'])
    plt.plot(sorted_avg_x_male, sorted_avg_y_male, marker='o', label="Male")
    plt.plot(sorted_avg_x_female, sorted_avg_y_female, marker='o', label="Female")
    
    # Spline
    smooth_plot(sorted_avg_x_male, sorted_avg_y_male, "Male Interpolated")
    smooth_plot(sorted_avg_x_female, sorted_avg_y_female, "Female Interpolated")
    
    plt.title("Avg Weight by Height Line Graph / Spline (Army Data)")
    plt.xlabel("Height")
    plt.ylabel("Average Weight")
    plt.legend()
    plt.show()


    # Scatter Plot
    plt.scatter(ds1['m']['h'], ds1['m']['w'], marker='o', alpha=0.8, label="Male")
    plt.scatter(ds1['f']['h'], ds1['f']['w'], marker='o', alpha=0.8, label="Female")
    plt.title("Weight by Height Scatter Plot(Army Data)")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()

    # Male BMI Pie Chart

    # Get BMIs for Men and Women

    bmis_m = []
    bmis_f = []

    for i in range(len(ds1['m']['h'])):
        bmis_m.append((ds1['m']['w'][i]/(ds1['m']['h'][i]**2))*703)
    for i in range(len(ds1['f']['h'])):
        bmis_f.append((ds1['f']['w'][i]/(ds1['f']['h'][i]**2))*703)

    # Convert lists to numpy arrays for easier processing
    bmis_m = np.array(bmis_m)
    bmis_f = np.array(bmis_f)

    # Define BMI categories based on standard cutoffs
    # Underweight: BMI < 18.5
    # Normal Weight: 18.5 <= BMI < 25
    # Overweight: 25 <= BMI < 30
    # Obese: BMI >= 30

    male_underweight, female_underweight = np.sum(bmis_m < 18.5), np.sum(bmis_f < 18.5)
    male_normal, female_normal = np.sum((bmis_m >= 18.5) & (bmis_m < 25)), np.sum((bmis_f >= 18.5) & (bmis_f < 25))
    male_overweight, female_overweight = np.sum((bmis_m >= 25) & (bmis_m < 30)), np.sum((bmis_f >= 25) & (bmis_f < 30))
    male_obese, female_obese = np.sum(bmis_m >= 30), np.sum(bmis_f >= 30)

    # Create a list of counts for each category
    male_bmi_counts = [male_underweight, male_normal, male_overweight, male_obese]
    female_bmi_counts = [female_underweight, female_normal, female_overweight, female_obese]

    # Define labels and colors for the pie charts
    bmi_labels = ['Underweight (< 18.5)', 'Normal Weight (18.5 - 24.9)', 'Overweight (25 - 29.9)', 'Obese (30+)']
    male_bmi_colors = sns.color_palette("blend:#1D2545,#B8E3E9")[0:4]  # Get 4 pastel colors
    female_bmi_colors = sns.color_palette("blend:#B53737,#FFD1DF")[0:4]

    # Plot the pie chart for Male BMI distribution
    plt.figure(figsize=(10, 8))
    plt.pie(male_bmi_counts, labels=bmi_labels, colors=male_bmi_colors, autopct=my_autopct)
    plt.title("Male BMI Distribution")
    plt.show()

    # Plot the pie chart for Female BMI distribution
    plt.figure(figsize=(10, 8))
    plt.pie(female_bmi_counts, labels=bmi_labels, colors=female_bmi_colors, autopct=my_autopct)
    plt.title("Female BMI Distribution")
    plt.show()

    # Bar
    bar_data = get_cnt_sum(ds2['c'], ds2['d'])
    x_vals = sorted(bar_data.keys())
    y_vals = [bar_data[x][1] for x in x_vals]

    # print(f"bar_data: {bar_data}")
    mylabels = ['Not Merlot', 'Merlot']
    colors = ['red', 'blue']
    plt.bar(x_vals, y_vals, label=mylabels, color=colors)
    plt.xlabel("Group")
    plt.ylabel("Number of Grapes")
    plt.legend()
    plt.title("Group Size")
    plt.show()
    
    # Pie
    my_explode = [0, 0.2]
    mylabels = ['Not Merlot', 'Merlot']
    plt.pie(y_vals, labels=mylabels, explode=my_explode, startangle=-20, autopct='%1.2f%%' )
    plt.title("Proportion of Groups in Total Sample")
    plt.legend(title = "Groups")
    plt.show()




    # Bubble Chart

    make_bubble_graph(ds1, scale_factor=20)


    # Area

    male_heights = ds1['m']['h']
    fem_heights = ds1['f']['h']

    m_freq = {}
    for m in male_heights:
        cnt = m_freq.get(m, 0)
        m_freq[m] = cnt + 1

    f_freq = {}
    for f in fem_heights:
        cnt = f_freq.get(f, 0)
        f_freq[f] = cnt + 1

    # x_vals_m = m_freq.keys()
    # x_vals_f = f_freq.keys()

    x_vals = sorted(set(m_freq.keys()) | set(f_freq.keys()))

    y_vals_m = []
    y_vals_f = []
    for x in x_vals:
        y_vals_m.append(m_freq.get(x,0))
        y_vals_f.append(f_freq.get(x,0))


    # Create Figure with Two Subplots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ##  Left: `stackplot()`
    ##  - traditional stacking method
    ##  - data may be obscured
    plt.stackplot(x_vals, y_vals_f, y_vals_m, labels=["Female", "Male"], colors=["darkred", "royalblue"], alpha=0.8)
    plt.xlabel("Height")
    plt.ylabel("Count")
    plt.title("Stacked Area Chart")
    plt.legend(title="Groups")

    ##  Right: `fill_between()`
    ## - shows both groups independently 
    ## - transparency ensures both are visible
    # axes[1].fill_between(x_vals, y_vals_m, alpha=0.6, label="Male", color="royalblue")
    # axes[1].fill_between(x_vals, y_vals_f, alpha=0.6, label="Female", color="darkred")
    # axes[1].set_xlabel("Height")
    # axes[1].set_ylabel("Count")
    # axes[1].set_title("Simple Area Chart")
    # axes[1].legend(title="Groups")

    # Adjust Layout
    plt.tight_layout()
    plt.show()


    # Dot Graph
    df = pd.DataFrame(ds2)
    # custom_palette = {'0': 'royalblue', '1': 'darkred'}

    sns.set_style('whitegrid')
    plt.figure(figsize=(8,6))
    ax = sns.stripplot(x='c', y='d', data=df, size=4, jitter=True, alpha=0.6)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Not Merlot', 'Merlot'])

    # Customize Labels
    plt.xlabel('Group')
    plt.ylabel('Diameter (cm)')
    plt.title('Cleveland Dot Graph - Diameter by Group')

    plt.show()




    # Radar

    max_h = max(max(ds1['m']['h']), max(ds1['f']['h']))
    max_w = max(max(ds1['m']['w']), max(ds1['f']['w']))
    min_h = min(min(ds1['m']['h']), min(ds1['f']['h']))
    min_w = min(min(ds1['m']['w']), min(ds1['f']['w']))


    avg_m_h, avg_m_w = get_height_weight_avg(ds1['m']['h'], ds1['m']['w'], max_h, min_h, max_w, min_w)
    avg_w_h, avg_w_w = get_height_weight_avg(ds1['f']['h'], ds1['f']['w'], max_h, min_h, max_w, min_w)
    avg_m_bmi, avg_f_bmi = get_avg_bmi(ds1['m']['h'], ds1['m']['w'], ds1['f']['h'], ds1['f']['w'])

    # Categories we'll plot around the radar
    categories = ["Average Height", "Average Weight", "Average BMI"]

    # Values for Males and Females
    male_vals = [avg_m_h, avg_m_w, avg_m_bmi]
    female_vals = [avg_w_h, avg_w_w, avg_f_bmi]


    # Number of variables (e.g., Height, Weight, BMI)
    N = len(categories)

    # Compute angle for each category (in radians), plus repeat the first to close the circle
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # repeat the first angle

    # Radar chart is a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 7))

    # --------
    # Plot for Males
    # --------
    male_vals_cycle = male_vals + male_vals[:1]  # repeat first value to close polygon
    ax.plot(angles, male_vals_cycle, color='blue', linewidth=2, label='Male')
    ax.fill(angles, male_vals_cycle, color='blue', alpha=0.1)  # optional fill

    # --------
    # Plot for Females
    # --------
    female_vals_cycle = female_vals + female_vals[:1]
    ax.plot(angles, female_vals_cycle, color='red', linewidth=2, label='Female')
    ax.fill(angles, female_vals_cycle, color='red', alpha=0.1)

    # --------
    # Fix the x-axis (around the circle) to use our categories
    # --------
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Optionally, adjust the radial limits if your values differ significantly
    # ax.set_ylim(0, max(male_vals + female_vals) * 1.2)

    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Radar Chart: Comparing Male vs. Female Averages", y=1.08)

    plt.show()

    # Pictogram

    bar_data = get_cnt_sum(ds2['c'], ds2['d'])
    groups = sorted(bar_data.keys())
    freq = [bar_data[x][1] for x in groups]
    # picto groups: ['0', '1']
    # picto freq: [100, 1000]

    make_pictogram_baby(groups, freq)

    # Box Plot

    # convert to pandas df for seaborn to leverage grouping capabilities
    df = pd.DataFrame(ds2)
    sns.set_style('darkgrid')
    plt.figure(figsize=(8,6))
    sns.stripplot(x='c', y='d', data=df, color='gray', size=4, jitter=True, alpha=0.4)
    ax = sns.boxplot(x='c', y='d', data=df)
    # Overlay Individual Data Points (stripplot)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Not Merlot', 'Merlot'])

    # Customize Labels
    plt.xlabel('')
    plt.ylabel('Diameter (cm)')
    plt.title('Box Plot of Diameter by Group')
    plt.show()

    # Histogram

    # Convert category values to integers if they aren't already
    categories = [int(c) for c in ds2['c']]
    diameters = ds2['d']  # Assuming these are numeric already

    # Separate the diameters based on the category
    d0 = [d for i, d in enumerate(diameters) if categories[i] == 0]
    d1 = [d for i, d in enumerate(diameters) if categories[i] == 1]

    # print("First 10 values for d0:", d0[:10])
    # print("First 10 values for d1:", d1[:10])

    # Determine the bins for a bin width of 1 mm.
    min_d = int(np.floor(min(diameters)))
    max_d = int(np.ceil(max(diameters)))
    bins = np.arange(min_d, max_d + 1, 1)

    # Plot the stacked histogram
    plt.figure(figsize=(10, 6))
    plt.hist([d0, d1], bins=bins, stacked=True, color=['#DDD5F3', '#B19CD7'], label=['Blueberry', 'Merlot'])
    plt.xlabel("Diameter (mm)")
    plt.ylabel("Frequency")
    plt.title("Stacked Histogram of Grape Diameters by Category")
    plt.legend()
    plt.show()

    # Heatmap

    num_bins = 8

    # Compute bin edges using min/max values for consistency across both sets
    height_bins_m = np.linspace(min(ds1['m']['h']), max(ds1['m']['h']), num_bins)
    weight_bins_m = np.linspace(min(ds1['m']['w']), max(ds1['m']['w']), num_bins)
    height_bins_f = np.linspace(min(ds1['f']['h']), max(ds1['f']['h']), num_bins)
    weight_bins_f = np.linspace(min(ds1['f']['w']), max(ds1['f']['w']), num_bins)

    # Create 2D histograms
    H_m, xedges_m, yedges_m = np.histogram2d(ds1['m']['h'], ds1['m']['w'], bins=[height_bins_m, weight_bins_m])
    H_f, xedges_f, yedges_f = np.histogram2d(ds1['f']['h'], ds1['f']['w'], bins=[height_bins_f, weight_bins_f])

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Male Heatmap
    ax1 = axes[0]
    c1 = ax1.pcolormesh(xedges_m, yedges_m, H_m.T, cmap='coolwarm', shading='auto')
    fig.colorbar(c1, ax=ax1)
    ax1.set_xlabel('Height (in)')
    ax1.set_ylabel('Weight (lbs)')
    ax1.set_title('Male Height-Weight Distribution')

    # Female Heatmap
    ax2 = axes[1]
    c2 = ax2.pcolormesh(xedges_f, yedges_f, H_f.T, cmap='coolwarm', shading='auto')
    fig.colorbar(c2, ax=ax2)
    ax2.set_xlabel('Height (in)')
    ax2.set_ylabel('Weight (lbs)')
    ax2.set_title('Female Height-Weight Distribution')

    plt.tight_layout()
    plt.show()

    # Mesh

    # Define Bins
    num_bins = 20

    # np.linspace(start, stop, num_bins) 
    #       - create array of num_bins evenly spaced between start and stop 
    #       - uses min/max of both data sets for consistency
    height_bins_m = np.linspace(min(ds1['m']['h']), max(ds1['m']['h']), num_bins)
    weight_bins_m = np.linspace(min(ds1['m']['w']), max(ds1['m']['w']), num_bins)

    height_bins_f = np.linspace(min(ds1['f']['h']), max(ds1['f']['h']), num_bins)
    weight_bins_f = np.linspace(min(ds1['f']['w']), max(ds1['f']['w']), num_bins)

    # # Create 2D histograms for pairs of values
    # #   - returns:
    # #       -  2d array (H) where each cell contains the count of points in that bin
    # #       -  bin edges for both x-axis and y-axis
    H_m, xedges_m, yedges_m = np.histogram2d(ds1['m']['h'], ds1['m']['w'], bins=[height_bins_m, weight_bins_m])
    H_f, xedges_f, yedges_f = np.histogram2d(ds1['f']['h'], ds1['f']['w'], bins=[height_bins_f, weight_bins_f])

    # H_m = H_m.astype(int)
    # H_f = H_f.astype(int)

    # # Compute bin centers
    # #   - generate grid of X, Y coordinates based on bin centers
    # #   - useful for mapping bin centers to z values for 3d plotting
    Xm, Ym = np.meshgrid((xedges_m[:-1] + xedges_m[1:]) / 2, (yedges_m[:-1] + yedges_m[1:]) / 2)
    Xf, Yf = np.meshgrid((xedges_f[:-1] + xedges_f[1:]) / 2, (yedges_f[:-1] + yedges_f[1:]) / 2)

    Zm = H_m.T  # Male frequency matrix
    Zf = H_f.T  # Female frequency matrix

    # Zm = Zm.astype(int)
    # Zf = Zf.astype(int)

    # Create 3D Figure with Subplots
    fig = plt.figure(figsize=(12, 6))

    # Male Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Xm, Ym, Zm, cmap='Blues', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Height (inch)')
    ax1.set_ylabel('Weight (lbs)')
    ax1.set_zlabel('Count')
    ax1.set_title('Male: Height vs Weight Frequency 20 Bins')

    ax1.set_zticks(range(int(Zm.min()), int(Zm.max()) + 1))  

    # Female Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(Xf, Yf, Zf, cmap='Reds', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Height (inch)')
    ax2.set_ylabel('Weight (lbs)')
    ax2.set_zlabel('Count')
    ax2.set_title('Female: Height vs Weight Frequency 20 Bins')

    ax2.set_zticks(range(int(Zf.min()), int(Zf.max()) + 1))  

    plt.show()

    bin_info = {
        "Male Height Bins": xedges_m,
        "Male Weight Bins": yedges_m,
        "Female Height Bins": xedges_f,
        "Female Weight Bins": yedges_f
    }

    df_bins = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in bin_info.items()]))
    # print(df_bins)

    # Heatmaps visualizing distance from the mean

    # Height and Weight Normalization
    height_m_norm = normalize(ds1['m']['h'])
    weight_m_norm = normalize(ds1['m']['w'])
    height_f_norm = normalize(ds1['f']['h'])
    weight_f_norm = normalize(ds1['f']['w'])

    # Compute mean of normalized height and weight
    mean_height_m, mean_weight_m = np.mean(height_m_norm), np.mean(weight_m_norm)
    mean_height_f, mean_weight_f = np.mean(height_f_norm), np.mean(weight_f_norm)

    # Compute Euclidean distance from the mean
    distance_m = np.sqrt((height_m_norm - mean_height_m) ** 2 + (weight_m_norm - mean_weight_m) ** 2)
    distance_f = np.sqrt((height_f_norm - mean_height_f) ** 2 + (weight_f_norm - mean_weight_f) ** 2)

    # Define number of bins
    num_bins = 15

    # Create bins for height and weight
    height_bins = np.linspace(0, 1, num_bins)
    weight_bins = np.linspace(0, 1, num_bins)
    
    # Create 2D histogram bins weighted by the computed distances
    H_m, xedges_m, yedges_m = np.histogram2d(height_m_norm, weight_m_norm, bins=[height_bins, weight_bins], weights=distance_m)
    H_f, xedges_f, yedges_f = np.histogram2d(height_f_norm, weight_f_norm, bins=[height_bins, weight_bins], weights=distance_f)

    # Create a structured mesh grid for plotting (exclude the last bin edge)
    X_m, Y_m = np.meshgrid(xedges_m[:-1], yedges_m[:-1])
    X_f, Y_f = np.meshgrid(xedges_f[:-1], yedges_f[:-1])

    # Normalize histogram values for better scaling
    H_m = H_m / np.max(H_m)
    H_f = H_f / np.max(H_f)

    # -----------------------------
    # Plotting Section
    # -----------------------------

    # 1. Male Distance Plot (3D Surface)
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X_m, Y_m, H_m.T, cmap='Blues', edgecolor='k')  # Transpose H_m for alignment
    ax1.set_xlabel('Normalized Height')
    ax1.set_ylabel('Normalized Weight')
    ax1.set_zlabel('Distance from Mean')
    ax1.set_title('Male Distance from Mean')
    plt.show()
    # 2. Female Distance Plot (3D Surface)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X_f, Y_f, H_f.T, cmap='Reds', edgecolor='k')
    ax2.set_xlabel('Normalized Height')
    ax2.set_ylabel('Normalized Weight')
    ax2.set_zlabel('Distance from Mean')
    ax2.set_title('Female Distance from Mean')
    plt.show()

    # 3. Male Heat Map (2D)
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    im3 = ax3.imshow(H_m.T, cmap='Blues', origin='lower', extent=[0, 1, 0, 1])
    ax3.set_xlabel('Normalized Height')
    ax3.set_ylabel('Normalized Weight')
    ax3.set_title('Male Heat Map')
    plt.colorbar(im3, ax=ax3)  # Add color scale to the heat map
    plt.grid(False)
    plt.show()

    # 4. Female Heat Map (2D)
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111)
    im4 = ax4.imshow(H_f.T, cmap='Reds', origin='lower', extent=[0, 1, 0, 1])
    ax4.set_xlabel('Normalized Height')
    ax4.set_ylabel('Normalized Weight')
    ax4.set_title('Female Heat Map')
    plt.colorbar(im4, ax=ax4)
    plt.grid(False)
    plt.show()