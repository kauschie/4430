import numpy as np
import math
import csv
import matplotlib
matplotlib.use('TkAgg')

steam_purch_path = 'MeasurementsExamples_SteamPurchase.csv'
steam_list_path = 'MeasurementsExamples_SteamList.csv'
iris_path = 'MeasurementsExamples_iris.csv'
judge_path = 'MeasurementsExamples_Judges.csv'

def read_steamPurchase(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        p1, p2, p3, p4, p5 = [], [], [], [], []
        persons = [p1, p2, p3, p4, p5]
        headers = next(reader)  # Read header row
        for row in reader:
            for i in range(len(headers)):
                if i < len(row) and row[i] == "":  # Check if column exists and is empty
                    # print("EMPTY", end=", ")
                    continue
                else:
                    # print(f"{row[i]}", end=", ")  # Placeholder for non-empty values
                    persons[i].append(int(row[i]))
            # print()
        for i in range(len(headers)):
            data[headers[i]]=persons[i]
        # print(data)
        # print()
        
    return data
def read_steamList(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        app_id, name, genre = [], [], []
        headers = next(reader)  # Read header row
        for row in reader:
            app_id.append(row[0])
            name.append(row[1])
            genre.append(row[2])
        data = {headers[0]:app_id, headers[1]:name, headers[2]:genre}

        # print(data)
        # print()
    return data
# Dataset order: Sepal length, Sepal width, Petal length, Petal width, Species 
def read_iris(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        s_len, s_wid, p_len, p_wid, species = [], [], [], [], []
        g1, g2, g3 = {}, {}, {}

        prev_class = None
        g2_start = None
        g3_start = None

        count = 0
        headers = ["Sepal length", "Sepal width", "Petal length", "Petal width", "Species"]  # Read header row
        for row in reader:
            f1 = float(row[0])
            f2 = float(row[1])
            f3 = float(row[2])
            f4 = float(row[3])
            s = row[4]

            # init prev_class

            if prev_class == None:
                prev_class = s
            elif prev_class != s:
                if g2_start == None:
                    g2_start = count
                    prev_class = s
                elif g3_start == None:
                    g3_start = count
                    prev_class = s

            s_len.append(f1)
            s_wid.append(f2)
            p_len.append(f3)
            p_wid.append(f4)
            species.append(s)
            count += 1
                
        all = {headers[0]:s_len, 
                headers[1]:s_wid, 
                headers[2]:p_len,
                headers[3]:p_wid,
                headers[4]:species}
        g1 = {headers[0]:s_len[:g2_start], 
                headers[1]:s_wid[:g2_start], 
                headers[2]:p_len[:g2_start],
                headers[3]:p_wid[:g2_start]}
        g2 = {headers[0]:s_len[g2_start:g3_start], 
                headers[1]:s_wid[g2_start:g3_start], 
                headers[2]:p_len[g2_start:g3_start],
                headers[3]:p_wid[g2_start:g3_start]}
        g3 = {headers[0]:s_len[g3_start:], 
                headers[1]:s_wid[g3_start:], 
                headers[2]:p_len[g3_start:],
                headers[3]:p_wid[g3_start:],}
        
        

        data = {"all": all, "g1": g1, "g2":g2, "g3":g3}
        # print(data)
    return data

def read_judges(filepath):
    data = {}
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [], [], [], [], [], [], [], [], [], []
        j1, j2, j3, j4, j5, j6 = [], [], [], [], [], []
        judges = [j1, j2, j3, j4, j5, j6]
        headers = next(reader)  # Read header row
        i = 0
        for row in reader:
            p1.append(float(row[0]))
            p2.append(float(row[1]))
            p3.append(float(row[2]))
            p4.append(float(row[3]))
            p5.append(float(row[4]))
            p6.append(float(row[5]))
            p7.append(float(row[6]))
            p8.append(float(row[7]))
            p9.append(float(row[8]))
            p10.append(float(row[9]))
            judges[i].append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])
            i += 1
        p_data = {headers[0]:p1, headers[1]:p2, headers[2]:p3,
                headers[3]:p4, headers[4]:p5, headers[0]:p6, 
                headers[1]:p7, headers[2]:p8, headers[3]:p9, 
                headers[4]:p10}
        j_data = {"Judge1":judges[0], "Judge2":judges[1], "Judge3":judges[2], "Judge4":judges[3], "Judge5":judges[4], "Judge6":judges[5]}
        data = {"Persons":p_data, "Judges":j_data}

        # print(data)
        # print()
    return data

def min_max_norm(list_of_vals):
    l_max = max(list_of_vals)
    l_min = min(list_of_vals)
    normalized_vals = []
    for v in list_of_vals:
        normalized_vals.append((v-l_min)/(l_max-l_min))

    print(f"min: {l_min}")
    print(f"max: {l_max}")
    return normalized_vals


def preprocess_iris():
    # outlier detection (intra-group)

    # min/max normalization (all group)

    # return preprocessed data
    pass

def main():
    # need nxn matrix for all
    steamPurchaseData = read_steamPurchase(steam_purch_path)
    steamListData = read_steamList(steam_list_path)
    irisData = read_iris(iris_path)
    judgeData = read_judges(judge_path)

    data = irisData['all']['Sepal length']
    norms = min_max_norm(data)
    print(f"data: {data}")
    print(f"norms: {norms}")


if __name__ == "__main__":
    main()