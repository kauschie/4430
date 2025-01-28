import csv
import os
import math

class DataPoint:
    def __init__(self, a, b, c=None):
        self.a = int(a)
        self.b = int(b)
        self.c = str(c)

    def __add__(self, other):
        if isinstance(other, DataPoint):
            return DataPoint(self.a + other.a, self.b + other.b)
        return NotImplemented
        
    def __sub__(self, other):
        if isinstance(other, DataPoint):
            return DataPoint(abs(self.a - other.a), abs(self.b - other.b))
        return NotImplemented
        
    def distance(self, other):
        if isinstance(other, DataPoint):
            d = other - self
            dist = math.sqrt(pow(d.a,2) + pow(d.b,2))
            return dist
        return NotImplemented

    def __str__(self):
        return (f"a: {self.a}\t b: {self.b}\t c: {self.c}")

#Q takes filepath
def read_csv(filepath):
    with open(filepath, 'r', newline = '')  as file:
        m_data = []
        f_data = []
        unk_data = []
        all_data = []
        reader = csv.reader(file)
        for row in reader:
            # print(f"len(row): {len(row)}")
            # print(row)
            if len(row) == 2:
                d = DataPoint(row[0], row[1])
                # print(d)
                unk_data.append(d)
            elif len(row) == 3:
                d = DataPoint(row[0], row[1], row[2])
                # print(d)
                # print(d.c.lower())
                if d.c.lower() == "m":
                    m_data.append(d)
                elif d.c.lower() == "f":
                    f_data.append(d)
                else:
                    print(f"unknown var in col 3 found: {d.c}")
                    os._exit(1)
                all_data.append(d)
            else:
                print("poorly formated csv file for job, exiting")
                os._exit(1)

        data = {'m': m_data, 'f': f_data, 'u':unk_data, 'all':all_data}
    return data
        
# input is list of DataPoint's
def get_center(data_list):

    min = None
    min_sum = 0
    
    for test_center in data_list:
        # print(f"test_center: {test_center}")
        sum = 0
        for d in data_list:
            dist = test_center.distance(d)
            sum += dist
        
        # print(f"sum from {test_center}: {sum}")

        if min is None or sum < min_sum:
            min = test_center
            min_sum = sum
        
    return min, min_sum

def fix_data(input_file, m_center, f_center):
    rows = []
    with open(input_file, 'r', newline = '') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] == "":
                d = DataPoint(row[0], row[1])
                row[2], _, _ = get_class(d, m_center, f_center)
                rows.append(row[2])
    with open("fixed_data.txt", mode='w') as fout:
        for c in rows:
            fout.write(c)
    
    print("Finished fixing data. Data output to fixed_data.csv")
    

def get_class(data_point, m_center, f_center):
    m_dist = m_center.distance(data_point)
    f_dist = f_center.distance(data_point)

    if m_dist < f_dist:
        c = 'M'
    elif f_dist < m_dist:
        c = 'F'
    else:
        c = 'U'

    return (c, m_dist, f_dist)


def classify_data(data, m_center, f_center):
    for d in data:
        c, m, f = get_class(d, m_center, f_center)
        print(f"{d} is most likely {c} (m: {m}\t f: {f})")

# Takes CSV reader object
def print_data(data):
    for k,v in data.items():
        print(f"{k}:")
        for d in v:
            print(d)

if __name__ == "__main__":
    # Read in training Data
    training_data = read_csv('KNN_armyTraining1.csv')
    # print("----Training Data---")
    # print_data(training_data)
    # Read in testing Data
    testing_data = read_csv('KNN_armyTesting1.csv')
    # print("\n----Testing Data---")
    # print_data(testing_data)
    # print("\n")

    # find center point of m:
    m_center, m_min = get_center(training_data['m'])

    # find center point of f:
    f_center, f_min = get_center(training_data['f'])

    print("Results:")
    print(f"min {m_center.c} value was {m_center} with sum {m_min}")
    print(f"min {f_center.c} value was {f_center} with sum {f_min}")

    # classify_data(testing_data['u'], m_center, f_center)

    all_data_path = "KNN_armyAllData1.csv"

    fix_data(all_data_path, m_center, f_center)