from collections import defaultdict



# read in data of format:
# sex,age,urbanization,education,geographic_area,allergy,smoke,sedentary,asthma
# male,adult,low,low,south/islands,yes,yes,yes,yes

def read_data(file_name):
    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    header = lines[0].split(',')
    data = [dict(zip(header, row.split(','))) for row in lines[1:]]
    return header, data


def read_dag(dag_file):
    """
    should have the following format:
    dag = {
    "asthma": ["allergy", "smoke", "sedentary"],
    ...
    }   
    """
    dag = {}
    with open(dag_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(',')
                node = parts[0].strip()
                if len(parts) > 1:
                    parents = [part.strip() for part in parts[1:]]
                else:
                    parents = []
                dag[node] = parents
    return dag



def init_cpts(dag):
    cpts = {}
    for node, parents in dag.items():
        if parents:  # node has parents → needs a CPT
            cpts[node] = defaultdict(int)         # counts of full combinations
            cpts[f"{node}_total"] = defaultdict(int)  # parent-only counts for denominator
    return cpts



def update_cpts(data_row, dag, cpts, cpts_total):
    """
    Updates CPTs including prior (no-parent) nodes and conditional nodes.
    """
    for node, parents in dag.items():
        node_val = data_row[node]
        
        if parents:
            parent_vals = tuple(data_row[parent] for parent in parents)
            key = parent_vals + (node_val,)
            cpts[node][key] += 1
            cpts_total[node][parent_vals] += 1
        else:
            # Prior node — no parents to condition on
            cpts[node][(node_val,)] += 1
            cpts_total[node]["__total__"] += 1  # special key for global count


def main():
    header, data_rows = read_data("BN_Asthma_data.csv")
    dag = read_dag("dag.txt")

    # Initialize
    from collections import defaultdict
    cpts = defaultdict(lambda: defaultdict(int))
    cpts_total = defaultdict(lambda: defaultdict(int))

    # Update all CPTs
    for row in data_rows:
        update_cpts(row, dag, cpts, cpts_total)

    print("CPTs:")
    print("=====")
    for key, value in cpts.items():
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    # print(cpts)
    print("\nCPTs Total:")
    print("===========")
    for key, value in cpts_total.items():
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    # print(cpts_total)

if __name__ == "__main__":
    main()