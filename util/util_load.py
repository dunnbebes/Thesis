import xlwings as xw
import pandas as pd
import numpy as np


def load(defined_name):
    wb                 = xw.Book(r"D:\D\University\4th year\Thesis\Code T4\Small_example.xlsx")
    named_range        = wb.names[defined_name]
    data               = named_range.refers_to_range.value
    df                 = pd.DataFrame(data)
    return df


def machine_data(J, I, K, n_j, MachineCapability):
    MC_ji              = [[[] for _ in range(I)] for _ in range(J)]
    n_MC_ji            = [[[] for _ in range(I)] for _ in range(J)]
    h_ijk              = np.zeros((I, J, K))

    for _, row in MachineCapability.iterrows():

        j              = row['Job']
        i              = row['Operation']
        machines       = row['Machine capable']

        machines_list  = [int(machine.strip()) for machine in machines.split(',') if machine.strip()]
        MC_ji   [j][i] = machines_list
        n_MC_ji [j][i] = int(len(machines_list))
    
    for j in range(J):
        for i in range(int(n_j[j])):
            for k in range(K):
                if k in MC_ji[j][i]:
                    h_ijk[i][j][k] = 1

    return MC_ji, n_MC_ji, h_ijk


def read_txt(filename):
    # Read data from the text file
    with open(filename, "r") as file:
        data = file.readlines()

    # Extracting parameters J, K, I
    J = int(data[0].strip())  # Number of jobs
    K = int(data[1].strip())  # Number of machines
    I = max(map(int, data[2].strip().split()))  # Maximum number of operations per job
    d_j = np.array(list(map(int, data[3].strip().split())), dtype=int)

    # Initialize p_ijk matrix, n_j vector, and MC_ji list of lists
    p_ijk = np.full((I, J, K), 999)
    h_ijk = np.zeros((I, J, K), dtype=int)

    n_j     = np.zeros(J, dtype=int)
    MC_ji   = [[[] for _ in range(I)] for _ in range(J)]
    n_MC_ji = [[0 for _ in range(I)] for _ in range(J)]

    # Extract job, operation, machine, and processing time data
    for line in data[4:]:
        job, operation, machine, processingtime = map(int, line.split())
        job       = int(job)       - 1     # Adjust to 0-based indexing
        operation = int(operation) - 1     # Adjust to 0-based indexing
        machine   = int(machine)   - 1     # Adjust to 0-based indexing
        # Update p_ijk and h_ijk matrix
        p_ijk[operation, job, machine] = processingtime
        h_ijk[operation, job, machine] = 1

        # Update n_j vector
        n_j[job] = max(n_j[job], operation + 1)

        # Update MC_ji list
        MC_ji[job][operation].append(machine)
        n_MC_ji[job][operation] += 1
    
    OperationPool = pd.DataFrame({
        "Job": np.arange(J), 
        "Num operation left": n_j  
    })

    return J, I, K, p_ijk, h_ijk, d_j, n_j, MC_ji, n_MC_ji, OperationPool

def read_scenario (file_path, K, critical_machines):
    JA_event = []
    MB_event = [[] for _ in range(K)]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_section = None

        for line in lines:
            if 'Defected Jobs' in line:
                current_section = 'JA'
                continue
            elif 'MachineBreakdown' in line:
                current_section = 'MB'
                continue
            elif line.strip() == '':
                continue

            if current_section == 'JA':
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        job_id = int(parts[0].strip())-1
                        deadline_info = int(parts[1].strip())
                        description = parts[2].strip()
                        JA_event.append((job_id, deadline_info, description))
                    except ValueError:
                        # Skip lines that cannot be converted to integer (likely headers or malformed data)
                        continue
            elif current_section == 'MB':
                parts = line.split()
                if len(parts) == 3:
                    try:
                        machine_id = int(parts[0].strip())
                        bd_time = float(parts[1].strip())
                        re_time = float(parts[2].strip())
                        description = "critical" if machine_id in critical_machines else "normal"
                        MB_event[machine_id - 1].append((bd_time, re_time, description))
                    except ValueError:
                        continue

    return JA_event, MB_event
    