import csv
import os

base = "Run_1_11_p0"


percentiles = [1,2,3,4,6,7,8,9]
seeds = [44,22,33]

results_path = "/usr/xtmp/vs196/ReRunOutputs/AllOracleRuns/"

run_folders = []

for root, dirs, _ in os.walk(results_path):
    if base in dirs:
        run_folders.append(os.path.join(root, dirs))

for i in run_folders:
    print(i)

# for p in percentiles:
#     for s in seeds:
#         filename = f"{base}{p}"