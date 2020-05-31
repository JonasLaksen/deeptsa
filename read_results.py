import json
import os

from numpy import sort

results_folder = "./server_results/tren_pa_individuelle_aksjer.py/"
folders = os.listdir(results_folder)
sorted = sort(folders)[-12:]
print(sorted)

for s in sorted:
    with open(f'{results_folder}{s}/aksje-0-AAPL/evaluation.json') as json_file:
        lol = json.load(json_file)
        a,b,c,d = lol['validation'].values()
        print(f'{a} & {b} & {c} & {d}')
