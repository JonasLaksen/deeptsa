import os
from shutil import copyfile

from numpy import sort

results_folder = "./server_results/tren_pa_individuelle_aksjer.py/"
folders = os.listdir(results_folder)
sorted = sort(folders)[-12:]
# sorted = sort(folders)[1:13]
# print(sorted[-12:])

for i,folder in enumerate(sorted):
    print(i)
    print(folder)
    copyfile(f'{results_folder}{folder}/aksje-0-AAPL/plot-val.png', f'./plots/plot-val-{i+1}.png')
