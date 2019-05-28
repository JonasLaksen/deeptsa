import csv
from functools import reduce

import pandas

files = list(map(lambda i: pandas.read_csv(f'hyperparameter_search/{i}', header=None), range(3)))
test = list(map(lambda x: x[x.columns[0:4]],files))

columns = files[0].columns[0:4]
test = files[0][columns].append(files[1][columns]).append(files[1][columns])
test = reduce(lambda left,right: pandas.merge(left,right, left_index=True, right_index=True), (map(lambda x: x[columns], files)))
lol = pandas.merge(test, files[0][files[0].columns[4:]], left_index=True, right_index=True)
mape_cols = ['0_x', '0_y', 0]
mae_cols = ['1_x', '1_y', 1]
mse_cols = ['2_x', '2_y', 2]
da_cols = ['3_x', '3_y', 3]

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


lol['avg_mape'] = lol[mape_cols].mean(axis=1)
lol['avg_mape_rank'] = lol['avg_mape'].rank(ascending=True).astype(int)
lol['avg_mae'] = lol[mae_cols].mean(axis=1)
lol['avg_mae_rank'] = lol['avg_mae'].rank(ascending=True).astype(int)
lol['avg_mse'] = lol[mse_cols].mean(axis=1)
lol['avg_mse_rank'] = lol['avg_mse'].rank(ascending=True).astype(int)
lol['avg_da'] = lol[da_cols].mean(axis=1)
lol['avg_da_rank'] = lol['avg_da'].rank(ascending=False).astype(int)
lol['sum_ranks'] = lol['avg_da_rank'] + lol['avg_mape_rank'] + lol['avg_mse_rank'] + lol['avg_mae_rank']
lol = lol.round(2)
lol = lol.sort_values(by=['sum_ranks', 'avg_mape_rank', 'avg_mae_rank', 'avg_mse_rank', 'avg_da_rank'])

for i, row in lol.iterrows():
    line1 = f'{map(lambda x: f"{x} & ", row[4])} & 0 & {row["0_x"]} & {row["1_x"]} & {row["2_x"]} & {row["3_x"]} \\\\'
    line2 = f'{"& "*len(row[4])} & 1 & {row["0_y"]} & {row["1_y"]} & {row["2_y"]} & {row["3_y"]} \\\\'
    line3 = f'{"& "*len(row[4])} & 2 & {row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\'
    line4 = '\midrule'
    line5 = f'{"& "*len(row[4])} & Mean & {row["avg_mape"]} & {row["avg_mae"]} & {row["avg_mse"]} & {row["avg_da"]} \\\\'
    line6 = f'{"& "*len(row[4])} & Mean Rank & {row["avg_mape_rank"]} & {row["avg_mae_rank"]} & {row["avg_mse_rank"]} & {row["avg_da_rank"]} \\\\'
    line7 = f'{"& "*len(row[4])} & Sum rank & {row["sum_ranks"]} \\\\'
    line8 = '\midrule'
    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(line5)
    print(line6)
    print(line7)
    print(line8)
print(lol)
