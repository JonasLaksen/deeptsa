from functools import reduce

import pandas
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

files = list(map(lambda i: pandas.read_csv(f'feature_{i}_128_stacked_context', names=range(25), header=None), range(3)))
columns = files[0].columns[0:4]
df = files[0][columns].append(files[1][columns]).append(files[1][columns])
df = reduce(lambda left, right: pandas.merge(left, right, left_index=True, right_index=True),
            (map(lambda x: x[columns], files)))
df = pandas.merge(df, files[0][files[0].columns[4:]], left_index=True, right_index=True)
mape_cols = ['0_x', '0_y', 0]
mae_cols = ['1_x', '1_y', 1]
mse_cols = ['2_x', '2_y', 2]
da_cols = ['3_x', '3_y', 3]

for i in zip(['avg_mape', 'avg_mae', 'avg_mse', 'avg_da'], [mape_cols, mae_cols, mse_cols, da_cols], [True, True, True, False]):
    df[i[0]] = df[i[1]].mean(axis=1)
    df[f'{i[0]}_rank'] = df[i[0]].rank(ascending=i[2]).astype(int)

df['sum_ranks'] = df['avg_da_rank'] + df['avg_mape_rank'] + df['avg_mse_rank'] + df['avg_mae_rank']
df = df.round(4)
df = df.sort_values(by=['sum_ranks', 'avg_mape_rank', 'avg_mae_rank', 'avg_mse_rank', 'avg_da_rank'])

for i, row in df.iterrows():
    row = row.dropna()
    print(f'''{",".join(map(str, row[12:-9]))} \\\\
    0 & {" & ".join(map(str, row[0:4]))} \\\\
    1 & {" & ".join(map(str, row[4:8]))} \\\\
    2 & {" & ".join(map(str, row[8:12]))} \\\\
    \midrule
    Mean & {" & ".join(map(lambda x: str(row[x]), ["avg_mape", "avg_mae", "avg_mse", "avg_da"]))} & \\\\
    Mean Rank & {" & ".join(
        map(lambda x: str(row[x]), ["avg_mape_rank", "avg_mae_rank", "avg_mse_rank", "avg_da_rank"]))} & \\\\
    Sum rank & {row["sum_ranks"]} \\\\
    \midrule''')
