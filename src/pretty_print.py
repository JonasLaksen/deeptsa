import json
from glob import glob

import pandas


def print_for_master_thesis_compact(path, group_fields, sort_by=['sum_ranks'], fields_to_show=['features'], show_model=True):
    subdirectories = glob(path)

    subexperiments = []
    for subdirectory in subdirectories:
        meta_path = f'{subdirectory}meta.json'
        with open(meta_path, 'r') as json_file:
            meta = json.load(json_file)

        evaluation_path = f'{subdirectory}evaluation.json'
        with open(evaluation_path, 'r') as json_file:
            evaluation = json.load(json_file)

        subexperiments.append({'seed': meta['seed'],
                               'layer': meta['layer-sizes'],
                               'dropout': meta['dropout'],
                               'loss': meta['loss'],
                               'features': meta['features'],
                               'model-type': meta['model-type'],
                               'mape': evaluation['validation'][0]['MAPE'],
                               'mae': evaluation['validation'][0]['MAE'],
                               'mse': evaluation['validation'][0]['MSE'],
                               'da': evaluation['validation'][0]['DA'] * 100,
                               })

    df = pandas.DataFrame(subexperiments)
    metrics = [('mape', '\%'), ('mae', ''), ('mse', ''), ('da', '\%')]
    for (metric, unit) in metrics:
        df[f'mean_{metric}'] = df.groupby(group_fields)[metric].transform('mean')
        df[f'mean_{metric}_rank'] = df[f'mean_{metric}'].rank(method='dense', ascending=metric != 'da')
        df[metric] = df[metric].transform(lambda x: f'{x:.4}' if x < 1000 else int(x))
        df[f'mean_{metric}'] = df[f'mean_{metric}'].transform(lambda x: f'{x:.4}' if x < 1000 else int(x))

    df['sum_ranks'] = df[[f'mean_{metric}_rank' for (metric, unit) in metrics]].sum(axis=1)
    df = df.sort_values(sort_by + group_fields + ['seed'])
    print(sort_by + group_fields + ['seed'])
    list_of_rows = df.to_dict('records')
    list_of_groups = zip(*(iter(list_of_rows),) * 3)

    backslashes = '\\\\'
    newline = '\n\t\t'
    midrule = '\midrule'
    newline2 = '\\newline'
    newline_after_result = newline2
    for group in list_of_groups:
        if(group[0]["model-type"] == 'BidirLSTM'):
            model_type = 'Bidirectional'
        elif len(group[0]["layer"].split(',')) == 1:
            model_type = 'Vanilla'
        else:
            model_type = 'Stacked'

        model = f'{model_type}{newline2}{group[0]["layer"]} &'
        fields = ' & '.join([str( group[0][field] ) for field in fields_to_show])
        output = f'''{model if show_model else ''}{fields} & {' & '.join([f'{group[0][f"mean_{metric}"]}{unit}{newline_after_result}(#{int(group[0][f"mean_{metric}_rank"])}){newline_after_result}' for (metric, unit) in metrics])} \\\\
        \hline '''

        print(output.replace("_", "\\_").replace('#', '\#'))
        # Sum rank & {int(group[0]['sum_ranks'])} \\\\
        # {# Mean Rank & {' '.join([f'{int(group[0][f"mean_{metric}_rank"])} &' for (metric, unit) in metrics])} \\\\}



def print_for_master_thesis(path, group_fields, sort_by=['sum_ranks']):
    subdirectories = glob(path)

    subexperiments = []
    for subdirectory in subdirectories:
        meta_path = f'{subdirectory}meta.json'
        with open(meta_path, 'r') as json_file:
            meta = json.load(json_file)

        evaluation_path = f'{subdirectory}evaluation.json'
        with open(evaluation_path, 'r') as json_file:
            evaluation = json.load(json_file)

        subexperiments.append({'seed': meta['seed'],
                               'layer': meta['layer-sizes'],
                               'dropout': meta['dropout'],
                               'loss': meta['loss'],
                               'features': meta['features'],
                               'model-type': meta['model-type'],
                               'mape': evaluation['validation'][0]['MAPE'],
                               'mae': evaluation['validation'][0]['MAE'],
                               'mse': evaluation['validation'][0]['MSE'],
                               'da': evaluation['validation'][0]['DA'] * 100,
                               })

    df = pandas.DataFrame(subexperiments)
    metrics = [('mape', '\%'), ('mae', ''), ('mse', ''), ('da', '\%')]
    for (metric, unit) in metrics:
        df[f'mean_{metric}'] = df.groupby(group_fields)[metric].transform('mean')
        df[f'mean_{metric}_rank'] = df[f'mean_{metric}'].rank(method='dense', ascending=metric != 'da')
        df[metric] = df[metric].transform(lambda x: f'{x:.4}' if x < 1000 else int(x))
        df[f'mean_{metric}'] = df[f'mean_{metric}'].transform(lambda x: f'{x:.4}' if x < 1000 else int(x))

    df['sum_ranks'] = df[[f'mean_{metric}_rank' for (metric, unit) in metrics]].sum(axis=1)
    df = df.sort_values(sort_by + group_fields + ['seed'])
    list_of_rows = df.to_dict('records')
    list_of_groups = zip(*(iter(list_of_rows),) * 3)

    backslashes = '\\\\'
    newline = '\n\t\t'
    infinity = 'inf'
    for group in list_of_groups:
        output = f'''{', '.join([str(group[0][field]) for field in group_fields])} \\\\
        {newline.join([f"{group[i]['seed']} & {' '.join([f'{infinity if float( group[i][metric] ) > 1000000 else group[i][metric]}{unit} &' for (metric, unit) in metrics])} {backslashes}"
                       for i in range(3)])}
        \midrule
        Mean & {' '.join([f'{"inf" if float(group[0][f"mean_{metric}"]) > 1000000 else group[0][f"mean_{metric}"]}{unit} &' for (metric, unit) in metrics])} \\\\
        Mean Rank & {' '.join([f'{int(group[0][f"mean_{metric}_rank"])} &' for (metric, unit) in metrics])} \\\\
        Sum rank & {int(group[0]['sum_ranks'])} \\\\
        \midrule '''

        print(output.replace("_", "\\_"))

def pretty_print_evaluate(arr1, arr2):
    newline = '\\newline'
    return " \\\\ \\hline \n".join([
        f'{ev1["stock"]} & Baseline{newline} LSTM & {ev1["MAPE"]:.4}\%{newline}{ev2["MAPE"]:.4}\% & {ev1["MAE"]:.4}{newline}{ev2["MAE"]:.4} & {ev1["MSE"]:.4}{newline}{ev2["MSE"]:.4} & {ev1["DA"] * 100:.4}\%{newline}{ev2["DA"] * 100:.4}\%'
        for ev1, ev2 in zip(arr1, arr2)])


