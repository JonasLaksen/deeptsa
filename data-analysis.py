import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(data, x_name, y_name):
    sns.jointplot(x=x_name, y=y_name, data=data)


if __name__ == '__main__':
    data = pd.read_csv('dataset_v2.csv')
    data = data[data['stock'] == 'TWTR']
    plot_correlation(data, 'trendscore', 'price')
    plt.show()