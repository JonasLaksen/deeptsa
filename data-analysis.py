import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(data, x_name, y_name):
    sns.jointplot(x=x_name, y=y_name, data=data)


if __name__ == '__main__':
    data = pd.read_csv('dataset.csv')
    data = data[data['stock'] == 'FB']
    plot_correlation(data, 'negative', 'next_price')
    plt.show()