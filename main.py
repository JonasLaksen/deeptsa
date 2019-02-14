import pandas as pd
from sklearn.model_selection import train_test_split

from model import create_model, evaluate
from preprocessing import preprocess

data = pd.read_csv("data.csv", quotechar="~").as_matrix()

# Choose the third column of the csv file - this is the tweet in this dataset
# Only take the first 20 tweets for now
X = data[:,2][0:10000]
y = data[:,1][0:10000]
X = preprocess(X)
x_train,x_test,y_train, y_test = train_test_split(X,y, test_size=.2)
model = create_model(x_train, y_train)
evaluate(model, x_test, y_test)
