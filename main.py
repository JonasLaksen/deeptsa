import pandas as pd
from preprocessing import preprocess

data = pd.read_csv("data.csv", quotechar="~").as_matrix()

# Choose the third column of the csv file - this is the tweet in this dataset
# Only take the first 20 tweets for now
X = data[:,2][0:20]
preprocessed = preprocess(X)
pass
