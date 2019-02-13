from nltk import TweetTokenizer, re
import numpy as np

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


def clean_tweet(tweet):
    # Function applied to every tweet
    t1 = re.sub(r"http\S+", "", tweet).lower()
    t2 = tokenizer.tokenize(t1)
    return t2


def preprocess(X):
    cleaned = np.array(list(map(clean_tweet, X)))
    return cleaned
