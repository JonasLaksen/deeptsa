import nltk
import numpy as np
from nltk import TweetTokenizer, re, collections
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download('stopwords')


tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def clean_tweet(tweet, ngrams_count=3):
    # Function applied to every tweet
    t1 = re.sub(r"(http\S+|[^a-zA-Z0-9_! ])", "", tweet).lower()
    t2 = [token for token in tokenizer.tokenize(t1) if token not in stopwords.words('english')]
    t3 = [list(ngrams(t2, i)) for i in range(1, ngrams_count + 1)]
    return t3


def preprocess(X):
    cleaned = np.array(list(map(clean_tweet, X)))
    all_grams = [[item for sublist in x for item in sublist] for x in zip(*cleaned)]

    counts = [collections.Counter(gram).most_common(500) for gram in all_grams]
    flattened_counts = [item for sublist in counts for item in sublist]
    dict = {}
    for i in range(len(flattened_counts)):
        dict[flattened_counts[i][0]] = i

    cleaned = [[item for sublist in x for item in sublist] for x in cleaned]
    cleaned = list(map(lambda x: list(map(lambda y: dict.get(y, -1), x)), cleaned)) #Convert to list of ids
    cleaned = list(map( lambda x: x + [-2] * (100-len(x)),cleaned)) #Add padding to end
    cleaned = np.asarray(list(map(lambda x: np.asarray(x), cleaned)))
    return cleaned
