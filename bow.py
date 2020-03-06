from sklearn.feature_extraction.text import CountVectorizer


class BoW:
    def __init__(self, ngram_range, max_features, stop_words=None):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words

    def fit(self, data):
        v = CountVectorizer(ngram_range=self.ngram_range, max_features=self.max_features, stop_words=self.stop_words,
                            strip_accents=None)
        return v.fit_transform(data)
