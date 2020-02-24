from sklearn.feature_extraction.text import CountVectorizer

class BoW:
    def __init__(self, ngram_range, max_features):
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, data):
        v = CountVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        return v.fit_transform(data)