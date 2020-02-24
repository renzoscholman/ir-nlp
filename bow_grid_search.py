import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_validate
import numpy as np

from bow import BoW

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'


def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [tuple(headers)]
        for row in csv_reader:
            row_data = [(row[header]) for header in headers]
            rows.append(tuple(row_data))

        return rows


def logistic_regression(data, target):
    clf = LogisticRegression()
    return kfold_cross(clf, data, target)


def kfold_cross(clf, data, target):
    return np.mean(cross_val_score(clf, data, target, cv=10, scoring='f1_macro'))


def grid_search_bow(data, target):
    ngram_range = [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3)]
    max_features = [100,200,300,400,500,600,700,800]
    res = []
    for i in ngram_range:
        for j in max_features:
            bow = BoW(ngram_range=i, max_features=j)
            x = bow.fit(data)
            res.append([logistic_regression(x, target), i, j])

    print(sorted(res, reverse=True))


def split_data(data):
    y = list(map(lambda row: row[1], data))
    x = list(map(lambda row: row[0], data))
    return x[1:], y[1:]


data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance'])

print(f'Headers: {data[0]}')
print(f'First row: {data[1]}')

data = split_data(data)
x = data[0]
y = data[1]

grid_search_bow(x, y)




