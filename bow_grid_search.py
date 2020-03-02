import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from bow import BoW
from cross_val import cv_fold_generator

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'


def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [tuple(headers)]
        for row in csv_reader:
            row_data = [(row[header]) for header in headers]
            rows.append(tuple(row_data))

        return rows


def extract_questionmark_features(data, header_index):
    features = []
    for i, row in enumerate(data):
        # Skip the headers
        if i == 0:
            continue

        has_questionmark = '?' in row[header_index]
        features.append(has_questionmark)
    return sparse.csr_matrix(np.array([features]).T)


def logistic_regression(data, target, folds):
    clf = LogisticRegression(multi_class="ovr")
    return kfold_cross(clf, data, target, folds)


def kfold_cross(clf, data, target, folds=10):
    cv = cross_validate(clf, data, target, cv=folds, scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
    accuracy  = np.mean(cv['test_accuracy'])
    f1 = np.mean(cv['test_f1_macro'])
    recall = np.mean(cv['test_recall_macro'])
    precision = np.mean(cv['test_precision_macro'])
    return [accuracy, f1, recall, precision]


def add_question_mark_feature(data, questionmark_features):
    return sparse.hstack((data, questionmark_features))


def grid_search_bow(data, target, ids, questionmark_features, folds=10, do_custom_folds=True):
    ngram_range = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]
    max_features = range(80, 95)
    custom_folds = cv_fold_generator(ids, folds)
    res = []
    count = 0
    for i in ngram_range:
        for j in max_features:
            print(count / (len(max_features) * len(ngram_range)))
            count += 1
            bow = BoW(ngram_range=i, max_features=j, stop_words=None)
            x = bow.fit(data)
            combined = add_question_mark_feature(x, questionmark_features)
            if do_custom_folds:
                res.append([logistic_regression(combined, target, custom_folds), i, j])
            else:
                res.append([logistic_regression(combined, target, folds), i, j])

    print(sorted(res, key=lambda x: x[0], reverse=True))


def split_data(data):
    y = list(map(lambda row: row[1], data))
    x = list(map(lambda row: row[0], data))
    ids = list(map(lambda row: row[2], data))
    return x[1:], y[1:], ids[1:]


if __name__ == "__main__":
    data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance', 'claimId'])

    print(f'Headers: {data[0]}')
    print(f'First row: {data[1]}')

    headers = ['articleHeadline', 'articleHeadlineStance']
    questionmark_features = extract_questionmark_features(data, headers.index('articleHeadline'))

    data = split_data(data)

    x = data[0]
    y = data[1]
    ids = data[2]


    grid_search_bow(x, y, ids, questionmark_features)
