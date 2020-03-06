import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def extract_questionmark_features(data, header_index):
    features = []
    for i, row in enumerate(data):
        # Skip the headers
        if i == 0:
            continue

        has_questionmark = '?' in row[header_index]
        features.append(has_questionmark)
    return sparse.csr_matrix(np.array([features]).T)


def logistic_regression(data, target):
    clf = LogisticRegression(multi_class="ovr")
    return kfold_cross(clf, data, target)


def kfold_cross(clf, data, target):
    cv = cross_validate(clf, data, target, cv=10, scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
    accuracy  = np.mean(cv['test_accuracy'])
    f1 = np.mean(cv['test_f1_macro'])
    recall = np.mean(cv['test_recall_macro'])
    precision = np.mean(cv['test_precision_macro'])
    return [accuracy, f1, recall, precision]


def add_question_mark_feature(data, questionmark_features):
    return sparse.hstack((data, questionmark_features))


def grid_search_bow(data, target, questionmark_features):
    ngram_range = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    max_features = [5,10,100,200,300,400,500,600,700,800,900,1000]
    res = []
    count = 0

    for i in ngram_range:
        for j in max_features:
            count += 1
            bow = BoW(ngram_range=i, max_features=j)

            d = bow.fit(data)
            # combined = add_question_mark_feature(d, questionmark_features)

            r = logistic_regression(d, target)
            res.append([r, i, j])

    plot_grid_search_bow(res, ngram_range, max_features)

    print(sorted(res, key=lambda x: x[0], reverse=True))


def hyperparam_bow(data, target, questionmark_features):
    max_features = range(80, 120)
    res = []
    count = 0

    for i in max_features:
        count += 1
        bow = BoW(ngram_range=(1, 2), max_features=i)

        d = bow.fit(data)
        # combined = add_question_mark_feature(d, questionmark_features)

        r = logistic_regression(d, target)
        res.append([r, i])

    plot_hyperparam_bow(res, max_features)

    print(sorted(res, key=lambda x: x[0], reverse=True))


def plot_hyperparam_bow(results, max_features):
    x = []
    accuracy = []
    f1 = []
    recall = []
    precision = []
    index = 0
    for i in max_features:
        x.append(i)
        accuracy.append(results[index][0][0])
        f1.append(results[index][0][1])
        recall.append(results[index][0][2])
        precision.append(results[index][0][3])
        index += 1

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, accuracy, label='Accuracy')
    ax.plot(x, f1, label='F1-Score')
    ax.plot(x, recall, label='Recall')
    ax.plot(x, precision, label='Precision ')
    ax.legend()
    ax.set_xlabel("Max features")
    fig.show()


def plot_grid_search_bow(results, ngram_range, max_features):
    x = []
    y = []
    accuracy = []
    index = 0
    for i in range(0, len(ngram_range)):
        for j in max_features:
            x.append(i)
            y.append(j)
            accuracy.append(results[index][0][0])
            index += 1

    fig = plt.figure()
    labels = ['(1, 1)', '(1, 2)', '(1, 3)', '(2, 2)', '(2, 3)', '(3, 3)']
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(labels)
    ax.scatter(x, y, accuracy)
    ax.set_xlabel("N-gram range")
    ax.set_ylabel("Max features")
    ax.set_zlabel("Accuracy")
    fig.show()


def split_data(data):
    y = list(map(lambda row: row[1], data))
    x = list(map(lambda row: row[0], data))
    return x[1:], y[1:]


data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance'])

print(f'Headers: {data[0]}')
print(f'First row: {data[1]}')

headers = ['articleHeadline', 'articleHeadlineStance']
questionmark_features = extract_questionmark_features(data, headers.index('articleHeadline'))

data = split_data(data)

x = data[0]
y = data[1]

grid_search_bow(x, y, questionmark_features)
hyperparam_bow(x, y, questionmark_features)
