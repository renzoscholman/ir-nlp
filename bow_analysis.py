import csv

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def extract_questionmark_features(data, index):
    features = []
    for i, row in enumerate(data[index]):
        has_questionmark = '?' in row
        features.append(has_questionmark)
    return sparse.csr_matrix(np.array([features]).T)


def logistic_regression_var(data, target, folds, regularization='l2', max_iter=10000):
    clf = LogisticRegression(multi_class="ovr", penalty=regularization, max_iter=max_iter)
    return kfold_cross_var(clf, data, target, folds)


def logistic_regression(data, target, folds, regularization='l2', max_iter=10000):
    clf = LogisticRegression(multi_class="ovr", penalty=regularization, max_iter=max_iter)
    return kfold_cross(clf, data, target, folds)


def svm_rbf(data, target, folds, regularization='l2', max_iter=-1):
    clf = SVC(C=0.5, kernel='rbf', gamma='scale', max_iter=max_iter)
    return kfold_cross(clf, data, target, folds)


def kfold_cross_var(clf, data, target, folds=10):
    cv = cross_validate(clf, data, target, cv=folds,
                        scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
    accuracy = np.mean(cv['test_accuracy'])
    f1 = np.mean(cv['test_f1_macro'])
    recall = np.mean(cv['test_recall_macro'])
    precision = np.mean(cv['test_precision_macro'])
    accuracy_var = np.var(cv['test_accuracy'])
    f1_var = np.var(cv['test_f1_macro'])
    recall_var = np.var(cv['test_recall_macro'])
    precision_var = np.var(cv['test_precision_macro'])
    return [accuracy, f1, recall, precision, accuracy_var, f1_var, recall_var, precision_var]


def kfold_cross(clf, data, target, folds=10):
    cv = cross_validate(clf, data, target, cv=folds,
                        scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
    accuracy = np.mean(cv['test_accuracy'])
    f1 = np.mean(cv['test_f1_macro'])
    recall = np.mean(cv['test_recall_macro'])
    precision = np.mean(cv['test_precision_macro'])
    return [accuracy, f1, recall, precision]


def add_question_mark_feature(data, questionmark_features):
    return sparse.hstack((data, questionmark_features))


def plot_2D_data(data, target):
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(data)
    # Group data by target 'observing', 'for' and 'against' in a dict
    targets = {}
    length = reduced.shape[0]
    for ind in range(0, length):
        targeted = target[ind]
        PC = reduced[ind]
        if targeted not in targets:
            targets[targeted] = ([PC[0]], [PC[1]])
        else:
            targets[targeted] = (targets[targeted][0] + [PC[0]], targets[targeted][1] + [PC[1]])
    targets_list = list(targets.items())
    plt.scatter(targets_list[0][1][0], targets_list[0][1][1], label='observing', alpha=0.7)
    plt.scatter(targets_list[1][1][0], targets_list[1][1][1], label='for', alpha=0.7)
    plt.scatter(targets_list[2][1][0], targets_list[2][1][1], label='against', alpha=0.7)
    plt.legend()
    plt.show()


def grid_search_bow(data, target):
    ngram_range = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    max_features = [5,10,100,200,300,400,500,600,700,800,900,1000]
    res = []
    count = 0

    for i in ngram_range:
        for j in max_features:
            count += 1
            bow = BoW(ngram_range=i, max_features=j)

            d = bow.fit(data)

            r = logistic_regression(d, target, 10)
            res.append([r, i, j])

    plot_grid_search_bow(res, ngram_range, max_features)

    print(sorted(res, key=lambda x: x[0], reverse=True))


def grid_search_bow_custom_fold(data_h, target, ids, questionmark_features, folds=10, do_custom_folds=True):
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
            x = bow.fit(data_h)
            if i == (1, 2) and j == 90:
                plot_2D_data(x, target)

            # print(reduced)
            # combined2 = np.column_stack((reduced, questionmark_features.toarray()))
            combined = add_question_mark_feature(x, questionmark_features)
            # print(combined.toarray()[0])
            regularization = 'l2'
            if do_custom_folds:
                res.append([logistic_regression(combined, target, custom_folds, regularization), i, j])
            else:
                res.append([logistic_regression(combined, target, folds, regularization), i, j])

    print(sorted(res, key=lambda x: x[0], reverse=True))


def hyperparam_bow(data, target):
    max_features = range(80, 120)
    res = []
    count = 0

    for i in max_features:
        count += 1
        bow = BoW(ngram_range=(1, 2), max_features=i)

        d = bow.fit(data)

        r = logistic_regression(d, target, 10)
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
    ids = list(map(lambda row: row[2], data))
    return x[1:], y[1:], ids[1:]


if __name__ == "__main__":
    data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance', 'claimId'])

    print(f'Headers: {data[0]}')
    print(f'First row: {data[1]}')

    headers = ['articleHeadline', 'articleHeadlineStance']
    data_split = split_data(data)

    x = data_split[0]
    y = data_split[1]
    ids = data_split[2]

    questionmark_features = extract_questionmark_features(data_split, headers.index('articleHeadline'))
    print(list(questionmark_features.toarray()))

    grid_search_bow(x, y)
    hyperparam_bow(x, y)
    grid_search_bow_custom_fold(x, y, ids, questionmark_features)
