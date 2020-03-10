import numpy as np
import functools
import operator
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC


# Custom fold generator preventing information leakage by keeping claim ids in their own folds
def cv_fold_generator(data, n_folds=10):
    claim_dict = {}

    # Group data by claim ID in a dict
    for i in range(0, len(data)):
        claim_id = data[i]
        index = i
        if claim_id not in claim_dict:
            claim_dict[claim_id] = [index]
        else:
            claim_dict[claim_id].append(index)

    # Convert dict to list for custom order
    claim_list = list(claim_dict.items())

    # Shuffle list for randomization, use seed for reproducibility
    random.seed(1)
    random.shuffle(claim_list)

    # Placeholder for folds
    folds = []

    # Counter for iterating through randomized claim list
    counter = 0

    # Desired size of each fold
    fold_size = len(data) / n_folds

    # Separate list into folds using greedy approach, it stops filling a fold after it is equal or passed the fold size
    for i in range(n_folds):
        fold = []
        # First n_folds - 1 folds, fill fold with at least fold_size data rows
        if i < n_folds - 1:
            while (len(fold) + len(claim_list[counter][1])) < fold_size:
                fold = fold + (claim_list[counter][1])
                counter = counter + 1
        # Last fold, fills it with remaining data rows
        elif i == n_folds - 1:
            while counter < len(claim_list):
                fold = fold + (claim_list[counter][1])
                counter = counter + 1
        # This should never execute
        else:
            raise Exception("Cross validation fold generation error")
        folds.append(fold)

    # Folds containing train and test indices
    cv = []

    # Construct the train and test indices for each fold
    for i in range(0, len(folds)):
        # Fold i as test fold
        test_fold = folds[i]
        # Remaining folds as train folds
        train_folds = functools.reduce(operator.iconcat, folds[:i] + folds[i + 1:], [])  # get all folds except fold i
        # Append folds to cv
        cv.append((train_folds, test_fold))

    return cv

def split_data(data):
    y = list(map(lambda row: row[1], data))
    x = list(map(lambda row: row[0], data))
    ids = list(map(lambda row: row[2], data))
    return x[1:], y[1:], ids[1:]

def logistic_regression_var(data, target, folds, regularization='l2', max_iter=10000):
    clf = LogisticRegression(multi_class="ovr", penalty=regularization, max_iter=max_iter)
    return kfold_cross_var(clf, data, target, folds)


def logistic_regression(data, target, folds, multiclass='ovr', regularization='l2', max_iter=10000):
    solver = 'saga' if regularization == 'l1' else 'lbfgs'
    clf = LogisticRegression(multi_class=multiclass, penalty=regularization, solver=solver, max_iter=max_iter)
    return kfold_cross(clf, data, target, folds)

def naive_bayes(data, target, folds):
    return kfold_cross(GaussianNB(), data, target, folds)


def svm_rbf(data, target, folds, regularization='l2', max_iter=-1, C=0.5, gamma='scale'):
    clf = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=max_iter)
    return kfold_cross(clf, data, target, folds)


def kfold_cross_var(clf, data, target, folds=10):
    cv = cross_validate(clf, data, target, cv=folds, scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
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
    cv = cross_validate(clf, data, target, cv=folds, scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'])
    accuracy = np.mean(cv['test_accuracy'])
    f1 = np.mean(cv['test_f1_macro'])
    recall = np.mean(cv['test_recall_macro'])
    precision = np.mean(cv['test_precision_macro'])
    return [accuracy, f1, recall, precision]