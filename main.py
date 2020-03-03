import csv

from sklearn.decomposition import TruncatedSVD

from bow import BoW
from bow_grid_search import split_data, logistic_regression, add_question_mark_feature, svm_rbf, plot_2D_data, \
    logistic_regression_var
from cross_val import cv_fold_generator
from rootdist import get_rootdist_matrix, crossval_rootdist
from alignment_score import get_ppdb_alignment_feature
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'


def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [tuple(headers)]
        for row in csv_reader:
            row_data = [(row[header]) for header in headers]
            rows.append(tuple(row_data))

        return rows


def extract_column(data, header_index):
    # Extract a column without the header
    return list(map(lambda row: row[header_index], data[1:]))


def extract_questionmark_features(data, index):
    features = []
    for i, row in enumerate(data[index]):
        has_questionmark = '?' in row
        features.append(has_questionmark)
    return sparse.csr_matrix(np.array([features]).T)


def variance_plot_cv(claim_ids, target, rootdist_matrix, tf_matrix, questionmark, fold_range=range(2, 30), regularization='l2'):
    rootdist_feature = sparse.csr_matrix(rootdist_matrix)
    questionmark_feature = questionmark
    ppdb_alignment_feature = sparse.csr_matrix(get_ppdb_alignment_feature())

    combined_all = sparse.hstack((
        rootdist_feature,
        questionmark_feature,
        ppdb_alignment_feature,
        tf_matrix
    ))
    plot_2D_data(combined_all, target)

    results = []

    for n_folds in fold_range:
        print(n_folds)
        custom_folds = cv_fold_generator(claim_ids, n_folds)
        results.append(logistic_regression_var(combined_all, target, custom_folds, regularization, 10000))

    results_arr = np.array(results)
    plt.plot(fold_range, results_arr[:, 0], label='Accuracy')  # Plot accuracy
    plt.plot(fold_range, results_arr[:, 1], label='F1-Score')  # Plot F1-score
    plt.plot(fold_range, results_arr[:, 2], label='Recall')  # Plot recall
    plt.plot(fold_range, results_arr[:, 3], label='Precision')  # Plot precision
    plt.legend()
    plt.show()
    plt.plot(fold_range, results_arr[:, 4], label='Accuracy var.')  # Plot accuracy variance
    plt.plot(fold_range, results_arr[:, 5], label='F1-Score var.')  # Plot F1-score variance
    plt.plot(fold_range, results_arr[:, 6], label='Recall var.')  # Plot recall variance
    plt.plot(fold_range, results_arr[:, 7], label='Precision var.')  # Plot precision variance
    plt.legend()
    plt.show()


def questionmark_only(claim_ids, target, questionmark, folds=5, do_custom_folds=True, regularization='l2'):
    custom_folds = cv_fold_generator(claim_ids, folds)
    print('accuracy', 'f1_macro', 'recall_macro', 'precision_macro')
    if do_custom_folds:
        print(logistic_regression(questionmark, target, custom_folds, regularization, 1000000))
    else:
        print(logistic_regression(questionmark, target, folds, regularization, 1000000))


def bow_rootdist(claim_ids, target, rootdist_matrix, tf_matrix, folds=5, do_custom_folds=True, regularization='l2'):
    custom_folds = cv_fold_generator(claim_ids, folds)
    data_sparse = sparse.csr_matrix(rootdist_matrix)
    combined_all = sparse.hstack((data_sparse, tf_matrix))
    plot_2D_data(combined_all, target)

    print('accuracy', 'f1_macro', 'recall_macro', 'precision_macro')
    if do_custom_folds:
        print(logistic_regression(combined_all, target, custom_folds, regularization, 1000000))
    else:
        print(logistic_regression(combined_all, target, folds, regularization, 1000000))


def combined_crossval(claim_ids, target, rootdist_matrix, tf_matrix, questionmark, folds=5, do_custom_folds=True,
                      regularization='l2'):
    custom_folds = cv_fold_generator(claim_ids, folds)
    rootdist_feature = sparse.csr_matrix(rootdist_matrix)
    questionmark_feature = questionmark
    ppdb_alignment_feature = sparse.csr_matrix(get_ppdb_alignment_feature())

    combined_all = sparse.hstack((
        rootdist_feature,
        questionmark_feature,
        ppdb_alignment_feature,
        tf_matrix
    ))
    plot_2D_data(combined_all, target)

    print('accuracy', 'f1_macro', 'recall_macro', 'precision_macro')
    if do_custom_folds:
        print(logistic_regression(combined_all, target, custom_folds, regularization, 1000000))
    else:
        print(logistic_regression(combined_all, target, folds, regularization, 1000000))


if __name__ == "__main__":
    data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance', 'claimId'])

    print(f'Headers: {data[0]}')
    print(f'First row: {data[1]}')

    headers = ['articleHeadline', 'articleHeadlineStance']

    data_split = split_data(data)
    x = data_split[0]
    y = data_split[1]
    ids = data_split[2]

    rootdist = get_rootdist_matrix()
    questionmark_features = extract_questionmark_features(data_split, headers.index('articleHeadline'))
    bow = BoW(ngram_range=(1, 2), max_features=90, stop_words=None)
    tf = bow.fit(x)

    variance_plot_cv(ids, y, rootdist, tf, questionmark_features, range(2, 10))
    print("Questionmark only")
    questionmark_only(ids, y, questionmark_features, 10, True)
    print("Rootdist without questionmark")
    crossval_rootdist(rootdist, y, ids, None)
    print("Rootdist with questionmark")
    crossval_rootdist(rootdist, y, ids, questionmark_features)
    print("BoW with Rootdist")
    bow_rootdist(ids, y, rootdist, tf, 10, True)
    print("All features")
    combined_crossval(ids, y, rootdist, tf, questionmark_features, 10, False)
