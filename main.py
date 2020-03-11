import csv

from bow import BoW
from bow_analysis import plot_2D_data
from classification import logistic_regression, logistic_regression_var, svm_rbf, naive_bayes, split_data, cv_fold_generator
from alignment_score import get_ppdb_alignment_feature
from rootdist import get_rootdist_matrix, crossval_grid_search
from scipy import sparse
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.linear_model import LogisticRegression

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'


def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [tuple(headers)]
        for row in csv_reader:
            row_data = [(row[header]) for header in headers]
            rows.append(tuple(row_data))

        return rows

def print_data_distribution(labels):
    all_observing = len(list(filter(lambda x: x == 'observing', labels)))/len(labels) * 100
    all_for = len(list(filter(lambda x: x == 'for', labels)))/len(labels) * 100
    all_against = len(list(filter(lambda x: x == 'against', labels)))/len(labels) * 100
    print('Distribution')
    print(f'Observing: {all_observing}% For: {all_for}% Against: {all_against}%')

def show_confusion_matrix(y_test, class_names, rootdist, questionmark_features):
    # Copy pasted classification code to avoid a huge code restructure
    rootdist_feature = sparse.csr_matrix(rootdist)
    ppdb_alignment_feature = sparse.csr_matrix(get_ppdb_alignment_feature())
    X_test = sparse.hstack((rootdist_feature, questionmark_features, ppdb_alignment_feature, tf))
    clf = LogisticRegression(multi_class="ovr", penalty='l2', max_iter=10000).fit(X_test, y_test)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()

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


def ppdb_only(claim_ids, target, ppdb, folds=5, do_custom_folds=True, regularization='l2'):
    custom_folds = cv_fold_generator(claim_ids, folds)
    print('accuracy', 'f1_macro', 'recall_macro', 'precision_macro')
    if do_custom_folds:
        print(logistic_regression(ppdb, target, custom_folds, regularization, 1000000))
    else:
        print(logistic_regression(ppdb, target, folds, regularization, 1000000))


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

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def svm_crossval_grid(data, target, folds):
    print("Crossvalidating SVM with regularization")
    res = []
    gamma_range = np.logspace(-3.5, -2.5, 20)
    c_range = np.linspace(60, 250, 20)
    gamma_range_ticks = [round(i,6) for i in gamma_range]
    print(gamma_range)
    print(c_range)
    total = len(c_range) * len(gamma_range)
    for j, C in enumerate(c_range):
        for i, gamma in enumerate(gamma_range):
            print("At ", round(((j * len(gamma_range) + i) * 100.0) / total, 2), "%")
            res.append([svm_rbf(data, target, folds, C=C, gamma=gamma, max_iter=1000000), C, gamma])

    acc = np.asarray([[a[0][0], a[1], a[2]] for a in res])
    print("Max acc at: ", acc[np.argmax(acc[:, 0]), 1], " ", acc[np.argmax(acc[:, 0]), 2], " ", np.max(acc[:, 0]))

    acc = np.asarray([a[0] for a in acc])
    scores = acc.reshape(len(c_range), len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.65, midpoint=0.75))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range_ticks, rotation=45)
    plt.yticks(np.arange(len(c_range)), c_range, rotation=45)
    plt.title('SVM with L2 regularization accuracy')
    plt.show()

def svm_crossval(data, target, folds):
    for reg in ['l1', 'l2']:
        print("Crossvalidating SVM with "+reg+" regularization")
        res = []
        C_range = np.logspace(-3, 0, 100)
        total = len(C_range)
        for i, C in enumerate(C_range):
            print("At ", round((i * 100.0) / total, 2), "%")
            res.append([svm_rbf(data, target, folds, reg, C=C, max_iter=1000000), C])

        acc = np.asarray([[a[0][0], a[1]] for a in res])
        f1 = np.asarray([[a[0][1], a[1]] for a in res])
        recall = np.asarray([[a[0][2], a[1]] for a in res])
        precision = np.asarray([[a[0][3], a[1]] for a in res])
        print("Max acc without question at default_dist: ", acc[np.argmax(acc[:, 0]), 1], " ", np.max(acc[:, 0]))
        print("Max f1 without question at default_dist: ", f1[np.argmax(f1[:, 0]), 1], " ", np.max(f1[:, 0]))
        print("Max recall without question at default_dist: ", recall[np.argmax(recall[:, 0]), 1], " ",
              np.max(recall[:, 0]))
        print("Max precision without question at default_dist: ", precision[np.argmax(precision[:, 0]), 1], " ",
              np.max(precision[:, 0]))
        plt.plot(acc[:, 1], acc[:, 0], label='Accuracy')
        plt.plot(f1[:, 1], f1[:, 0], label='F1-Score')
        plt.plot(recall[:, 1], recall[:, 0], label='Recall')
        plt.plot(precision[:, 1], precision[:, 0], label='Precision')
        plt.legend()
        plt.xscale('log')
        plt.xlabel("C regularization parameter")
        plt.title("SVM with "+reg+" regularization")
        plt.show()

def combined_crossval(claim_ids, target, rootdist_matrix, tf_matrix, questionmark, folds=7, do_custom_folds=True):
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

    if do_custom_folds:
        folds = custom_folds

    print("Classifier: ", '[accuracy,', 'f1_macro,', 'recall_macro,', 'precision_macro]')
    print("Logistic regression ovr L1: ", logistic_regression(combined_all, target, folds, 'l1', 1000000, 'ovr'))
    print("Logistic regression ovr L2: ", logistic_regression(combined_all, target, folds, 'l2', 1000000, 'ovr'))
    print("Logistic regression multiclass L1: ", logistic_regression(combined_all, target, folds, 'l1', 1000000, 'multinomial'))
    print("Logistic regression multiclass L2: ", logistic_regression(combined_all, target, folds, 'l2', 1000000, 'multinomial'))
    print("SVM Cross-validation")
    svm_crossval_grid(combined_all, target, folds)
    print("Naive Bayes: ", naive_bayes(combined_all.toarray(), target, folds))


if __name__ == "__main__":
    data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance', 'claimId'])
    headers = ['articleHeadline', 'articleHeadlineStance']

    data_split = split_data(data)
    x = data_split[0]
    y = data_split[1]
    print_data_distribution(y)

    ids = data_split[2]

    rootdist = get_rootdist_matrix(7)
    questionmark_features = extract_questionmark_features(data_split, headers.index('articleHeadline'))
    bow = BoW(ngram_range=(1, 2), max_features=90, stop_words=None)
    tf = bow.fit(x)
    ppdb_alignment_feature = sparse.csr_matrix(get_ppdb_alignment_feature())

    # print("Rootdist grid_search")
    # crossval_grid_search(y, ids, min_rootdist=100, max_rootdist=100, bow=tf, ppdb=ppdb_alignment_feature, questionmark_features=questionmark_features)
    variance_plot_cv(ids, y, rootdist, tf, questionmark_features, range(2, 10))
    print("PPDB only")
    ppdb_only(ids, y, ppdb_alignment_feature, 10, False)
    print("K-fold variance plot")
    #variance_plot_cv(ids, y, rootdist, tf, questionmark_features, range(2, 30))
    print("Questionmark only")
    questionmark_only(ids, y, questionmark_features, 7, True)
    print("BoW with Rootdist")
    bow_rootdist(ids, y, rootdist, tf, 7, True)
    print("All features")
    show_confusion_matrix(y, ['for', 'against', 'observing'], rootdist, questionmark_features)
    combined_crossval(ids, y, rootdist, tf, questionmark_features, 7, True)
