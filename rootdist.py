from parse.corenlp_parse import *

from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from bow_grid_search import add_question_mark_feature, logistic_regression
from cross_val import cv_fold_generator

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'

def get_rootdist_matrix(default_score=100):
    data = get_dataset(DATA_PATH)
    stanparse_depths = get_stanparse_depths('data/pickled')
    stanparse_data = get_stanparse_data('data/pickled')

    mat = np.zeros((len(data), 2))
    min_hedge_depth = min_refute_depth = default_score
    for i, (_, s) in enumerate(data.iterrows()):
        try:
            sp_data = stanparse_data[s.articleId]
            sp_depths = stanparse_depths[s.articleId]
            min_hedge_depth = min_refute_depth = default_score

            for j, sentence in enumerate(sp_data.sentences):
                grph, grph_labels, grph_depths = sp_depths[j]
                lemmas = list(enumerate([d.lemma.lower() for d in sentence.words], start=1))
                h_depths = [h if lem in hedging_words else 0 for (h, lem) in lemmas]
                r_depths = [h if lem in refuting_words else 0 for (h, lem) in lemmas]

                hedge_depths = [grph_depths[d] for d in h_depths if d > 0]
                refute_depths = [grph_depths[d] for d in r_depths if d > 0]

                hedge_depths.append(min_hedge_depth)
                refute_depths.append(min_refute_depth)

                min_hedge_depth = min(hedge_depths)
                min_refute_depth = min(refute_depths)
        except:
            pass
        mat[i, 0] = min_hedge_depth
        mat[i, 1] = min_refute_depth
    return mat


def crossval_rootdist(data, target, ids, questionmark_features=None, folds=10, do_custom_folds=True):
    custom_folds = cv_fold_generator(ids, folds)
    data = sparse.csr_matrix(data)
    if questionmark_features is not None:
        combined = add_question_mark_feature(data, questionmark_features)
    else:
        combined = data
    print('accuracy', 'f1_macro', 'recall_macro', 'precision_macro')
    if do_custom_folds:
        print(logistic_regression(combined, target, custom_folds))
    else:
        print(logistic_regression(combined, target, folds))

def crossval_grid_search(target, ids, min_rootdist=1, max_rootdist=200, step=1, ppdb=None, questionmark_features=None, bow=None, folds=10):
    default_score = range(min_rootdist, max_rootdist+1, step)
    res = []
    count = 0
    custom_folds = cv_fold_generator(ids, folds)
    for i in default_score:
        data = sparse.csc_matrix(get_rootdist_matrix(i))
        print("At ", round((count * 100.0) / (len(default_score)), 2), "%")
        count += 1
        combined = sparse.hstack((
            data,
            questionmark_features,
            bow,
            ppdb
        ))

        regularization = 'l2'
        res.append([logistic_regression(combined, target, custom_folds, regularization), i])

    acc = np.asarray([[a[0][0], a[1]] for a in res])
    print("Max acc without question at default_dist: ", acc[np.argmax(acc[:, 0]), 1])
    plt.plot(acc[:, 1], acc[:, 0])
    plt.xlabel("Default rootdist score")
    plt.ylabel("Accuracy")
    plt.show()

    return res

