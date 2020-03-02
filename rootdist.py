from parse.corenlp_parse import *

from scipy import sparse
import numpy as np

from bow_grid_search import add_question_mark_feature, logistic_regression
from cross_val import cv_fold_generator

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'

def get_rootdist_matrix():
    data = get_dataset(DATA_PATH)
    stanparse_depths = get_stanparse_depths('data/pickled')
    stanparse_data = get_stanparse_data('data/pickled')

    mat = np.zeros((len(data), 2))
    min_hedge_depth = min_refute_depth = 100
    for i, (_, s) in enumerate(data.iterrows()):
        try:
            sp_data = stanparse_data[s.articleId]
            sp_depths = stanparse_depths[s.articleId]
            min_hedge_depth = min_refute_depth = 100

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
