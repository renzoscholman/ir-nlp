from corenlp import *
import os
import pandas as pd
import warnings
import csv
try:
    import cPickle as pickle
except:
    import pickle

hedging_words = [
    'alleged', 'allegedly',
    'apparently',
    'appear', 'appears',
    'claim', 'claims',
    'could',
    'evidently',
    'largely',
    'likely',
    'mainly',
    'may', 'maybe', 'might',
    'mostly',
    'perhaps',
    'presumably',
    'probably',
    'purported', 'purportedly',
    'reported', 'reportedly',
    'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
    'says',
    'seem',
    'somewhat',
    # 'supposedly',
    'unconfirmed'
]

refuting_words = [
    'fake',
    'fraud',
    'hoax',
    'false',
    'deny', 'denies',
    # 'refute',
    'not',
    'despite',
    'nope',
    'doubt', 'doubts',
    'bogus',
    'debunk',
    'pranks',
    'retract'
]

DATA_PATH = '../data/url-versions-2015-06-14-clean.csv'

def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [tuple(headers)]
        for row in csv_reader:
            row_data = [(row[header]) for header in headers]
            rows.append(tuple(row_data))

        return rows

def get_dataset(path=DATA_PATH):
    return pd.read_csv(path)

def stanford_data_parse(path=DATA_PATH):
    df = get_dataset(path)
    print(df)
    nlp = NLPRootDist(model_location='../corenlp_models')
    data = {}
    # Ignore UserWarning because nlp.parse throws a warning due to wrong data type,
    # but we have no control over Stanford CoreNLP code
    warnings.simplefilter("ignore", UserWarning)
    total = len(df)
    for id, row in df.iterrows():
        if id % 50 == 0:
            print("Calculating rootdist, at: ", "{0:.2f}".format((id/total)*100.0), "% (", id, " of ", total, ")")
        try:
            data[row.claimId] = nlp.parse(row.claimHeadline)
            data[row.articleId] = nlp.parse(row.articleHeadline)
        except:
            print("Can't parse the following")
            print("Claim: " + row.claimHeadline)
            print("Article: " + row.articleHeadline)
    # Reset the filter for UserWarnings as we are done with the Stanford CoreNLP
    warnings.simplefilter("default", UserWarning)

    with open(os.path.join('..', 'data', 'pickled', 'stanparse-data.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



def build_dep_graph(deps):
    dep_graph = {}
    dep_graph_labels = {}

    for d in deps:
        head, rel, dep = d
        dep_graph.setdefault(int(head.index), set()).add(int(dep.index))
        dep_graph_labels[(int(head.index), int(dep.index))] = rel
    return dep_graph, dep_graph_labels


def calc_depths(grph, n=0, d=0, depths=None):
    if depths is None:
        depths = {n: d}
    sx = grph.get(n)
    if sx:
        for s in sx:
            depths[s] = d + 1
            calc_depths(grph, s, d + 1, depths)
    return depths

def rootdist_parse():
    dep_parse_data = get_stanparse_data()
    data = {}
    for id, dep_parse in dep_parse_data.items():
        for i, s in enumerate(dep_parse.sentences):
            grph, grph_labels = build_dep_graph(s.dependencies)
            grph_depths = calc_depths(grph)
            d = data.setdefault(id, {})
            d[i] = grph, grph_labels, grph_depths

    with open(os.path.join('..', 'data', 'pickled', 'stanparse-depths.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_stanparse_data(path='../data/pickled'):
    with open(os.path.join(path,'stanparse-data.pickle'), 'rb') as f:
        return pickle.load(f)


def get_stanparse_depths(path='../data/pickled'):
    with open(os.path.join(path, 'stanparse-depths.pickle'), 'rb') as f:
        return pickle.load(f)