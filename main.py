import csv
from corenlp import NLPRootDist

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'

def extract_article_headers(data_path, headers):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        for row in csv_reader:
            row_data = [(header, row[header]) for header in headers]
            rows.append(row_data)

        return rows

data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance'])
models = ['en_lines', 'en_ewt', 'en_gum']
sentence = "Iraq Says Arrested Woman Is Not The Wife of ISIS Leader al-Baghdadi"
for i in models:
    nlp = NLPRootDist(treebank=i)
    hedge, refute = nlp.parse(sentence)
    print(i, " ", hedge, " ", refute, " ", sentence)
