import csv
from bow_grid_search import split_data
from rootdist import get_rootdist_matrix, crossval_rootdist
from scipy import sparse
import numpy as np

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


def extract_questionmark_features(data, header_index):
    features = []
    for i, row in enumerate(data):
        # Skip the headers
        if i == 0:
            continue

        has_questionmark = '?' in row[header_index]
        features.append(has_questionmark)
    return sparse.csr_matrix(np.array([features]).T)

if __name__ == "__main__":
    data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance', 'claimId'])

    print(f'Headers: {data[0]}')
    print(f'First row: {data[1]}')

    headers = ['articleHeadline', 'articleHeadlineStance']
    rootdist = get_rootdist_matrix()
    questionmark_features = extract_questionmark_features(data, headers.index('articleHeadline'))

    data = split_data(data)


    x = data[0]
    y = data[1]
    ids = data[2]

    print("Rootdist without questionmark")
    crossval_rootdist(rootdist, y, ids, [])
    print("Rootdist with questionmark")
    crossval_rootdist(rootdist, y, ids, questionmark_features)
