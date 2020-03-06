from parse.corenlp_parse import get_dataset
import nltk
import pandas as pd
import numpy as np
from munkres import Munkres, make_cost_matrix
from scipy import sparse
try:
    import cPickle as pickle
except:
    import pickle

MIN_SCORE = -10
MAX_SCORE = 10
PICKLED_PATH = './data/pickled'

# Get the ppdb pickle file from: https://github.com/willferreira/mscproject (dropbox/drive)
def get_ppdb_data(path=f'{PICKLED_PATH}/ppdb.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

munk = Munkres()
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()
ppdb_data = get_ppdb_data()


def get_tokenized_lemmas(sentence):
	return [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(sentence)]

def compute_paraphrase_score(c, a):
	c_stem = stemmer.stem(c) # claim headline stem
	a_stem = stemmer.stem(a) # article headline stem

	if c_stem == a_stem:
		return MAX_SCORE

	c_paraphrases = set(ppdb_data.get(c, [])).union(ppdb_data.get(c_stem, []))
	matches = list(filter(lambda x: x[0] == a or x[0] == a_stem, c_paraphrases))
	if len(matches) > 0:
		return max(matches, key=lambda x: x[1])[1]
	return MIN_SCORE

def calculate_alignment_score(claim_headline, article_headline):
	c_tokens = get_tokenized_lemmas(claim_headline)
	a_tokens = get_tokenized_lemmas(article_headline)

	df = pd.DataFrame(index=c_tokens, columns=a_tokens, data=0.)

	for c in c_tokens:
		for a in a_tokens:
			df.loc[c, a] = compute_paraphrase_score(c, a)

	matrix = df.values
	cost_matrix = make_cost_matrix(matrix, lambda cost: MAX_SCORE - cost)
	indexes = munk.compute(cost_matrix)
	total = 0.0
	for row, column in indexes:
		value = matrix[row][column]
		total += value
	return indexes, total / float(np.min(matrix.shape))

def parse_alignment_score_calculation(data_path='data', pickled_path=PICKLED_PATH):
	df = get_dataset(f'{data_path}/url-versions-2015-06-14-clean.csv')
	df_size = len(df.index)
	data = {}

	for i, row in df.iterrows():
		data[(row.claimId, row.articleId)] = calculate_alignment_score(row.claimHeadline, row.articleHeadline)
		if i % 100 == 0:
			print(f'Alignment calculation progress: {i}/{df_size} ({i/df_size*100}%)')

	with open(f'{pickled_path}/alignment-score.pickle', 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def get_alignment_data(pickled_path=PICKLED_PATH):
	with open(f'{pickled_path}/alignment-score.pickle', 'rb') as f:
		return pickle.load(f)

def get_ppdb_alignment_feature(pickled_path=PICKLED_PATH):
	data = get_alignment_data()
	alignment_scores = [v[1] for k, v in data.items()]

	return np.array([alignment_scores]).T	
