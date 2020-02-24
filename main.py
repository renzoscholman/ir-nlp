import csv

DATA_PATH = './data/url-versions-2015-06-14-clean.csv'

def extract_article_headers(data_path, headers):
	with open(data_path, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		rows = [tuple(headers)]
		for row in csv_reader:
			row_data = [(row[header]) for header in headers]
			rows.append(tuple(row_data))

		return rows

def extract_header_column(data, header_index):
	return list(map(lambda row: row[header_index], data[1:]))

def extract_questionmark_features(data, header_index):
	features = []
	for i, row in enumerate(data):
		# Skip the headers
		if i == 0:
			continue

		has_questionmark = '?' in row[header_index]
		features.append(has_questionmark)
	return features

headers = ['articleHeadline', 'articleHeadlineStance']
data = extract_article_headers(DATA_PATH, headers)

questionmark_features = extract_questionmark_features(data, headers.index('articleHeadline'))
count_true = len(list(filter(lambda x: x, questionmark_features)))
count_false = len(questionmark_features) - count_true
print(f'- Questionmarks total: {len(questionmark_features)}, with: {count_true}, without: {count_false}')
