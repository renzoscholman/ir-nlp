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

data = extract_article_headers(DATA_PATH, ['articleHeadline', 'articleHeadlineStance'])
print(f'Headers: {data[0]}')
print(f'First row: {data[1]}')
