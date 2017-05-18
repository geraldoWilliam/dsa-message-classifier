import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def normalize_text(s):
	global stopwords
	s = s.lower()
	s = "".join([c for c in s if c.isalnum() or c==' '])
	s = " ".join([w for w in s.split(' ') if w not in stopwords])
	return s

def write_feature_to_file(filename, count_vect, data):
	feature = count_vect.fit_transform(data)

	df = pd.DataFrame(feature.toarray())
	df.to_csv(filename, header = count_vect.get_feature_names())

def main():
	global stopwords
	with open('id_stopwords.txt') as f:
		stopwords = f.read().splitlines()
	
	import pandas as pd
	data = pd.read_csv('../preprocessing/cleaned_dump.csv')
	u_data = pd.DataFrame(columns = data.columns)

	# normalize the text
	for i,row in data.iterrows():
		data.loc[i,'text'] = normalize_text(str(data.loc[i,'text']))
	
	count_vect_freq = CountVectorizer()
	count_vect_bool = CountVectorizer(binary = True)

	write_feature_to_file('feature_freq.csv', count_vect_freq, data['text'])
	write_feature_to_file('feature_bool.csv', count_vect_bool, data['text'])

if __name__ == '__main__':
	main()