import pandas as pd
import pickle
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
	df.to_csv(filename, header = count_vect.get_feature_names(), index=False)

def main():
	global stopwords
	with open('id_stopwords.txt') as f:
		stopwords = f.read().splitlines()
	
	data = pd.read_csv('../preprocessing/output/cleaned_dump.csv', index_col=0)
	
	# normalize the text
	for i,row in data.iterrows():
		data.loc[i,'text'] = normalize_text(str(data.loc[i,'text']))
	
	unique_intents = data['intent'].unique()
	unique_data = {elem : pd.DataFrame for elem in unique_intents}

	for key in unique_data.keys():
		unique_data[key] = data[:][data['intent'] == key]

	min_row = min([unique_data[key].shape[0] for key in unique_data.keys()])

	balanced_data = pd.DataFrame(columns = data.columns)
	for key in unique_data.keys():
		rows = unique_data[key].sample(n=min_row)
		balanced_data = balanced_data.append(rows)

	balanced_data = balanced_data.sample(frac=1)
	balanced_data.to_csv('output/balanced_data.csv', index=False)

	count_vect_freq = CountVectorizer()
	count_vect_bool = CountVectorizer(binary = True)
	count_vect_balance = CountVectorizer()

	write_feature_to_file('output/feature_freq.csv', count_vect_freq, data['text'])
	write_feature_to_file('output/feature_bool.csv', count_vect_bool, data['text'])
	write_feature_to_file('output/feature_balanced.csv', count_vect_balance, balanced_data['text'])

	label = data['intent']
	label.to_csv('output/label.csv', header=['intent'], index = False)

	balanced_label = balanced_data['intent']
	balanced_label.to_csv('output/balanced_label.csv', header=['intent'], index = False)

	pickle.dump(count_vect_freq, open('output/count_vect_freq.pkl', 'wb'))
	pickle.dump(count_vect_bool, open('output/count_vect_bool.pkl', 'wb'))
	pickle.dump(count_vect_balance, open('output/count_vect_balance.pkl', 'wb'))

if __name__ == '__main__':
	main()
