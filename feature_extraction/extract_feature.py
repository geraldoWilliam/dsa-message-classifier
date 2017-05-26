import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def write_feature_to_file(filename, count_vect, data):
	feature = count_vect.fit_transform(data.values.astype('U'))
	df = pd.DataFrame(feature.toarray())
	df.to_csv(filename, header = count_vect.get_feature_names(), index=False)

def main():
	data = pd.read_csv('../preprocessing/output/dataset.csv', index_col=0)
	balanced_data = pd.read_csv('../preprocessing/output/balanced_dataset.csv', index_col=0)

	vect_freq = CountVectorizer()
	vect_bool = CountVectorizer(binary = True)
	vect_freq_balanced = CountVectorizer()
	vect_bool_balanced = CountVectorizer(binary = True)

	if not os.path.exists('output'):
	    os.makedirs('output')

	# Fit vectorizer and Save features to files
	write_feature_to_file('output/features_freq.csv', vect_freq, data['text'])
	write_feature_to_file('output/features_bool.csv', vect_bool, data['text'])
	write_feature_to_file('output/features_freq_balanced.csv', vect_freq_balanced, balanced_data['text'])
	write_feature_to_file('output/features_bool_balanced.csv', vect_bool_balanced, balanced_data['text'])

	# Save labels to files
	label = data['intent']
	label.to_csv('output/label.csv', header=['intent'], index = False)

	balanced_label = balanced_data['intent']
	balanced_label.to_csv('output/balanced_label.csv', header=['intent'], index = False)

	# Dump persistence Vectorizer
	pickle.dump(vect_freq, open('output/vect_freq.pkl', 'wb'))
	pickle.dump(vect_bool, open('output/vect_bool.pkl', 'wb'))
	pickle.dump(vect_freq_balanced, open('output/vect_freq_balanced.pkl', 'wb'))
	pickle.dump(vect_bool_balanced, open('output/vect_bool_balanced.pkl', 'wb'))

if __name__ == '__main__':
	main()
