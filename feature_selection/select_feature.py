import pandas as pd
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer

def main():

    print '[+] Loading data from \'feature_extraction/output/\''

    data_freq_instances = pd.read_csv('../feature_extraction/output/features_freq.csv')
    data_bool_instances = pd.read_csv('../feature_extraction/output/features_bool.csv')
    balanced_data_freq_instances = pd.read_csv('../feature_extraction/output/features_freq_balanced.csv')
    balanced_data_bool_instances = pd.read_csv('../feature_extraction/output/features_bool_balanced.csv')

    data_labels = pd.read_csv('../feature_extraction/output/label.csv')
    balanced_data_labels = pd.read_csv('../feature_extraction/output/balanced_label.csv')

    # Univariate Feature Selection
    K_INBALANCED = 700
    K_BALANCED = 300

    print '[+] Selecting K Best features in inbalanced dataset, k =', K_INBALANCED
    
    c2_freq_inb = SelectKBest(chi2, k=K_INBALANCED)
    c2_freq_inb_result = c2_freq_inb.fit_transform(data_freq_instances, data_labels)

    c2_bool_inb = SelectKBest(chi2, k=K_INBALANCED)
    c2_bool_inb_result = c2_bool_inb.fit_transform(data_bool_instances, data_labels)

    print '[+] Selecting K Best features in balanced dataset, k =', K_BALANCED
    
    c2_freq_bal = SelectKBest(chi2, k=K_BALANCED)
    c2_freq_bal_result = c2_freq_bal.fit_transform(balanced_data_freq_instances, balanced_data_labels)

    c2_bool_bal = SelectKBest(chi2, k=K_BALANCED)
    c2_bool_bal_result = c2_bool_bal.fit_transform(balanced_data_bool_instances, balanced_data_labels)


    print '[+] Output selected features to \'output/\''
    selected_features_freq_inb = data_freq_instances.columns.values[c2_freq_inb.get_support()]
    selected_features_bool_inb = data_bool_instances.columns.values[c2_bool_inb.get_support()]
    selected_features_freq_bal = balanced_data_freq_instances.columns.values[c2_freq_bal.get_support()]
    selected_features_bool_bal = balanced_data_bool_instances.columns.values[c2_bool_bal.get_support()]

    open('output/selected_features_freq.txt', 'wb').write('\n'.join(selected_features_freq_inb))
    open('output/selected_features_bool.txt', 'wb').write('\n'.join(selected_features_bool_inb))
    open('output/selected_features_freq_balanced.txt', 'wb').write('\n'.join(selected_features_freq_bal))
    open('output/selected_features_bool_balanced.txt', 'wb').write('\n'.join(selected_features_bool_bal))

    data_freq_instances.loc[:, c2_freq_inb.get_support()].to_csv('output/features_freq_selected.csv', index=False)
    data_bool_instances.loc[:, c2_bool_inb.get_support()].to_csv('output/features_bool_selected.csv', index=False)
    balanced_data_freq_instances.loc[:, c2_freq_bal.get_support()].to_csv('output/features_freq_balanced_selected.csv', index=False)
    balanced_data_bool_instances.loc[:, c2_bool_bal.get_support()].to_csv('output/features_bool_balanced_selected.csv', index=False)

    print '[+] Output updated vectorizer to \'output\''

    vect_freq = CountVectorizer(vocabulary = selected_features_freq_inb[:-1])
    vect_bool = CountVectorizer(vocabulary = selected_features_bool_inb[:-1], binary = True)
    vect_freq_balanced = CountVectorizer(vocabulary = selected_features_freq_bal[:-1])
    vect_bool_balanced = CountVectorizer(vocabulary = selected_features_bool_bal[:-1], binary = True)

    # Dump persistence Vectorizer
    pickle.dump(vect_freq, open('output/vect_freq_selected.pkl', 'wb'))
    pickle.dump(vect_bool, open('output/vect_bool_selected.pkl', 'wb'))
    pickle.dump(vect_freq_balanced, open('output/vect_freq_balanced_selected.pkl', 'wb'))
    pickle.dump(vect_bool_balanced, open('output/vect_bool_balanced_selected.pkl', 'wb'))


if __name__ == '__main__':
	main()
