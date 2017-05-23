import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def main():

    print 'Loading data from \'feature_extraction/output/\''

    data_freq_instances = pd.read_csv('../feature_extraction/output/feature_freq.csv')
    data_bool_instances = pd.read_csv('../feature_extraction/output/feature_bool.csv')
    data_labels = pd.read_csv('../feature_extraction/output/label.csv')


    balanced_data_instances = pd.read_csv('../feature_extraction/output/feature_balanced.csv')
    balanced_data_labels = pd.read_csv('../feature_extraction/output/balanced_label.csv')

    k = 100
    print 'Selecting K Best features, k =', k

    # Univariate Feature Selection
    chi2select = SelectKBest(chi2, k=k)
    chi2result = chi2select.fit_transform(balanced_data_instances, balanced_data_labels)

    print 'Output selected features to \'output/\''
    balanced_data_instances.loc[:, chi2select.get_support()].to_csv('output/balanced_features_best', index=False)



if __name__ == '__main__':
	main()
