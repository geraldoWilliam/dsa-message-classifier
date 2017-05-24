import pandas as pd
import pickle

def generate_report(model, training_data, label):
	# classification report and k-fold cross validation
	from sklearn.model_selection import cross_val_score
	from sklearn import metrics

	scores = cross_val_score(model, training_data, label, cv=5)
	print scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	predicted = model.predict(training_data)
	report = metrics.classification_report(label, predicted)
	c_mat = metrics.confusion_matrix(label, predicted)

	print c_mat
	print report

def main():
	training_data = pd.read_csv('../feature_extraction/output/feature_freq.csv')
	balanced_training_data = pd.read_csv('../feature_extraction/output/feature_balanced.csv')
	label = pd.read_csv('../feature_extraction/output/label.csv')
	balanced_label = pd.read_csv('../feature_extraction/output/balanced_label.csv')

	training_data = training_data.as_matrix()
	label = label['intent']

	balanced_training_data = balanced_training_data.as_matrix()
	balanced_label = balanced_label['intent']

	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	data_selected = SelectKBest(chi2, k=700).fit_transform(training_data, label)
	balanced_data_selected = SelectKBest(chi2, k=300).fit_transform(balanced_training_data, balanced_label)

	from sklearn.naive_bayes import MultinomialNB

	print "MultnomialNB, biased training data"
	MNB_model = MultinomialNB().fit(training_data, label)
	generate_report(MNB_model, training_data, label)
	pickle.dump(MNB_model, open('output/MNB_model_biased_not-selected.pkl', 'wb'))
	print "*****************************\n"

	print "MultinomialNB, biased training data, feature selected"
	MNB_model = MultinomialNB().fit(data_selected, label)
	generate_report(MNB_model, data_selected, label)
	pickle.dump(MNB_model, open('output/MNB_model_biased_selected.pkl', 'wb'))
	print "*****************************\n"

	print "MultinomialNB, balanced training data"
	balanced_MNB_model = MultinomialNB().fit(balanced_training_data, balanced_label)
	generate_report(balanced_MNB_model, balanced_training_data, balanced_label)
	pickle.dump(balanced_MNB_model, open('output/MNB_model_balanced_not-selected.pkl', 'wb'))
	print "*****************************\n"

	print "MultinomialNB, balanced training data, feature selected"
	balanced_MNB_model = MultinomialNB().fit(balanced_data_selected, balanced_label)
	generate_report(balanced_MNB_model, balanced_data_selected, balanced_label)
	pickle.dump(balanced_MNB_model, open('output/MNB_model_balanced_selected.pkl', 'wb'))
	print "*****************************\n"


	print "\n\n"
	from sklearn.tree import DecisionTreeClassifier

	print "DecisionTree, biased training data"
	tree_model = DecisionTreeClassifier().fit(training_data, label)
	generate_report(tree_model, training_data, label)
	pickle.dump(tree_model, open('output/tree_model_biased_not-selected.pkl', 'wb'))
	print "*****************************\n"

	print "DecisionTree, biased training data, feature selected"
	tree_model = DecisionTreeClassifier().fit(data_selected, label)
	generate_report(tree_model, data_selected, label)
	pickle.dump(tree_model, open('output/tree_model_biased_selected.pkl', 'wb'))
	print "*****************************\n"

	print "DecisionTree, balanced training data"
	tree_model = DecisionTreeClassifier().fit(balanced_training_data, balanced_label)
	generate_report(tree_model, balanced_training_data, balanced_label)
	pickle.dump(tree_model, open('output/tree_model_balanced_not-selected.pkl', 'wb'))
	print "*****************************\n"

	print "DecisionTree, balanced training data, feature selected"
	tree_model = DecisionTreeClassifier().fit(balanced_data_selected, balanced_label)
	generate_report(tree_model, balanced_data_selected, balanced_label)
	pickle.dump(tree_model, open('output/tree_model_balanced_selected.pkl', 'wb'))
	print "*****************************\n"


if __name__ == '__main__':
	main()
