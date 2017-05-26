import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pickle

def main():

    print "[+] Reading data from files"
    dataset_freq_inb = pd.read_csv('../feature_selection/output/features_freq_selected.csv')
    dataset_bool_inb = pd.read_csv('../feature_selection/output/features_bool_selected.csv')
    dataset_freq_bal = pd.read_csv('../feature_selection/output/features_freq_balanced_selected.csv')
    dataset_bool_bal = pd.read_csv('../feature_selection/output/features_bool_balanced_selected.csv')
    dataset_label = pd.read_csv('../feature_extraction/output/label.csv')['intent']
    dataset_label_bal = pd.read_csv('../feature_extraction/output/balanced_label.csv')['intent']

    print "[+] Training MNB models"
    MNB_freq_inb = MultinomialNB().fit(dataset_freq_inb, dataset_label)
    MNB_bool_inb = MultinomialNB().fit(dataset_bool_inb, dataset_label)
    MNB_freq_bal = MultinomialNB().fit(dataset_freq_bal, dataset_label_bal)
    MNB_bool_bal = MultinomialNB().fit(dataset_bool_bal, dataset_label_bal)

    print "[+] Training Decision Tree models"
    tree_freq_inb = DecisionTreeClassifier().fit(dataset_freq_inb, dataset_label)
    tree_bool_inb = DecisionTreeClassifier().fit(dataset_bool_inb, dataset_label)
    tree_freq_bal = DecisionTreeClassifier().fit(dataset_freq_bal, dataset_label_bal)
    tree_bool_bal = DecisionTreeClassifier().fit(dataset_bool_bal, dataset_label_bal)

    print "[+] Saving models to files"
    pickle.dump(MNB_freq_inb, open('output/MNB_freq_inb.pkl', 'wb'))
    pickle.dump(MNB_bool_inb, open('output/MNB_bool_inb.pkl', 'wb'))
    pickle.dump(MNB_freq_bal, open('output/MNB_freq_bal.pkl', 'wb'))
    pickle.dump(MNB_bool_bal, open('output/MNB_bool_bal.pkl', 'wb'))
    pickle.dump(tree_freq_inb, open('output/tree_freq_inb.pkl', 'wb'))
    pickle.dump(tree_bool_inb, open('output/tree_bool_inb.pkl', 'wb'))
    pickle.dump(tree_freq_bal, open('output/tree_freq_bal.pkl', 'wb'))
    pickle.dump(tree_bool_bal, open('output/tree_bool_bal.pkl', 'wb'))

if __name__ == '__main__':
    main()
