import pandas as pd
import pickle
from sklearn import metrics

test_data = pd.read_csv('test-dataset.csv')

def gen_report(model, vectorizer, test_data):
    features = vectorizer.transform(test_data['text'])
    predicted = model.predict(features)
    report = metrics.classification_report(test_data['intent'], predicted)
    print '*' * 70
    print str(model)
    print '*' * 70
    print report

MNB_freq_inb = pickle.load(open('output/MNB_freq_inb.pkl', 'rb'))
MNB_bool_inb = pickle.load(open('output/MNB_bool_inb.pkl', 'rb'))
MNB_freq_bal = pickle.load(open('output/MNB_freq_bal.pkl', 'rb'))
MNB_bool_bal = pickle.load(open('output/MNB_bool_bal.pkl', 'rb'))
tree_freq_inb = pickle.load(open('output/tree_freq_inb.pkl', 'rb'))
tree_bool_inb = pickle.load(open('output/tree_bool_inb.pkl', 'rb'))
tree_freq_bal = pickle.load(open('output/tree_freq_bal.pkl', 'rb'))
tree_bool_bal = pickle.load(open('output/tree_bool_bal.pkl', 'rb'))

vect_freq_selected = pickle.load(open('../feature_selection/output/vect_freq_selected.pkl', 'rb'))
vect_bool_selected = pickle.load(open('../feature_selection/output/vect_bool_selected.pkl', 'rb'))
vect_freq_balanced_selected = pickle.load(open('../feature_selection/output/vect_freq_balanced_selected.pkl', 'rb'))
vect_bool_balanced_selected = pickle.load(open('../feature_selection/output/vect_bool_balanced_selected.pkl', 'rb'))

gen_report(MNB_freq_inb, vect_freq_selected, test_data)
gen_report(MNB_bool_inb, vect_bool_selected, test_data)
gen_report(MNB_freq_bal, vect_freq_balanced_selected, test_data)
gen_report(MNB_bool_bal, vect_freq_balanced_selected, test_data)
gen_report(tree_freq_inb, vect_freq_selected, test_data)
gen_report(tree_bool_inb, vect_bool_selected, test_data)
gen_report(tree_freq_bal, vect_freq_balanced_selected, test_data)
gen_report(tree_bool_bal, vect_freq_balanced_selected, test_data)
