import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def generate_report(model, training_data, label):
    # classification report and k-fold cross validation

    scores = cross_val_score(model, training_data, label, cv=5)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    predicted = model.predict(training_data)
    report = metrics.classification_report(label, predicted)
    c_mat = metrics.confusion_matrix(label, predicted)

    print c_mat
    print report

def get_headers(model_name, feature_methods, is_balanced, is_selected):
    return ', '.join([
        model_name,
        feature_methods,
        'balanced' if is_balanced else 'biased',
        'selected' if is_selected else 'not-selected'
        ]) + 'training data'

def gen_reports():
    dataset_freq_inb = pd.read_csv('../feature_selection/output/features_freq_selected.csv')
    dataset_bool_inb = pd.read_csv('../feature_selection/output/features_bool_selected.csv')
    dataset_freq_bal = pd.read_csv('../feature_selection/output/features_freq_balanced_selected.csv')
    dataset_bool_bal = pd.read_csv('../feature_selection/output/features_bool_balanced_selected.csv')
    dataset_label = pd.read_csv('../feature_extraction/output/label.csv')['intent']
    dataset_label_bal = pd.read_csv('../feature_extraction/output/balanced_label.csv')['intent']

    MNB_freq_inb = pickle.load(open('output/MNB_freq_inb.pkl', 'rb'))
    MNB_bool_inb = pickle.load(open('output/MNB_bool_inb.pkl', 'rb'))
    MNB_freq_bal = pickle.load(open('output/MNB_freq_bal.pkl', 'rb'))
    MNB_bool_bal = pickle.load(open('output/MNB_bool_bal.pkl', 'rb'))
    tree_freq_inb = pickle.load(open('output/tree_freq_inb.pkl', 'rb'))
    tree_bool_inb = pickle.load(open('output/tree_bool_inb.pkl', 'rb'))
    tree_freq_bal = pickle.load(open('output/tree_freq_bal.pkl', 'rb'))
    tree_bool_bal = pickle.load(open('output/tree_bool_bal.pkl', 'rb'))

    print get_headers('MultinomialNB', 'freq', False, True)
    generate_report(MNB_freq_inb, dataset_freq_inb, dataset_label)
    print "*****************************\n"

    print get_headers('MultinomialNB', 'bool', False, True)
    generate_report(MNB_bool_inb, dataset_bool_inb, dataset_label)
    print "*****************************\n"

    print get_headers('MultinomialNB', 'freq', True, True)
    generate_report(MNB_freq_bal, dataset_freq_bal, dataset_label_bal)
    print "*****************************\n"

    print get_headers('MultinomialNB', 'bool', True, True)
    generate_report(MNB_bool_bal, dataset_bool_bal, dataset_label_bal)
    print "*****************************\n"

    print "\n\n"

    print get_headers('DecisionTree', 'freq', False, True)
    generate_report(tree_freq_inb, dataset_freq_inb, dataset_label)
    print "*****************************\n"

    print get_headers('DecisionTree', 'bool', False, True)
    generate_report(tree_bool_inb, dataset_bool_inb, dataset_label)
    print "*****************************\n"

    print get_headers('DecisionTree', 'freq', True, True)
    generate_report(tree_freq_bal, dataset_freq_bal, dataset_label_bal)
    print "*****************************\n"

    print get_headers('DecisionTree', 'bool', True, True)
    generate_report(tree_bool_bal, dataset_bool_bal, dataset_label_bal)
    print "*****************************\n"

if __name__ == '__main__':
    gen_reports()
