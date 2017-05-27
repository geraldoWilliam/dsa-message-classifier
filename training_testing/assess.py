import pandas as pd
import pickle
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics

stopwords = open('../preprocessing/stopwords_id.txt', 'rb').read().splitlines()

abbr_tuples = [ line.split(' = ') for line in open('../preprocessing/abbreviations_id.txt', 'rb').read().splitlines() ]
abbreviations = { s[0] : s[1] for s in abbr_tuples }

def remove_symbols(s):
    return ''.join([c for c in s if c.isalnum() or c == ' '])

def remove_numbers(s):
    return ''.join([c for c in s if c.isalpha() or c == ' '])

def normalize_abbreviations(s):
    words = []
    for w in s.split(' '):
        words.append(abbreviations[w] if w in abbreviations else w)
    return ' '.join(words)

def remove_stopwords(s):
    return ' '.join([w for w in s.split(' ') if w not in stopwords])

def normalize_text(s):
    s = s.lower()
    s = remove_symbols(s)
    s = remove_numbers(s)
    s = normalize_abbreviations(s)
    s = remove_stopwords(s)
    return s

def add_question_mark_attribute(data):
    count_question_mark = [('?' in row['text']) for i, row in data.iterrows()]
    data['__question_mark__'] = pd.Series(count_question_mark)

    return data

test_data = pd.read_csv('test-dataset.csv')
test_data = add_question_mark_attribute(test_data)

for i,row in test_data.iterrows():
    test_data.loc[i,'text'] = normalize_text(str(test_data.loc[i,'text']))

def gen_report(model, vectorizer, test_data):
    features = vectorizer.transform(test_data['text'])
    df = pd.DataFrame(features.toarray())
    df['__question_mark__'] = test_data['__question_mark__'].values
    predicted = model.predict(df)
    report = metrics.classification_report(test_data['intent'], predicted)
    c_mat = metrics.confusion_matrix(test_data['intent'], predicted)

    print '*' * 70
    print str(model)
    print '*' * 70

    print("Accuracy: %0.2f" % (metrics.accuracy_score(test_data['intent'], predicted)))

    # scores = cross_val_score(model, features, test_data['intent'], cv=5)
    # print scores
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print c_mat
    print report
    print

        # for i in range(len(predicted)):
        #     print "%30s %30s %s" % (test_data['intent'][i], predicted[i], 'OK' if predicted[i] == test_data['intent'][i] else '')

vect_freq_selected = pickle.load( open('../feature_selection/output/vect_freq_selected.pkl', 'rb') )
vect_freq_balanced_selected = pickle.load( open('../feature_selection/output/vect_freq_balanced_selected.pkl', 'rb') )
vect_freq = pickle.load( open('../feature_extraction/output/vect_freq.pkl', 'rb') )
vect_freq_balanced = pickle.load( open('../feature_extraction/output/vect_freq_balanced.pkl', 'rb') )

MNB_freq_inb_s = pickle.load( open('output/MNB_freq_inb_s.pkl') )
MNB_freq_inb = pickle.load( open('output/MNB_freq_inb.pkl') )
MNB_freq_bal_s = pickle.load( open('output/MNB_freq_bal_s.pkl') )
MNB_freq_bal = pickle.load( open('output/MNB_freq_bal.pkl') )
tree_freq_inb_s = pickle.load( open('output/tree_freq_inb_s.pkl') )
tree_freq_inb = pickle.load( open('output/tree_freq_inb.pkl') )
tree_freq_bal_s = pickle.load( open('output/tree_freq_bal_s.pkl') )
tree_freq_bal = pickle.load( open('output/tree_freq_bal.pkl') )

gen_report(MNB_freq_inb_s, vect_freq_selected, test_data)
gen_report(MNB_freq_inb, vect_freq, test_data)

gen_report(MNB_freq_bal_s, vect_freq_balanced_selected, test_data)
gen_report(MNB_freq_bal, vect_freq_balanced, test_data)

gen_report(tree_freq_inb_s, vect_freq_selected, test_data)
gen_report(tree_freq_inb, vect_freq, test_data)

gen_report(tree_freq_bal_s, vect_freq_balanced_selected, test_data)
gen_report(tree_freq_bal, vect_freq_balanced, test_data)
