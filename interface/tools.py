import json
import pickle
import pandas as pd

def getFeatures(message, balanced_data, selected_features):
    if selected_features:
        raise NotImplementedError

    vectorizer_name = 'count_vect_' + ('balance' if balanced_data else 'freq') + '.pkl'
    count_vect_freq = pickle.load(open('../feature_extraction/output/' + vectorizer_name, 'rb'))

    features = count_vect_freq.transform([message])
    df = pd.DataFrame(features.toarray())

    return df

def predictIntent(message, classifier, balanced_data, selected_features):
    model_name = classifier + '_model_'
    model_name += ('balanced' if balanced_data else 'biased') + '_'
    model_name += ('selected' if selected_features else 'not-selected')

    model = pickle.load(open('../training/output/' + model_name + '.pkl', 'rb'))

    features = getFeatures(message, balanced_data, selected_features)
    predicted = model.predict(features)
    probabilities = zip(model.classes_, model.predict_proba(features)[0])

    return predicted[0], probabilities


def classify(message, classifier, balanced_data, selected_features):
    intent, probs = predictIntent(message, classifier, balanced_data, selected_features)
    probabilities = { i: format(x, '.8f') for i,x in probs }
    model = {
        'classifier': classifier,
        'balanced_data': balanced_data,
        'selected_features': selected_features
    }
    details = {
        'intent': intent,
        'text': message,
        'probabilities': probabilities,
        'model': model
    }
    return {
        'intent': intent,
        'text': message,
        'details': json.dumps(details, indent=4, sort_keys=True),
        'confidence_labels': probabilities.keys(),
        'confidence_values': probabilities.values(),
    }
