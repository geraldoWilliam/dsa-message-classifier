import json
import pickle
import pandas as pd
from utils import normalize_text

def predict(message, model, vectorizer):
    normalized_message = normalize_text(message)
    features = vectorizer.transform([normalized_message])
    predicted = model.predict(features)
    probabilities = zip(model.classes_, model.predict_proba(features)[0])
    return predicted[0], probabilities

def predictIntent(message, classifier, balanced_data, selected_features, feature_mode):
    if not selected_features:
        raise NotImplementedError

    # getModel
    model_name = classifier + '_' + feature_mode + '_'
    model_name += ('bal' if balanced_data else 'inb')
    model = pickle.load(open('../training_testing/output/' + model_name + '.pkl', 'rb'))

    # getVectorizer
    vectorizer_name = 'vect_' + feature_mode
    vectorizer_name += ('_balanced' if balanced_data else '')
    vectorizer_name += '_selected'
    vectorizer = pickle.load(open('../feature_selection/output/' + vectorizer_name + '.pkl', 'rb'))

    return predict(message, model, vectorizer)


def classify(message, classifier, balanced_data, selected_features, feature_mode):
    intent, probs = predictIntent(message, classifier, balanced_data, selected_features, feature_mode)
    probabilities = { i: format(x, '.8f') for i,x in probs }
    model = {
        'classifier': classifier,
        'balanced_data': balanced_data,
        'selected_features': selected_features,
        'feature_mode': feature_mode
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
