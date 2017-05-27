import json
import pickle
import pandas as pd
from utils import normalize_text

def get_features(message, vectorizer):
    normalized_message = normalize_text(message)
    features = vectorizer.transform([normalized_message])
    df = pd.DataFrame(features.toarray())
    df['__question_mark__'] = '?' in message
    return df

def predict(message, model, vectorizer):
    features = get_features(message, vectorizer)
    predicted = model.predict(features)
    probabilities = zip(model.classes_, model.predict_proba(features)[0])
    return predicted[0], probabilities

def predictIntent(message, classifier, balanced_data, selected_features, feature_mode):
    # getModel
    model_name = classifier + '_' + feature_mode + '_'
    model_name += ('bal' if balanced_data else 'inb')
    model_name += ('_s' if selected_features else '')
    model = pickle.load(open('../training_testing/output/' + model_name + '.pkl', 'rb'))

    # getVectorizer
    vectorizer_name = 'vect_' + feature_mode
    vectorizer_name += ('_balanced' if balanced_data else '')
    if selected_features:
        vectorizer_name = '../feature_selection/output/' + vectorizer_name + '_selected.pkl'
    else:
        vectorizer_name = '../feature_extraction/output/' + vectorizer_name + '.pkl'

    vectorizer = pickle.load(open(vectorizer_name, 'rb'))

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
