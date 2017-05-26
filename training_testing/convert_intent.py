input_file = 'test-dataset.csv'
output_file = 'data/test-dataset-converted.csv'

def convert(s):
    if(s in ['ask-permission-to-ask', 'help', 'intro']):
        return 'asking-question-other'
    if(s in ['ask-for-direction']):
        return 'asking-question-direction'
    if(s in ['positive-feedback', 'negative-feedback', 'giving-positive-feedback', 
             'giving-negative-feedback', 'greeting']):
        return 'giving-feedback-or-greeting'
    # if(s in ['positive-feedback']):
    #   return 'giving-positive-feedback'
    # if(s in ['negative-feedback']):
    #   return 'giving-negative-feedback'
    if(s in ['ask-for-schedule-krl', 'ask-transjakarta', 'ask-for-schedule-krl-complete']):
        return 'asking-question-schedule'
    if(s == None or s == "" or s == '!' or s == 'bye' or s == 'nan'):
        return 'other'
    else:
        return s

import pandas as pd

data = pd.read_csv(input_file)
count_intent = {}

for i,row in data.iterrows():
    data.loc[i,'intent'] = convert(str(data.loc[i,'intent']))
    if data.loc[i,'intent'] in count_intent:
        count_intent[data.loc[i,'intent']] += 1
    else:
        count_intent[data.loc[i,'intent']] = 1

data.to_csv(output_file)

unique_intent = list(set(data['intent']))
print 'Unique intents:', len(unique_intent)

intent_summary = pd.DataFrame({ 'Count': count_intent.values() }, index=count_intent.keys())
print intent_summary
