def convert(s):
	if(s in ['ask-permission-to-ask', 'help', 'intro']):
		return 'asking-question-other'
	if(s in ['ask-for-direction']):
		return 'asking-question-direction'
	if(s in ['positive-feedback']):
		return 'giving-positive-feedback'
	if(s in ['negative-feedback']):
		return 'giving-negative-feedback'
	if(s in ['ask-for-schedule-krl', 'ask-transjakarta', 'ask-for-schedule-krl-complete']):
		return 'asking-question-schedule'
	if(s == None or s == "" or s == '!' or s == 'bye' or s == 'nan'):
		return 'other'
	else:
		return s

import pandas as pd

data = pd.read_csv('dump.csv')
count_intent = {}

for i,row in data.iterrows():
	data.loc[i,'intent'] = convert(str(data.loc[i,'intent']))
	if data.loc[i,'intent'] in count_intent:
		count_intent[data.loc[i,'intent']] += 1
	else:
		count_intent[data.loc[i,'intent']] = 1

unique_intent = list(set(data['intent']))
print 'Unique intents:'
print len(unique_intent)
for s in unique_intent:
	print s, count_intent[s]

data.to_csv('cleaned_dump.csv')
