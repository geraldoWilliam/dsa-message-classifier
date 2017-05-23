import csv
import json
import datetime

input_file = 'data/dump.json'
output_file = 'data/dump.csv'

def getTime(t):
    return int(datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%s'))

with open(input_file) as f:
    d = json.load(f)

extracted_data = []

for row in d:
    intent = None
    for entity in row['conclusion']['outcomes'][0]['entities']:
        if entity['name'] == 'intent':
            intent = entity['value']

    extracted_data.append({
        "text": row['conclusion']['_text'],
        "intent": intent,
        "datetime": getTime(row['at'])
    })


# Write CSV
f = csv.writer(open(output_file, "wb+"))
f.writerow(["id", "text", "intent", "datetime"])
for index, row in enumerate(extracted_data):
    f.writerow([index,
                row["text"].encode("utf-8"),
                row["intent"],
                row["datetime"]])
