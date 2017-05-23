import json

output_file = 'data/dump.json'
files = []
events = []

for i in range(0, 34 + 1):
    files.append('raw/dump{:02d}.json'.format(i))

for file in files:
    with open(file) as f:
        d = json.load(f)
        events.extend(d['events'])

print 'dumped %d events' % len(events)

with open(output_file, 'w') as f:
    json.dump(events, f)
