import pandas as pd

input_file = 'data/converted_curated_dump.csv'
output_file = 'output/dataset.csv'
balanced_output_file = 'output/balanced_dataset.csv'

stopwords = open('stopwords_id.txt', 'rb').read().splitlines()

abbr_tuples = [ line.split(' = ') for line in open('abbreviations_id.txt', 'rb').read().splitlines() ]
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
    # s = remove_stopwords(s)
    return s

def get_balanced_data(data):
    unique_intents = data['intent'].unique()
    unique_data = {elem : pd.DataFrame for elem in unique_intents}

    for key in unique_data.keys():
        unique_data[key] = data[:][data['intent'] == key]

    min_row = min([unique_data[key].shape[0] for key in unique_data.keys()])

    balanced_data = pd.DataFrame(columns = data.columns)
    for key in unique_data.keys():
        rows = unique_data[key].sample(n=min_row)
        balanced_data = balanced_data.append(rows)

    balanced_data = balanced_data.sample(frac=1)
    return balanced_data

def main():
    data = pd.read_csv(input_file, index_col=0)
    for i,row in data.iterrows():
        data.loc[i,'text'] = normalize_text(str(data.loc[i,'text']))
    data.to_csv(output_file, index=False)

    balanced_data = get_balanced_data(data)
    balanced_data.to_csv(balanced_output_file, index=False)

if __name__ == '__main__':
    main()
