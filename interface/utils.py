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
