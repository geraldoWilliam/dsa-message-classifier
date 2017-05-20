import os
import pandas as pd
import constants
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def calculateWordCounts(labels, occurences):
    wordCounts = occurences.sum(axis=0)
    counts = {}
    for i in range(len(labels)):
        counts[labels[i]] = wordCounts.item(i)
    return counts

def displayWordCountChart(word_counts, filename):
    indexes = np.array([str(s) for s in word_counts.index])
    values = np.array([int(x) for x in word_counts[0]])
    plt.close()
    plt.title(filename)
    plt.bar(np.arange(30), values[0:30])
    plt.xticks(np.arange(30) + 0.75 * 0.5, indexes, rotation='vertical')
    plt.draw()
    plt.savefig(constants.OUTPUT_FOLDER + filename)

def getWordCounts(data, label=None):
    if label is not None:
        data = data[data['intent'] == label]

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    word_occurences = vectorizer.fit_transform(data['text'].values.astype('U'))
    word_counts_dict = calculateWordCounts(vectorizer.get_feature_names(), word_occurences)
    word_counts_df = pd.DataFrame.from_dict(word_counts_dict, orient='index')
    word_counts_df_sorted = word_counts_df.sort_values(0, ascending=False)
    return word_counts_df_sorted


def getLengthSeries(data):
    s = []
    for text in data['text']:
        length = len(text) if type(text) is str else 0
        s.append(length)
    return pd.Series(s)

def countTextLength(data):
    lengthCounter = {}
    for text in data['text']:
        length = len(text) if type(text) is str else 0
        if length in lengthCounter:
            lengthCounter[length] += 1
        else:
            lengthCounter[length] = 1

    length = range(max(lengthCounter.keys()) + 1)
    count = []
    for i in length:
        count.append(lengthCounter[i] if i in lengthCounter else 0)

    return pd.DataFrame({ 'count': count }, index=length)

def calculateLengthStatistics(data):
    series = getLengthSeries(data)
    return series.describe()

def plotLengthStats(data, filename):
    series = getLengthSeries(data)
    series_counts = series.value_counts(sort=False).sort_index()

    plt.close()
    plt.title(filename)
    fig, ax = plt.subplots()
    series_counts.plot(ax=ax, kind='bar')
    plt.draw()
    plt.savefig(constants.OUTPUT_FOLDER + filename)

def main():
    data = pd.read_csv(constants.DATA_FOLDER + constants.DATASET_FILENAME, index_col=0)
    normalized_data = pd.read_csv(constants.DATA_FOLDER + constants.NORMALIZED_DATASET_FILENAME, index_col=0)

    ## Related to Message Length
    data_lengthCounts = countTextLength(data)
    n_data_lengthCounts = countTextLength(normalized_data)
    data_lengthStats = calculateLengthStatistics(data)
    n_data_lengthStats = calculateLengthStatistics(normalized_data)

    try:
        os.stat(constants.OUTPUT_FOLDER)
    except:
        os.mkdir(constants.OUTPUT_FOLDER)

    data_lengthCounts.to_csv(constants.OUTPUT_FOLDER + 'text_length_counts.csv', index_label='Length')
    n_data_lengthCounts.to_csv(constants.OUTPUT_FOLDER + 'normalized_text_length_counts.csv', index_label='Length')
    data_lengthStats.to_csv(constants.OUTPUT_FOLDER + 'text_length_stats.csv')
    n_data_lengthStats.to_csv(constants.OUTPUT_FOLDER + 'normalized_text_length_stats.csv', index_label='Length')
    plotLengthStats(data, 'text_length_statistic_chart.png')
    plotLengthStats(normalized_data, 'normalized_text_length_statistic_chart.png')


    ## Related to Word in messages
    labels = data['intent'].unique()

    data_word_counts = getWordCounts(data)
    data_word_counts_per_label = { label: getWordCounts(data, label) for label in labels }

    n_data_word_counts = getWordCounts(normalized_data)
    n_data_word_counts_per_label = { label: getWordCounts(normalized_data, label) for label in labels }

    try:
        os.stat(constants.OUTPUT_FOLDER)
    except:
        os.mkdir(constants.OUTPUT_FOLDER)

    data_word_counts.to_csv(constants.OUTPUT_FOLDER + 'data_word_counts.csv', index_label='Word')
    n_data_word_counts.to_csv(constants.OUTPUT_FOLDER + 'normalized_data_word_counts.csv', index_label='Word')
    for label in labels:
        data_word_counts_per_label[label].to_csv(constants.OUTPUT_FOLDER + 'data_word_counts_' + label + '.csv', index_label='Word')
        n_data_word_counts_per_label[label].to_csv(constants.OUTPUT_FOLDER + 'normalized_data_word_counts_' + label + '.csv', index_label='Word')

    displayWordCountChart(getWordCounts(data), 'word-count-all.png')
    displayWordCountChart(getWordCounts(normalized_data), 'normalized_word-count-all.png')
    for label in labels:
        displayWordCountChart(getWordCounts(data, label), 'word-count-' + label + '.png')
        displayWordCountChart(getWordCounts(normalized_data, label), 'normalized_word-count-' + label + '.png')


if __name__ == '__main__':
    main()
