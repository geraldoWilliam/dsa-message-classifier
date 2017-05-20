import pandas as pd
import constants

def normalize_text(s):
    s = s.lower()
    s = "".join([c for c in s if c.isalpha() or c==' '])
    return s

def main():
    data = pd.read_csv('../preprocessing/cleaned_dump.csv', index_col=0)
    data.to_csv(constants.DATA_FOLDER + constants.DATASET_FILENAME, index = False)

    for i, row in data.iterrows():
        data.loc[i, 'text'] = normalize_text(str(data.loc[i, 'text']))
    data.to_csv(constants.DATA_FOLDER + constants.NORMALIZED_DATASET_FILENAME, index = False)


if __name__ == '__main__':
    main()
