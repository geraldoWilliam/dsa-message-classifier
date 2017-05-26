# dsa-message-classifier

# Steps

## Preprocessing

Raw dumps are available at `preprocessing/raw/` directory.

- Combine dumps of messages
- Convert into CSV, containing only necessary informations

```
cd preprocessing
python combine_dumps.py
python make_csv.py
```

- Do manual curation to data labels, save them to `data/curated_dump.csv`
- Run a script just to make sure

```
python convert_intent.py
```

- Run data cleaning script
	- toLowercase text
	- remove symbols
	- remove numbers
	- normalize abbreviations
	- remove stopwords

```
python clean_data.py
```

### Output:

- `dataset.csv`
- `balanced_dataset.csv`

Output(s) are available at `preprocessing/output/`



## Feature Extraction

- Extract feature
```
cd feature_extraction
python extract_feature.py
```

### Output:

- `features_freq.csv`
- `features_bool.csv`
- `features_freq_balanced.csv`
- `features_bool_balanced.csv`

- `vect_freq.pkl`
- `vect_bool.pkl`
- `vect_freq_balanced.pkl`
- `vect_bool_balanced.pkl`

Output(s) are available at `preprocessing/output/`


## Feature Selection

- Select K best features with chi-squared univariate test

```
cd feature_selection
python select_feature.py
```

### Output:

- `features_freq_selected.csv`
- `features_bool_selected.csv`
- `features_freq_balanced_selected.csv`
- `features_bool_balanced_selected.csv`
- `selected_features_*.txt`
- `vect_*_selected.pkl`

Output(s) are available at `preprocessing/output/`

## Model Building / Training

- Do train your model
```
cd training_testing
python training.py
```

### Output:

- `***model***.pkl`

Output(s) are available at `preprocessing/output/`



## Model Assessments / Testing

- Generate training cross-validation reports
- Generate performance reports on testing data

```
cd training_testing
python genreport.py
python assess.py
```


# Result and Analysis

## Exploratory Data Visualization

- Message length distribution
- Frequency distribution of words
```
cd exploratory
python preprocessor.py // generate cleaned data to be processed
python statistics.py
```
The output (visualization and statistics) are available in `exploratory/output`

## Running Web Graphical Interface

- Run Flask-based web interface in localhost
```
cd interface
python app.py
```
Open localhost:5000
