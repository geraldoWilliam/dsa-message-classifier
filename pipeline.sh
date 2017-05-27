#!/bin/bash
echo -e "\e[1;31m[*] Preprocessing\e[0m"
cd preprocessing
python convert_intent.py
python clean_data.py

echo -e "\n"
echo -e "\e[1;31m[*] Feature Extraction\e[0m"
cd ../feature_extraction
python extract_feature.py

echo -e "\n"
echo -e "\e[1;31m[*] Feature Selection\e[0m"
cd ../feature_selection
python select_feature.py

echo -e "\n"
echo -e "\e[1;31m[*] Training / Testing\e[0m"
cd ../training_testing
python training.py
# python genreport.py
python assess.py
