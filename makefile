clean:
    rm --force preprocessing/*.csv
    rm --force preprocessing/intent_summary.txt
    rm --force feature_extraction/*.csv
    rm --force feature_extraction/balanced_label feature_extraction/label
    rm --force training/*.p
    rm --force training/metadata
preprocess:
    python preprocessing/make_csv.py
    python preprocessing/convert_intent.py
generate_feature:
    python feature_extraction/extract_feature.py
train:
    python training/training.py

help:
    @echo "    clean"
    @echo "        clean all artifacts."
    @echo "    preprocess"
    @echo "        Build process artifacts."
    @echo "    generate_feature"
    @echo "        Generate Feature."
    @echo "    train"
    @echo "        Do training."

