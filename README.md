# Clothing-Review-Sentiment-Analysis
Customer review sentiment analysis using Python, NLP preprocessing, and TF-IDF features. Compares Naive Bayes and linear SVM models to classify clothing reviews as positive, negative, or neutral, with cross-validation, hyperparameter tuning, and performance evaluation.

Dataset
Selected source: https://www.kaggle.com/datasets/vashez/clothing-review-dataset
The analyzed CSV contains 50,000 rows and 15 columns. Only Review is used as the model input. Ratings supply proxy sentiment labels:
- Ratings 1–2: negative (16,500 reviews).
- Rating 3: neutral (12,000 reviews).
- Ratings 4–5: positive (21,500 reviews).

Setup in PyCharm
1. Open your project and select its Python interpreter/virtual environment.
2. Download sentiment_analysis.py and clothing-review-dataset.csv in the project.
3. Open PyCharm's Terminal and install dependencies:
python -m pip install pandas numpy scikit-learn nltk matplotlib seaborn
4. Download the NLTK resources:
python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng
5. Check the script's pd.read_csv(...) path. It must point to clothing-review-dataset.csv. For example, if the CSV is alongside the script, use pd.read_csv("clothing_reviews.csv") else if inside data folder, project_folder / "data/clothing_reviews.csv"
6. Set the run configuration's working directory to the project folder.
How to Run
Right-click sentiment_analysis.py and select Run, or execute:
python sentiment_analysis.py
Read the console metrics and check the output locations configured in your script for the generated CSVs and figure.
Method
1. Inspect missing values, blank reviews, duplicates, and sentiment counts.
2. Lowercase text, handle common contractions, remove URLs/HTML, replace punctuation with spaces, tokenize alphabetic words, and apply POS-aware WordNet lemmatization.
3. Remove stop words while preserving no, not, nor, never, neither, and but.
4. Group identical cleaned review texts; exclude groups with conflicting labels. Split unique groups using a stratified 80/20 allocation and random seed 42, then assign their rows to the same partition.
5. Use TF-IDF with unigrams/bigrams, max_features=10000, min_df=2, max_df=0.95, and sublinear_tf=True.
6. Fit TF-IDF within each model pipeline to avoid learning the vocabulary from validation or test reviews.
7. Tune with three-fold StratifiedGroupKFold (shuffle=True, random_state=42), grouped by cleaned review text, using macro F1.
8. Evaluate on the held-out test set using accuracy, precision, recall, macro F1, and confusion matrices.
Parameter grids
- MultinomialNB: alpha = 0.1, 0.5, 1.0.
- LinearSVC: C = 0.1, 1, 10; class_weight = None or balanced; dual=False; max_iter=10000; random_state=42.
- GridSearchCV: scoring="f1_macro", n_jobs=1, refit=True.
Observed results
Both models achieved accuracy and macro precision, recall, and F1 of 1.000 on the 10,000-review test set, with no misclassifications. Test support was 3,300 negative, 2,400 neutral, and 4,300 positive reviews.
All tuning candidates tied with cross-validation macro F1 of 1.000. The first grid candidates were selected: Naive Bayes alpha=0.1 and SVM C=0.1, class_weight=None. The workflow retained Naive Bayes on the cross-model tie.
Perfect scores apply only to this experiment. Repeated language patterns may make this dataset unusually easy; exact-text grouping does not eliminate similar templates. Rating labels may also differ from the sentiment expressed in a review. Validate on independently sourced, varied reviews before business use.
Troubleshooting
- CSV not found: check the filename, read_csv path, and working directory.
- NLTK LookupError: download the resource named in the error using python -m nltk.downloader RESOURCE_NAME.
- Blank figure: save the populated figure before calling plt.show() or closing it.
- Results differ: check the exact CSV, preprocessing, group split, seeds, and package versions.

