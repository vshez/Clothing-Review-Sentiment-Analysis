# Sentiment Analysis
# Copy clothing_reviews.csv into the data folder
# Open PyCharm’s Terminal and install the libraries:
# python -m pip install pandas numpy matplotlib seaborn scikit-learn nltk

from pathlib import Path
import pandas as pd
import re
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

# Locate the CSV relative to this script.
project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "data/clothing_reviews.csv"

# Load the dataset.
df = pd.read_csv(dataset_path)

print(f"Number of reviews: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string(index=False))

# Check data quality.
print("\nMissing values by column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate review texts:", df["Review"].duplicated().sum())

blank_reviews = df["Review"].fillna("").str.strip().eq("").sum()
print("Blank reviews:", blank_reviews)

print("\nRating distribution:")
print(df["Rating"].value_counts().sort_index())

# Step 2: Create sentiment labels and visualize their distribution
# Ratings provide proxy labels for sentiment.
sentiment_mapping = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}

df["Sentiment"] = df["Rating"].map(sentiment_mapping)

# Check that every rating mapped to a sentiment.
if df["Sentiment"].isna().any():
    raise ValueError("Some ratings are missing or outside the 1–5 range.")

sentiment_order = ["Negative", "Neutral", "Positive"]

sentiment_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)

sentiment_summary = pd.DataFrame({
    "Reviews": sentiment_counts,
    "Percentage": (sentiment_counts / len(df) * 100).round(1),
})

print("\nSentiment distribution:")
print(sentiment_summary)

print("\nSample reviews and labels:")
print(df[["Review", "Rating", "Sentiment"]].head().to_string(index=False))

# Create and Save a bar chart
import matplotlib.pyplot as plt

# Save project outputs in a dedicated folder.
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    sentiment_summary.index,
    sentiment_summary["Reviews"],
    color=["#C44E52", "#D9A441", "#55A868"],
)

ax.set_title("Sentiment Distribution of Clothing Reviews")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")

ax.bar_label(
    bars,
    labels=[
        f"{count:,} ({percentage:.1f}%)"
        for count, percentage in zip(
            sentiment_summary["Reviews"],
            sentiment_summary["Percentage"],
        )
    ],
    padding=5,
)

ax.set_ylim(0, sentiment_counts.max() * 1.15)
fig.tight_layout()

fig.savefig(
    output_folder / "sentiment_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

sentiment_summary.to_csv(output_folder / "sentiment_summary.csv")

plt.show()

# Step 3: Clean, tokenize, and lemmatize the reviews

for resource in [
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"[a-z]+")

# Preserve negation and contrast words that can affect sentiment.
preserved_words = {"no", "not", "nor", "never", "neither", "but"}
stop_words = set(stopwords.words("english")) - preserved_words

# Create the preprocessing function:
@lru_cache(maxsize=50000)
def lemmatize_word(word, pos):
    """Cache repeated words to reduce processing time."""
    return lemmatizer.lemmatize(word, pos=pos)

def preprocess_review(text):
    """Clean and normalize one review while preserving negation."""
    text = str(text).lower()

    # Normalize curly apostrophes.
    text = text.replace("’", "'")

    # Expand common contractions before removing punctuation.
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "shan't": "shall not",
        "ain't": "is not",
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"\bi'm\b", "i am", text)

    # Remove URLs and HTML tags.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove possessive endings, then replace punctuation with spaces.
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and tag before removing stop words.
    tokens = tokenizer.tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    pos_mapping = {
        "J": "a",  # Adjective
        "V": "v",  # Verb
        "N": "n",  # Noun
        "R": "r",  # Adverb
    }

    cleaned_tokens = [
        lemmatize_word(word, pos_mapping.get(tag[0], "n"))
        for word, tag in tagged_tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

# Apply preprocessing and save the results
print("\nPreprocessing reviews. This may take several minutes...")

df["Cleaned_Review"] = df["Review"].apply(preprocess_review)

# Check whether preprocessing left any reviews empty.
empty_cleaned = df["Cleaned_Review"].str.strip().eq("")
print("Empty reviews after preprocessing:", empty_cleaned.sum())

# Keep only reviews with usable text.
df = df.loc[~empty_cleaned].copy()

# Display three examples for inspection.
for _, row in df.head(3).iterrows():
    print("\nOriginal:", row["Review"])
    print("Cleaned: ", row["Cleaned_Review"])
    print("Sentiment:", row["Sentiment"])

df.to_csv(
    output_folder / "clothing_reviews_preprocessed.csv",
    index=False,
)

print(f"\nReviews retained: {len(df):,}")
print("Saved: outputs/clothing_reviews_preprocessed.csv")

# Step 4: Split the data into training and test sets
# Keep identical cleaned reviews in the same split.
# Check whether identical cleaned text has conflicting sentiment labels.
labels_per_text = df.groupby("Cleaned_Review")["Sentiment"].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1].index

print("\nCleaned texts with conflicting labels:", len(conflicting_texts))

# Exclude ambiguous groups from this experiment.
model_df = df.loc[
    ~df["Cleaned_Review"].isin(conflicting_texts)
].copy()

# Create one entry per unique cleaned text for splitting.
unique_texts = model_df[
    ["Cleaned_Review", "Sentiment"]
].drop_duplicates(subset="Cleaned_Review")

train_texts, test_texts = train_test_split(
    unique_texts["Cleaned_Review"],
    test_size=0.20,
    random_state=42,
    stratify=unique_texts["Sentiment"],
)

train_df = model_df.loc[
    model_df["Cleaned_Review"].isin(train_texts)
].copy()

test_df = model_df.loc[
    model_df["Cleaned_Review"].isin(test_texts)
].copy()

# Only review text will be used as the model input.
X_train = train_df["Cleaned_Review"]
X_test = test_df["Cleaned_Review"]

y_train = train_df["Sentiment"]
y_test = test_df["Sentiment"]

assert set(X_train).isdisjoint(set(X_test)), (
    "Identical cleaned reviews appear in both sets."
)
# Check the split and save the records:
print(f"\nTraining reviews: {len(train_df):,}")
print(f"Test reviews: {len(test_df):,}")

split_summary = pd.DataFrame({
    "Training count": y_train.value_counts(),
    "Test count": y_test.value_counts(),
    "Training %": y_train.value_counts(normalize=True).mul(100),
    "Test %": y_test.value_counts(normalize=True).mul(100),
}).reindex(sentiment_order).round(2)

print("\nSentiment distribution by split:")
print(split_summary)

train_df.to_csv(output_folder / "train_reviews.csv", index=False)
test_df.to_csv(output_folder / "test_reviews.csv", index=False)
split_summary.to_csv(output_folder / "split_summary.csv")

print("\nTraining and test files saved.")

# Step 5: Convert reviews into TF-IDF features
# Define settings that we will also use during model tuning.
tfidf_settings = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

tfidf = TfidfVectorizer(**tfidf_settings)

# Learn vocabulary and weights from training reviews only.
X_train_tfidf = tfidf.fit_transform(X_train)

# Apply the same vocabulary and weights to test reviews.
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF feature extraction completed.")
print("Training feature matrix:", X_train_tfidf.shape)
print("Test feature matrix:", X_test_tfidf.shape)
print("Number of features:", len(tfidf.get_feature_names_out()))

feature_names = tfidf.get_feature_names_out()
first_review_vector = X_train_tfidf.getrow(0)

review_features = pd.DataFrame({
    "Feature": feature_names[first_review_vector.indices],
    "TF-IDF weight": first_review_vector.data,
}).sort_values("TF-IDF weight", ascending=False)

print("\nFirst training review:")
print(X_train.iloc[0])

print("\nTop 10 TF-IDF features in this review:")
print(review_features.head(10).to_string(index=False))

review_features.head(10).to_csv(
    output_folder / "sample_tfidf_features.csv",
    index=False,
)
# Step 6: Train and tune Naive Bayes and SVM
# Identical cleaned reviews receive the same group ID.
training_groups = pd.factorize(X_train)[0]

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

# Reuse exactly the same folds for both models.
cv_splits = list(
    cv.split(X_train, y_train, groups=training_groups)
)

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_settings)),
    ("classifier", MultinomialNB()),
])

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_settings)),
    ("classifier", LinearSVC(
        dual=False,
        max_iter=10000,
        random_state=42,
    )),
])
# Tune Naive Bayes:
nb_search = GridSearchCV(
    estimator=nb_pipeline,
    param_grid={
        "classifier__alpha": [0.1, 0.5, 1.0],
    },
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=1,
    verbose=1,
    refit=True,
    error_score="raise",
)

print("\nTuning Naive Bayes...")
nb_search.fit(X_train, y_train)

print("Naive Bayes best parameters:", nb_search.best_params_)
print(f"Naive Bayes cross-validation macro-F1: "
      f"{nb_search.best_score_:.4f}")

# Tune the linear SVM
svm_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid={
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=1,
    verbose=1,
    refit=True,
    error_score="raise",
)

print("\nTuning linear SVM...")
svm_search.fit(X_train, y_train)

print("SVM best parameters:", svm_search.best_params_)
print(f"SVM cross-validation macro-F1: "
      f"{svm_search.best_score_:.4f}")

# Each best pipeline is already fitted on all training reviews.
best_nb = nb_search.best_estimator_
best_svm = svm_search.best_estimator_

# Save the tuning results:
pd.DataFrame(nb_search.cv_results_).to_csv(
    output_folder / "naive_bayes_cv_results.csv",
    index=False,
)

pd.DataFrame(svm_search.cv_results_).to_csv(
    output_folder / "svm_cv_results.csv",
    index=False,
)

print("\nModel tuning completed. Results saved.")

# Step 7: Evaluate both models on the test set
models = {
    "Naive Bayes": best_nb,
    "Linear SVM": best_svm,
}

evaluation_results = []
test_predictions = {}

for model_name, model in models.items():
    # Pipelines apply their fitted TF-IDF transformation automatically.
    predictions = model.predict(X_test)
    test_predictions[model_name] = predictions

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=sentiment_order,
        average="macro",
        zero_division=0,
    )

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    })

    print(f"\n{model_name} — Test Classification Report")
    print(classification_report(
        y_test,
        predictions,
        labels=sentiment_order,
        digits=4,
        zero_division=0,
    ))

    report = classification_report(
        y_test,
        predictions,
        labels=sentiment_order,
        output_dict=True,
        zero_division=0,
    )

    file_prefix = model_name.lower().replace(" ", "_")

    pd.DataFrame(report).T.to_csv(
        output_folder / f"{file_prefix}_classification_report.csv"
    )

    # Create Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_models = [
        ("Naive Bayes", best_nb),
        ("Linear SVM", best_svm),
    ]

    for ax, (model_name, model) in zip(axes, plot_models):
        predictions = model.predict(X_test)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=sentiment_order,
            display_labels=sentiment_order,
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=ax,
        )

        ax.set_title(f"{model_name} — Test Set")

    fig.tight_layout()

    fig.savefig(
        output_folder / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

# Save the comparision and individual predictions
comparison_df = pd.DataFrame(evaluation_results)

print("\nTest Performance Comparison:")
print(comparison_df.round(4).to_string(index=False))

comparison_df.to_csv(
    output_folder / "model_comparison.csv",
    index=False,
)

prediction_output = test_df[
    ["id", "Review", "Rating", "Sentiment"]
].copy()

for model_name, predictions in test_predictions.items():
    prediction_output[f"{model_name} Prediction"] = predictions

prediction_output.to_csv(
    output_folder / "test_predictions.csv",
    index=False,
)

# Select the preferred model using training cross-validation results.
selected_model_name = (
    "Linear SVM"
    if svm_search.best_score_ > nb_search.best_score_
    else "Naive Bayes"
)

print("\nModel selected by cross-validation:", selected_model_name)
print("Evaluation files saved.")

# Step 8: Inspect classification errors and interpret the results
# Regenerate predictions for both models.
for model_name, model in models.items():
    predictions = model.predict(X_test)

    reviewed_results = test_df[
        ["id", "Review", "Rating", "Sentiment"]
    ].copy()

    reviewed_results["Predicted Sentiment"] = predictions

    errors = reviewed_results.loc[
        reviewed_results["Sentiment"]
        != reviewed_results["Predicted Sentiment"]
    ]

    print(f"\n{model_name}: {len(errors):,} misclassified reviews")

    # Show up to five errors.
    for _, row in errors.head(5).iterrows():
        print("\nReview:", row["Review"])
        print("Rating-derived label:", row["Sentiment"])
        print("Predicted label:", row["Predicted Sentiment"])

    file_prefix = model_name.lower().replace(" ", "_")

    errors.to_csv(
        output_folder / f"{file_prefix}_misclassified_reviews.csv",
        index=False,
    )

# Inspect examples from every sentiment class.
print("\nSample reviews by sentiment:")

for sentiment in sentiment_order:
    sample = test_df.loc[
        test_df["Sentiment"] == sentiment
    ].sample(n=3, random_state=42)

    print(f"\n--- {sentiment} ---")

    for review in sample["Review"]:
        print(review)
        print()

# Step 9: Identify influential words and phrases
# Extract the vectorizer and classifier from the fitted SVM pipeline.
svm_vectorizer = best_svm.named_steps["tfidf"]
svm_classifier = best_svm.named_steps["classifier"]

svm_feature_names = svm_vectorizer.get_feature_names_out()

influential_features = []

for class_index, sentiment in enumerate(svm_classifier.classes_):
    coefficients = svm_classifier.coef_[class_index]

    # Select the 10 largest positive coefficients for this class.
    positive_indices = (coefficients > 0).nonzero()[0]
    top_indices = positive_indices[
        coefficients[positive_indices].argsort()[-10:][::-1]
    ]

    print(f"\nTop features associated with {sentiment}:")

    for rank, feature_index in enumerate(top_indices, start=1):
        feature = svm_feature_names[feature_index]
        coefficient = coefficients[feature_index]

        print(f"{feature}: {coefficient:.4f}")

        influential_features.append({
            "Sentiment": sentiment,
            "Rank": rank,
            "Feature": feature,
            "Coefficient": coefficient,
        })

feature_summary = pd.DataFrame(influential_features)

feature_summary.to_csv(
    output_folder / "svm_influential_features.csv",
    index=False,
)

print("\nSaved: outputs/svm_influential_features.csv")
