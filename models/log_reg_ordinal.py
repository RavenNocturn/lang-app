from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix


# --------------------
# Paths & config
# --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent 

WORDS_PATH = DATA_DIR / "words_cefr.csv"
TEXTS_PATH = DATA_DIR / "cefr_level_texts_with_stats.csv"


# CEFR_RANK = {
#     "A1": 1,
#     "A2": 2,
#     "B1": 3,
#     "B2": 4,
#     "C1": 5,
#     "C2": 6,
# }

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_RANK = {level: i for i, level in enumerate(CEFR_ORDER)}


CUSTOM_STOPWORDS = [
    "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "okay", "yes"
]


# --------------------
# Data loading
# --------------------
def load_data():
    words = pd.read_csv(WORDS_PATH)
    texts = pd.read_csv(TEXTS_PATH)
    return words, texts


def encode_labels(texts: pd.DataFrame):
    label_encoder = LabelEncoder()
    texts["label_encoded"] = label_encoder.fit_transform(texts["label"])
    return texts, label_encoder


# --------------------
# Feature engineering
# --------------------
def build_word_level_dict(words: pd.DataFrame):
    words["level_num"] = words["CEFR"].map(CEFR_RANK)
    return dict(zip(words["headword"], words["level_num"]))


def lexical_cefr_features(texts, word_level_dict):
    features = []

    for text in texts:
        tokens = text.lower().split()
        levels = [
            word_level_dict[w]
            for w in tokens
            if w in word_level_dict
        ]

        if levels:
            features.append([
                np.mean(levels),   # average lexical level
                np.max(levels),    # hardest word
                np.std(levels),    # lexical spread
                len(levels),       # known words
            ])
        else:
            features.append([0, 0, 0, 0])

    return np.array(features)


def build_feature_pipeline(word_level_dict):
    lexical_pipeline = Pipeline([
        ("lexical", FunctionTransformer(
            lambda x: lexical_cefr_features(x, word_level_dict),
            validate=False
        )),
        ("scaler", StandardScaler())
    ])

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=8000,
        min_df=3,
        stop_words=CUSTOM_STOPWORDS
    )

    return FeatureUnion([
        ("tfidf", tfidf),
        ("lexical_cefr", lexical_pipeline),
    ])

# =========================
# Ordinal targets
# =========================

def make_ordinal_targets(y):
    """
    For CEFR levels 0..5, create binary targets:
    >A1, >A2, >B1, >B2, >C1
    """
    ordinal_targets = {}

    for i, level in enumerate(CEFR_ORDER[:-1]):
        ordinal_targets[level] = (y > i).astype(int)

    return ordinal_targets

# =========================
# Ordinal classifier
# =========================

class OrdinalLogisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline
        self.classifiers = {}

    def fit(self, X, y):
        X_features = self.feature_pipeline.fit_transform(X)
        y_ord = make_ordinal_targets(y)

        for level, y_bin in y_ord.items():
            clf = LogisticRegression(
                max_iter=10000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=1  # avoids macOS multiprocessing spam
            )
            clf.fit(X_features, y_bin)
            self.classifiers[level] = clf

        return self

    def predict_proba(self, X):
        X_features = self.feature_pipeline.transform(X)

        threshold_probs = []
        for level in CEFR_ORDER[:-1]:
            p = self.classifiers[level].predict_proba(X_features)[:, 1]
            threshold_probs.append(p)

        threshold_probs = np.vstack(threshold_probs).T

        # Convert threshold probabilities → class probabilities
        class_probs = []
        prev = np.ones(len(X))

        for i in range(len(CEFR_ORDER) - 1):
            p = threshold_probs[:, i]
            class_probs.append(prev * (1 - p))
            prev = prev * p

        class_probs.append(prev)
        return np.vstack(class_probs).T

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# --------------------
# Training & evaluation
# --------------------

def mean_cefr_distance(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def train_and_evaluate(X, y, feature_pipeline, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = OrdinalLogisticClassifier(feature_pipeline)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ))

    print("\nMean CEFR distance:", mean_cefr_distance(y_test, y_pred))


# --------------------
# Main
# --------------------
def main():
    words, texts = load_data()
    texts, label_encoder = encode_labels(texts)

    X = texts["text"]
    y = texts["label_encoded"]

    word_level_dict = build_word_level_dict(words)
    feature_pipeline = build_feature_pipeline(word_level_dict)

    train_and_evaluate(X, y, feature_pipeline, label_encoder)

if __name__ == "__main__":
    main()
