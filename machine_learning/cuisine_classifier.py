import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# STEP 1: LOAD AND SPLIT THE DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("CUISINE CLASSIFIER — Metadata Imputation via NLP")
print("=" * 60)
print()

print("Loading data...")
reviews = pd.read_csv('../merged/master_reviews.csv')

reviews = reviews[reviews['review_text_cleaned'].notna()]
reviews = reviews[reviews['review_text_cleaned'].str.strip() != '']

print(f"Total reviews after cleaning: {len(reviews)}")

known   = reviews[reviews['cuisine_primary'] != 'Unknown'].copy()
unknown = reviews[reviews['cuisine_primary'] == 'Unknown'].copy()

print(f"Reviews with known cuisine (training pool): {len(known)}")
print(f"Reviews with unknown cuisine (to predict):  {len(unknown)}")
print()


# ─────────────────────────────────────────────
# STEP 2: COLLAPSE RARE CUISINE CLASSES INTO "Other"
# ─────────────────────────────────────────────

MIN_REVIEWS_PER_CLASS = 50

cuisine_counts = known['cuisine_primary'].value_counts()
rare_cuisines  = cuisine_counts[cuisine_counts < MIN_REVIEWS_PER_CLASS].index

print(f"Rare cuisines collapsed into 'Other': {list(rare_cuisines)}")

known['cuisine_label'] = known['cuisine_primary'].apply(
    lambda c: 'Other' if c in rare_cuisines else c
)

print()
print("Final class distribution:")
print(known['cuisine_label'].value_counts().to_string())
print(f"\nTotal classes: {known['cuisine_label'].nunique()}")
print()


# ─────────────────────────────────────────────
# STEP 3: ENCODE LABELS
# ─────────────────────────────────────────────

le = LabelEncoder()
y  = le.fit_transform(known['cuisine_label'])
X  = known['review_text_cleaned'].values


# ─────────────────────────────────────────────
# STEP 4: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print()


# ─────────────────────────────────────────────
# STEP 5: BUILD PIPELINES
# ─────────────────────────────────────────────

tfidf_params = dict(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    stop_words='english'
)

models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}


# ─────────────────────────────────────────────
# STEP 6: TRAIN AND EVALUATE ALL THREE MODELS
# ─────────────────────────────────────────────
# We evaluate on ALL classes here (not just top 10) — this is important
# because best model selection should be based on overall performance,
# not just performance on the popular classes.
# The trimming to top 10 only happens in the confusion matrix visual later.

results     = {}
predictions = {}

print("=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

for model_name, pipeline in models.items():
    print(f"\n--- {model_name} ---")
    print(f"  Training...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    results[model_name]     = {'accuracy': acc, 'report': report, 'y_pred': y_pred}
    predictions[model_name] = y_pred

    print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))


# ─────────────────────────────────────────────
# STEP 7: PICK THE BEST MODEL
# ─────────────────────────────────────────────
# Selection is based on weighted F1 across ALL classes.
# This is fairer than accuracy because it accounts for class imbalance —
# a model can't just be great at Lebanese and ignore everything else.

best_model_name = max(
    results,
    key=lambda m: results[m]['report']['weighted avg']['f1-score']
)
best_pipeline = models[best_model_name]

print("=" * 60)
print(f"BEST MODEL: {best_model_name}")
print(f"  Accuracy:    {results[best_model_name]['accuracy']*100:.1f}%")
print(f"  Weighted F1: {results[best_model_name]['report']['weighted avg']['f1-score']:.4f}")
print("=" * 60)
print()

# STEP 8: SAVE MODEL COMPARISON CSV
# One row per model. The dashboard reads this to display a comparison table
# or bar chart without needing to re-run training.

comparison_rows = []
for model_name, res in results.items():
    comparison_rows.append({
        'model':        model_name,
        'accuracy':     round(res['accuracy'], 4),
        'weighted_f1':  round(res['report']['weighted avg']['f1-score'], 4),
        'weighted_precision': round(res['report']['weighted avg']['precision'], 4),
        'weighted_recall':    round(res['report']['weighted avg']['recall'], 4),
        'is_best':      model_name == best_model_name
    })

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv('ml_model_comparison.csv', index=False)
print("Saved: ml_model_comparison.csv")


# STEP 10: CONFUSION MATRIX — TOP 10 CLASSES ONLY

top10_cuisines = (
    known['cuisine_label']
    .value_counts()
    .head(10)
    .index
    .tolist()
)

top10_indices = [list(le.classes_).index(c) for c in top10_cuisines if c in le.classes_]

# Filter test set: only keep rows where the true label is in top 10
mask_top10 = np.isin(y_test, top10_indices)
y_test_top10 = y_test[mask_top10]
y_pred_top10 = predictions[best_model_name][mask_top10]

cm = confusion_matrix(y_test_top10, y_pred_top10, labels=top10_indices, normalize='true')

top10_labels = [le.classes_[i] for i in top10_indices]

# STEP 11: PREDICT UNKNOWN CUISINES
print("=" * 60)
print("PREDICTING CUISINE FOR UNKNOWN REVIEWS")

X_unknown = unknown['review_text_cleaned'].values

predicted_indices = best_pipeline.predict(X_unknown)
predicted_proba = best_pipeline.predict_proba(X_unknown)
predicted_cuisines = le.inverse_transform(predicted_indices)
predicted_confidences = predicted_proba.max(axis=1)

unknown = unknown.copy()
unknown['cuisine_predicted'] = predicted_cuisines
unknown['prediction_confidence'] = predicted_confidences.round(3)

LOW_CONFIDENCE_THRESHOLD = 0.30
low_conf_count = (unknown['prediction_confidence'] < LOW_CONFIDENCE_THRESHOLD).sum()

print(f"Predictions made: {len(unknown)}")
print("Predicted cuisine distribution:")
print(unknown['cuisine_predicted'].value_counts().to_string())
print("Confidence score distribution:")
print(unknown['prediction_confidence'].describe().round(3))
print(f"\nLow-confidence predictions (< {LOW_CONFIDENCE_THRESHOLD}): {low_conf_count} ({low_conf_count/len(unknown)*100:.1f}%)")

# STEP 12: SAVE PREDICTED DISTRIBUTION CSV
# The dashboard uses this to show "how many reviews were recovered per cuisine"
df_pred_dist = (
    unknown.groupby('cuisine_predicted')
    .agg(
        predicted_count=('cuisine_predicted', 'count'),
        avg_confidence=('prediction_confidence', 'mean'),
        high_conf_count=('prediction_confidence', lambda x: (x >= 0.5).sum())
    )
    .round(3)
    .sort_values('predicted_count', ascending=False)
    .reset_index()
    .rename(columns={'cuisine_predicted': 'cuisine'})
)

df_pred_dist.to_csv('ml_predicted_distribution.csv', index=False)
print("Saved: ml_predicted_distribution.csv")

# STEP 13: BUILD THE ENRICHED MASTER REVIEWS FILE
# known reviews   ->   cuisine_source = 'original'
# unknown reviews   ->   cuisine_primary filled with prediction, cuisine_source = 'predicted'

known_out = known.copy()
known_out['cuisine_source']          = 'original'
known_out['prediction_confidence']   = np.nan   # no confidence for original labels

unknown_out = unknown.copy()
unknown_out['cuisine_primary']       = unknown_out['cuisine_predicted']
unknown_out['cuisine_source']        = 'predicted'
unknown_out = unknown_out.drop(columns=['cuisine_predicted'])

master_enriched = pd.concat([known_out, unknown_out], ignore_index=True)

master_enriched.to_csv('../master_reviews_enriched.csv', index=False)
print("Saved: master_reviews_enriched.csv")
print()
print(f"Original labels:  {(master_enriched['cuisine_source'] == 'original').sum()}")
print(f"Predicted labels: {(master_enriched['cuisine_source'] == 'predicted').sum()}")
print(f"Total rows:       {len(master_enriched)}")


# STEP 14: SAVE SUMMARY JSON

# Per-class F1 scores from the best model (all classes, for the detailed table)
best_report = results[best_model_name]['report']
per_class_f1 = {
    cls: round(best_report[cls]['f1-score'], 3)
    for cls in le.classes_
    if cls in best_report
}

# Confusion pairs: which cuisines get confused most with each other
cm_full = confusion_matrix(y_test, predictions[best_model_name])
cm_full_norm = cm_full.astype(float) / cm_full.sum(axis=1, keepdims=True)

# Find the top 5 off-diagonal confusion pairs
confusion_pairs = []
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm_full_norm[i, j] > 0.05:   # only meaningful bleed > 5%
            confusion_pairs.append({
                'actual':     le.classes_[i],
                'predicted':  le.classes_[j],
                'rate':       round(float(cm_full_norm[i, j]), 3)
            })

confusion_pairs = sorted(confusion_pairs, key=lambda x: x['rate'], reverse=True)[:5]

summary = {
    # Overall pipeline stats
    'total_reviews_loaded':     int(len(reviews)),
    'known_cuisine_reviews':    int(len(known)),
    'unknown_cuisine_reviews':  int(len(unknown)),

    # Model comparison (all three)
    'models_evaluated': {
        m: {
            'accuracy':    round(results[m]['accuracy'], 4),
            'weighted_f1': round(results[m]['report']['weighted avg']['f1-score'], 4)
        }
        for m in results
    },

    # Best model
    'best_model':best_model_name,
    'best_model_accuracy':round(results[best_model_name]['accuracy'], 4),
    'best_model_f1': round(results[best_model_name]['report']['weighted avg']['f1-score'], 4),

    # Confusion matrix scope note
    'confusion_matrix_note': f'Confusion matrix shown for top 10 cuisines only (out of {len(le.classes_)} total classes)',

    # Top confused pairs (for the dashboard callout card)
    'top_confusion_pairs': confusion_pairs,

    # Prediction outcomes
    'predictions_made':        int(len(unknown)),
    'low_confidence_count':    int(low_conf_count),
    'low_confidence_pct':      round(low_conf_count / len(unknown) * 100, 1),
    'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
    'avg_prediction_confidence': round(float(predicted_confidences.mean()), 3),

    # Per-class F1 (for detailed breakdown table in dashboard)
    'per_class_f1': per_class_f1,

    # Enriched file stats
    'enriched_reviews_total': int(len(master_enriched)),
    'enriched_original_labels': int((master_enriched['cuisine_source'] == 'original').sum()),
    'enriched_predicted_labels':int((master_enriched['cuisine_source'] == 'predicted').sum()),
}

with open('ml_cuisine_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved: ml_cuisine_summary.json")
print("=" * 60)
print("SUMMARY")
print(f"Best model:               {best_model_name}")
print(f"Accuracy:                 {summary['best_model_accuracy']*100:.1f}%")
print(f"Weighted F1:              {summary['best_model_f1']:.4f}")
print(f"Reviews predicted:        {summary['predictions_made']}")
print(f"Low-confidence preds:     {summary['low_confidence_count']} ({summary['low_confidence_pct']}%)")
print(f"Avg confidence:           {summary['avg_prediction_confidence']:.3f}")
print(f"Enriched dataset size:    {summary['enriched_reviews_total']}")
print()
print("Output files:")
print("  ml_model_comparison.csv       ← model scores table")
print("  ml_predicted_distribution.csv ← predicted cuisine counts + confidence")
print("  ml_cuisine_summary.json       ← all summary stats for dashboard")
print("  ../master_reviews_enriched.csv ← enriched reviews (plug into NLP)")