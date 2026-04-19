import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# LOAD DATA

print("=" * 60)
print("CUISINE CLASSIFIER — Metadata Imputation via NLP")

reviews = pd.read_csv('../merged/master_reviews.csv')
reviews = reviews[reviews['review_text_cleaned'].notna()]
reviews = reviews[reviews['review_text_cleaned'].str.strip() != '']

known   = reviews[reviews['cuisine_primary'] != 'Unknown'].copy()
unknown = reviews[reviews['cuisine_primary'] == 'Unknown'].copy()

print(f"Total reviews:          {len(reviews)}")
print(f"Known cuisine reviews:  {len(known)}")
print(f"Unknown cuisine reviews:{len(unknown)}")

# COLLAPSE RARE CUISINES INTO "Other"
MIN_REVIEWS_PER_CLASS = 50

cuisine_counts = known['cuisine_primary'].value_counts()
rare_cuisines  = cuisine_counts[cuisine_counts < MIN_REVIEWS_PER_CLASS].index

known['cuisine_label'] = known['cuisine_primary'].apply(
    lambda c: 'Other' if c in rare_cuisines else c
)

print(f"\nRare cuisines collapsed into 'Other': {list(rare_cuisines)}")
print(f"Final class count: {known['cuisine_label'].nunique()}")
print(known['cuisine_label'].value_counts().to_string())

# ENCODE LABELS AND SPLIT
le = LabelEncoder()
y  = le.fit_transform(known['cuisine_label'])
X  = known['review_text_cleaned'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# SHARED TFIDF PARAMS (used in all phases)
tfidf_params = dict(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    stop_words='english'
)

# PHASE 1 — comapring ml models without balancing
print("PHASE 1 — Model Comparison (No Balancing)")

phase1_models = {
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

phase1_results = {}

for model_name, pipeline in phase1_models.items():
    print(f"\n  Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    phase1_results[model_name] = {
        'accuracy': acc,
        'report':   report,
        'y_pred':   y_pred
    }

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

# Pick the best model by weighted F1
best_model_name = max(
    phase1_results,
    key=lambda m: phase1_results[m]['report']['weighted avg']['f1-score']
)

print(f"\n  Best model from Phase 1: {best_model_name}")

# Save Phase 1 comparison CSV
df_phase1 = pd.DataFrame([
    {
        'model':               m,
        'accuracy':            round(r['accuracy'], 4),
        'weighted_f1':         round(r['report']['weighted avg']['f1-score'], 4),
        'weighted_precision':  round(r['report']['weighted avg']['precision'], 4),
        'weighted_recall':     round(r['report']['weighted avg']['recall'], 4),
        'is_best':             m == best_model_name
    }
    for m, r in phase1_results.items()
])
df_phase1.to_csv('ml_model_comparison.csv', index=False)
print("  Saved: ml_model_comparison.csv")

# PHASE 2 — comapring the balancing strategies using the best model from phase 1
# Uses the best model from Phase 1 with 5-Fold Stratified CV
print(f"PHASE 2 — Balancing Strategy Comparison (using {best_model_name})")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression is used here because it supports all 3 strategies cleanly.
# (Naive Bayes doesn't support class_weight, so LR is the fair reference.)
balancing_pipelines = {
    'Baseline (No Balancing)': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Class Weights': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]),
    'SMOTE': ImbPipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('smote', SMOTE(random_state=42)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42))
    ])
}

balancing_results = {}

for strategy_name, pipe in balancing_pipelines.items():
    print(f"\n  Testing: {strategy_name}...")
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='f1_weighted')
    balancing_results[strategy_name] = {
        'mean_f1': round(float(scores.mean()), 4),
        'std_f1':  round(float(scores.std()), 4)
    }
    print(f"  Mean F1: {scores.mean():.4f}  ±{scores.std():.4f}")

best_strategy = max(balancing_results, key=lambda s: balancing_results[s]['mean_f1'])
print(f"\n  Best balancing strategy: {best_strategy}")

# Save Phase 2 comparison CSV
df_balancing = pd.DataFrame([
    {
        'strategy': k,
        'mean_f1':  v['mean_f1'],
        'std_f1':   v['std_f1'],
        'is_best':  k == best_strategy
    }
    for k, v in balancing_results.items()
])
df_balancing.to_csv('ml_balancing_comparison.csv', index=False)
print("  Saved: ml_balancing_comparison.csv")

# PHASE 3 — RETRAIN ALL 3 MODELS WITH BEST STRATEGY
print(f"PHASE 3 — All Models with Best Strategy ({best_strategy})")

# class_weight='balanced' applies to LR and RF but not Naive Bayes
cw = 'balanced' if best_strategy == 'Class Weights' else None

def make_pipeline(clf, supports_class_weight=True):
    #Wraps classifier in the right pipeline based on the winning strategy.
    if best_strategy == 'SMOTE':
        return ImbPipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('smote', SMOTE(random_state=42)),
            ('clf',   clf)
        ])
    else:
        return Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf',   clf)
        ])

phase3_models = {
    'Logistic Regression': make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42, class_weight=cw),
        supports_class_weight=True
    ),
    'Naive Bayes': make_pipeline(
        MultinomialNB(alpha=0.1),
        supports_class_weight=False  # NB doesn't support class_weight
    ),
    'Random Forest': make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=cw),
        supports_class_weight=True
    )
}

phase3_results  = {}
phase3_predictions = {}

for model_name, pipeline in phase3_models.items():
    print(f"\n  Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    phase3_results[model_name]     = {'accuracy': acc, 'report': report, 'y_pred': y_pred}
    phase3_predictions[model_name] = y_pred

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

# Pick the final best model (by weighted F1 after balancing)
final_best_model_name = max(
    phase3_results,
    key=lambda m: phase3_results[m]['report']['weighted avg']['f1-score']
)
final_best_pipeline = phase3_models[final_best_model_name]

print(f"\n  Final best model: {final_best_model_name}")

# Save Phase 3 comparison CSV
df_phase3 = pd.DataFrame([
    {
        'model':              m,
        'accuracy':           round(r['accuracy'], 4),
        'weighted_f1':        round(r['report']['weighted avg']['f1-score'], 4),
        'weighted_precision': round(r['report']['weighted avg']['precision'], 4),
        'weighted_recall':    round(r['report']['weighted avg']['recall'], 4),
        'balancing_strategy': best_strategy,
        'is_best':            m == final_best_model_name
    }
    for m, r in phase3_results.items()
])
df_phase3.to_csv('ml_final_comparison.csv', index=False)
print("  Saved: ml_final_comparison.csv")

# PHASE 4 — same model, with and without balancing comparison
print(f"PHASE 4 — Before vs After Balancing ({final_best_model_name})")
before = phase1_results[final_best_model_name]
after  = phase3_results[final_best_model_name]

before_f1  = before['report']['weighted avg']['f1-score']
after_f1   = after['report']['weighted avg']['f1-score']
before_acc = before['accuracy']
after_acc  = after['accuracy']

print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Change':>10}")
print(f"  {'-'*50}")
print(f"  {'Accuracy':<20} {before_acc:>10.4f} {after_acc:>10.4f} {after_acc - before_acc:>+10.4f}")
print(f"  {'Weighted F1':<20} {before_f1:>10.4f} {after_f1:>10.4f} {after_f1 - before_f1:>+10.4f}")

# Per-class F1 before vs after for the final best model
before_per_class = {
    cls: round(before['report'][cls]['f1-score'], 3)
    for cls in le.classes_ if cls in before['report']
}
after_per_class = {
    cls: round(after['report'][cls]['f1-score'], 3)
    for cls in le.classes_ if cls in after['report']
}

before_after_rows = []
for cls in le.classes_:
    b = before_per_class.get(cls, 0)
    a = after_per_class.get(cls, 0)
    before_after_rows.append({
        'cuisine':    cls,
        'f1_before':  b,
        'f1_after':   a,
        'f1_change':  round(a - b, 3)
    })

df_before_after = pd.DataFrame(before_after_rows).sort_values('f1_change', ascending=False)
df_before_after.to_csv('ml_before_after.csv', index=False)
print("  Saved: ml_before_after.csv")

# PREDICT UNKNOWN CUISINES (using final best pipeline)
print("PREDICTING CUISINE FOR UNKNOWN REVIEWS")

X_unknown = unknown['review_text_cleaned'].values

predicted_indices    = final_best_pipeline.predict(X_unknown)
predicted_proba      = final_best_pipeline.predict_proba(X_unknown)
predicted_cuisines   = le.inverse_transform(predicted_indices)
predicted_confidences = predicted_proba.max(axis=1)

unknown = unknown.copy()
unknown['cuisine_predicted']    = predicted_cuisines
unknown['prediction_confidence'] = predicted_confidences.round(3)

LOW_CONFIDENCE_THRESHOLD = 0.30
low_conf_count = (unknown['prediction_confidence'] < LOW_CONFIDENCE_THRESHOLD).sum()

print(f"Predictions made: {len(unknown)}")
print(f"Low-confidence (< {LOW_CONFIDENCE_THRESHOLD}): {low_conf_count} ({low_conf_count/len(unknown)*100:.1f}%)")
print(f"Avg confidence: {predicted_confidences.mean():.3f}")
print("\nPredicted cuisine distribution:")
print(unknown['cuisine_predicted'].value_counts().to_string())

# SAVE PREDICTED DISTRIBUTION
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
print("\nSaved: ml_predicted_distribution.csv")

# BUILD ENRICHED MASTER REVIEWS FILE
known_out = known.copy()
known_out['cuisine_source']        = 'original'
known_out['prediction_confidence'] = np.nan

unknown_out = unknown.copy()
unknown_out['cuisine_primary']  = unknown_out['cuisine_predicted']
unknown_out['cuisine_source']   = 'predicted'
unknown_out = unknown_out.drop(columns=['cuisine_predicted'])

master_enriched = pd.concat([known_out, unknown_out], ignore_index=True)
master_enriched.to_csv('master_reviews_enriched.csv', index=False)
print("Saved: master_reviews_enriched.csv")

# SAVE SUMMARY JSON
best_report = phase3_results[final_best_model_name]['report']

per_class_f1 = {
    cls: round(best_report[cls]['f1-score'], 3)
    for cls in le.classes_ if cls in best_report
}

cm_full      = confusion_matrix(y_test, phase3_predictions[final_best_model_name])
cm_full_norm = cm_full.astype(float) / cm_full.sum(axis=1, keepdims=True)

confusion_pairs = []
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm_full_norm[i, j] > 0.05:
            confusion_pairs.append({
                'actual':    le.classes_[i],
                'predicted': le.classes_[j],
                'rate':      round(float(cm_full_norm[i, j]), 3)
            })
confusion_pairs = sorted(confusion_pairs, key=lambda x: x['rate'], reverse=True)[:5]

summary = {
    # Data stats
    'total_reviews_loaded':      int(len(reviews)),
    'known_cuisine_reviews':     int(len(known)),
    'unknown_cuisine_reviews':   int(len(unknown)),

    # Phase 1 — model comparison (no balancing)
    'phase1_models': {
        m: {
            'accuracy':    round(r['accuracy'], 4),
            'weighted_f1': round(r['report']['weighted avg']['f1-score'], 4)
        }
        for m, r in phase1_results.items()
    },
    'phase1_best_model': best_model_name,

    # Phase 2 — balancing strategy comparison
    'phase2_balancing_strategies': balancing_results,
    'phase2_best_strategy': best_strategy,

    # Phase 3 — final model results with balancing
    'phase3_models': {
        m: {
            'accuracy':    round(r['accuracy'], 4),
            'weighted_f1': round(r['report']['weighted avg']['f1-score'], 4)
        }
        for m, r in phase3_results.items()
    },
    'phase3_best_model': final_best_model_name,

    # Phase 4 — before vs after
    'before_after': {
        'model':            final_best_model_name,
        'accuracy_before':  round(before_acc, 4),
        'accuracy_after':   round(after_acc, 4),
        'f1_before':        round(before_f1, 4),
        'f1_after':         round(after_f1, 4),
    },

    # Confusion matrix
    'confusion_matrix_note': f'Top 10 cuisines only (out of {len(le.classes_)} total classes)',
    'top_confusion_pairs': confusion_pairs,

    # Prediction outcomes
    'predictions_made':           int(len(unknown)),
    'low_confidence_count':       int(low_conf_count),
    'low_confidence_pct':         round(low_conf_count / len(unknown) * 100, 1),
    'low_confidence_threshold':   LOW_CONFIDENCE_THRESHOLD,
    'avg_prediction_confidence':  round(float(predicted_confidences.mean()), 3),

    # Per-class F1 (final model)
    'per_class_f1': per_class_f1,

    # Enriched file stats
    'enriched_reviews_total':      int(len(master_enriched)),
    'enriched_original_labels':    int((master_enriched['cuisine_source'] == 'original').sum()),
    'enriched_predicted_labels':   int((master_enriched['cuisine_source'] == 'predicted').sum()),
}

with open('ml_cuisine_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: ml_cuisine_summary.json")

# FINAL SUMMARY PRINT
print("DONE")

print(f"Phase 1 best model:       {best_model_name}")
print(f"Phase 2 best strategy:    {best_strategy}")
print(f"Phase 3 final model:      {final_best_model_name}")
print(f"  Accuracy before/after:  {before_acc:.4f} → {after_acc:.4f}  ({after_acc - before_acc:+.4f})")
print(f"  F1 before/after:        {before_f1:.4f} → {after_f1:.4f}  ({after_f1 - before_f1:+.4f})")
print(f"Reviews predicted:        {len(unknown)}")
print(f"Avg confidence:           {predicted_confidences.mean():.3f}")
print(f"Enriched dataset size:    {len(master_enriched)}")

print("Output files:")
print("  ml_model_comparison.csv      Phase 1: 3 models, no balancing")
print("  ml_balancing_comparison.csv  Phase 2: 3 balancing strategies")
print("  ml_final_comparison.csv      Phase 3: 3 models with best strategy")
print("  ml_before_after.csv          Phase 4: per-class F1 before vs after")
print("  ml_predicted_distribution.csv")
print("  ml_cuisine_summary.json")
print("  master_reviews_enriched.csv")