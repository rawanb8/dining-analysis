import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer
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

# NOTES:
# COLLAPSE SPARSE AND OVERLAPPING CUISINES
# The grouped split requires enough *restaurants* per class, not just reviews.
# A class with 200 reviews from only 2 restaurants is untrainable under grouped CV
# because one of those restaurants may end up entirely in train or entirely in test.
MIN_RESTAURANTS_PER_CLASS =  15
MIN_REVIEWS_PER_CLASS = 400 

# Merge cuisines that genuinely overlap in Beirut's food scene.
CUISINE_MERGES = {
    'Lebanese': 'Levantine',
    'Middle Eastern': 'Levantine',
    'Turkish': 'Levantine',
    'Mediterranean': 'Levantine',
    'Armenian': 'Levantine',
}

#labels that aren't really cuisines drop to 'Other'.
VENUE_TYPES = {'Wine bars', 'Cocktail bars', 'Delis'}

known['cuisine_label'] = known['cuisine_primary'].replace(CUISINE_MERGES)
known['cuisine_label'] = known['cuisine_label'].apply(
    lambda c: 'Other' if c in VENUE_TYPES else c
)

#count BOTH reviews AND distinct restaurants per class
class_stats = known.groupby('cuisine_label').agg(
    n_reviews=('review_id', 'count'),
    n_restaurants=('restaurant_id', 'nunique')
).sort_values('n_reviews', ascending=False)

print("\nPre-collapse class stats:")
print(class_stats.to_string())

#keep only classes that meet BOTH thresholds
keep_mask = (
    (class_stats['n_reviews'] >= MIN_REVIEWS_PER_CLASS) &
    (class_stats['n_restaurants'] >= MIN_RESTAURANTS_PER_CLASS)
)
classes_to_keep = set(class_stats[keep_mask].index)
classes_to_drop = set(class_stats[~keep_mask].index)

known['cuisine_label'] = known['cuisine_label'].apply(
    lambda c: c if c in classes_to_keep else 'Other'
)

final_stats = known.groupby('cuisine_label').agg(
    n_reviews=('review_id', 'count'),
    n_restaurants=('restaurant_id', 'nunique')
).sort_values('n_reviews', ascending=False)

print(f"\nMerged cuisines: {CUISINE_MERGES}")
print(f"Venue types dropped to 'Other': {VENUE_TYPES}")
print(f"Collapsed into 'Other' (below threshold): {classes_to_drop}")
print(f"\nFinal class count: {known['cuisine_label'].nunique()}")
print(f"\nFinal class stats:")
print(final_stats.to_string())

assert final_stats['n_restaurants'].min() >= MIN_RESTAURANTS_PER_CLASS, \
    "Some class still has too few restaurants — tighten merges or raise threshold"

# ENCODE LABELS AND SPLIT (grouped by restaurant to prevent data leakage)
le = LabelEncoder()
y  = le.fit_transform(known['cuisine_label'])
X  = known['review_text_cleaned'].values
groups = known['restaurant_id'].values  

# GroupShuffleSplit: all reviews from one restaurant go entirely
# to train OR test, never both. Prevents the model from memorizing
# restaurant-specific vocab instead of learning cuisine vocab.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]  

# Sanity check: no restaurant in both splits
overlap = set(groups[train_idx]) & set(groups[test_idx])
assert len(overlap) == 0, f"Leakage! {len(overlap)} restaurants in both splits"

#verify every class is present in both splits
train_classes = set(y_train)
test_classes = set(y_test)
missing_in_test  = train_classes - test_classes
missing_in_train = test_classes - train_classes
if missing_in_test or missing_in_train:
    print(f"WARNING: class imbalance after grouped split")
    print(f"Classes missing in test:  {[le.classes_[c] for c in missing_in_test]}")
    print(f"Classes missing in train: {[le.classes_[c] for c in missing_in_train]}")

print(f"\nTraining: {len(X_train)} reviews from {len(set(groups[train_idx]))} restaurants")
print(f"Test:{len(X_test)} reviews from {len(set(groups[test_idx]))} restaurants")
print(f"Overlap:{len(overlap)} (must be 0)")

#NOTES: restaurant-level weighted F1
#phase 2 (balancing comparison) uses f1 per review but we need per rest. f1
_GROUPS_ARRAY = groups  

def restaurant_level_f1_scorer(estimator, X_fold, y_fold):
    try:
        proba = estimator.predict_proba(X_fold)
    except AttributeError:
        preds = estimator.predict(X_fold)
        n_classes = len(np.unique(y))
        proba = np.zeros((len(preds), n_classes))
        proba[np.arange(len(preds)), preds] = 1.0

    # group reviews by restaurant
    text_to_group = dict(zip(X, _GROUPS_ARRAY))
    fold_groups = np.array([text_to_group[x] for x in X_fold])

    # Aggregate probabilities by restaurant
    fold_df = pd.DataFrame(proba)
    fold_df['group'] = fold_groups
    fold_df['y_true'] = y_fold
    agg_proba = fold_df.groupby('group').mean()
    restaurant_y_true = agg_proba['y_true'].astype(int).values
    restaurant_y_pred = agg_proba.drop(columns=['y_true']).values.argmax(axis=1)

    return f1_score(restaurant_y_true, restaurant_y_pred, average='weighted', zero_division=0)
    
# SHARED TFIDF PARAMS (used in all phases)
tfidf_params = dict(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    stop_words='english'
)

# PHASE 1 comapring ml models without balancing
print("PHASE 1 — Model Comparison (No Balancing)")

phase1_models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

phase1_results = {}

for model_name, pipeline in phase1_models.items():
    print(f"\n  Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test,  y_pred)
    overfit_gap = train_acc - test_acc

    report = classification_report(
        y_test, y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    phase1_results[model_name] = {
        'accuracy': test_acc,
        'train_acc': train_acc,
        'overfit_gap': overfit_gap,
        'report': report,
        'y_pred': y_pred
    }

    print(f"  Train acc:   {train_acc:.4f}")
    print(f"  Test acc:    {test_acc:.4f}")
    print(f"  Overfit gap: {overfit_gap:+.4f}  (large positive = overfitting)")
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
        'model': m,
        'train_accuracy': round(r['train_acc'], 4),
        'test_accuracy': round(r['accuracy'], 4),
        'overfit_gap': round(r['overfit_gap'], 4),
        'weighted_f1': round(r['report']['weighted avg']['f1-score'], 4),
        'weighted_precision': round(r['report']['weighted avg']['precision'], 4),
        'weighted_recall': round(r['report']['weighted avg']['recall'], 4),
        'is_best': m == best_model_name
    }
    for m, r in phase1_results.items()
])
df_phase1.to_csv('ml_model_comparison.csv', index=False)
print("  Saved: ml_model_comparison.csv")

# PHASE 2 — comapring the balancing strategies using the best model from phase 1
# Uses the best model from Phase 1 with 5-Fold Stratified CV
print(f"PHASE 2 Balancing Strategy Comparison (using {best_model_name})")

# StratifiedGroupKFold: keeps class balance across folds 
# ensures no restaurant appears in more than one fold 
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression is used here because it supports all 3 strategies cleanly.
# Naive Bayes doesn't support class_weight
balancing_pipelines = {
    'Baseline (No Balancing)': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Class Weights': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]),
    'SMOTE': ImbPipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
}

balancing_results = {}
balancing_fold_scores = {} 

for strategy_name, pipe in balancing_pipelines.items():
    print(f"\n  Testing: {strategy_name}...")
    scores = cross_val_score(pipe, X, y, cv=sgkf, groups=groups, scoring=restaurant_level_f1_scorer)
    balancing_results[strategy_name] = {
        'mean_f1': round(float(scores.mean()), 4),
        'std_f1':  round(float(scores.std()), 4)
    }
    balancing_fold_scores[strategy_name] = [round(float(s), 4) for s in scores]
    print(f"  Fold scores (restaurant-level F1): {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean restaurant-level F1: {scores.mean():.4f}  ±{scores.std():.4f}")

# Save per-fold CV scores as a long-format CSV for the dashboard
cv_rows = []
for strategy_name, fold_scores in balancing_fold_scores.items():
    for fold_idx, score in enumerate(fold_scores, start=1):
        cv_rows.append({
            'strategy': strategy_name,
            'fold': fold_idx,
            'f1_score': score
        })
pd.DataFrame(cv_rows).to_csv('ml_cv_fold_scores.csv', index=False)
print("\n  Saved: ml_cv_fold_scores.csv")

best_strategy = max(balancing_results, key=lambda s: balancing_results[s]['mean_f1'])
print(f"\n  Best balancing strategy: {best_strategy}")

# Save Phase 2 comparison CSV
df_balancing = pd.DataFrame([
    {
        'strategy': k,
        'mean_f1': v['mean_f1'],
        'std_f1': v['std_f1'],
        'is_best': k == best_strategy
    }
    for k, v in balancing_results.items()
])
df_balancing.to_csv('ml_balancing_comparison.csv', index=False)
print("  Saved: ml_balancing_comparison.csv")

# PHASE 3 — RETRAIN ALL 3 MODELS WITH BEST STRATEGY
print(f"PHASE 3 All Models with Best Strategy ({best_strategy})")

cw = 'balanced' if best_strategy == 'Class Weights' else None

def build_phase3_pipeline(clf):
    if best_strategy == 'SMOTE':
        return ImbPipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('smote', SMOTE(random_state=42)),
            ('clf', clf)
        ])
    else:
        return Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', clf)
        ])

phase3_models = {
    'Logistic Regression': build_phase3_pipeline(
        LogisticRegression(max_iter=1000, random_state=42, class_weight=cw)
    ),
    'Naive Bayes': build_phase3_pipeline(
        MultinomialNB(alpha=0.1)  
    ),
    'Random Forest': build_phase3_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=cw)
    )
}

phase3_results  = {}
phase3_predictions = {}

for model_name, pipeline in phase3_models.items():
    print(f"\n  Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test,  y_pred)
    overfit_gap = train_acc - test_acc

    report = classification_report(
        y_test, y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    phase3_results[model_name] = {
        'accuracy': test_acc,
        'train_acc': train_acc,
        'overfit_gap': overfit_gap,
        'report': report,
        'y_pred': y_pred
    }
    phase3_predictions[model_name] = y_pred

    print(f" Train acc:   {train_acc:.4f}")
    print(f" Test acc:    {test_acc:.4f}")
    print(f" Overfit gap: {overfit_gap:+.4f}")
    print(f" Weighted F1: {report['weighted avg']['f1-score']:.4f}")

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
        'model': m,
        'train_accuracy': round(r['train_acc'], 4),
        'test_accuracy': round(r['accuracy'], 4),
        'overfit_gap': round(r['overfit_gap'], 4),
        'weighted_f1': round(r['report']['weighted avg']['f1-score'], 4),
        'weighted_precision': round(r['report']['weighted avg']['precision'], 4),
        'weighted_recall': round(r['report']['weighted avg']['recall'], 4),
        'balancing_strategy': best_strategy,
        'is_best': m == final_best_model_name
    }
    for m, r in phase3_results.items()
])
df_phase3.to_csv('ml_final_comparison.csv', index=False)
print("Saved: ml_final_comparison.csv")

# PHASE 3.5: HYPERPARAMETER TUNING (on final best model only)
print(f"\nPHASE 3.5 Hyperparameter Tuning ({final_best_model_name})")

# Use 3-fold grouped CV for tuning (do we increase to 5?)
tuning_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

# Tailor the grid to whichever model won
if final_best_model_name == 'Logistic Regression':
    param_grid = {
        'tfidf__max_features': [10000, 20000, 30000],   
        'tfidf__min_df': [3, 5, 10],
        'clf__C': [0.1, 0.3, 1.0, 3.0],
    }

elif final_best_model_name == 'Naive Bayes':
    param_grid = {
        'tfidf__max_features': [10000, 20000, 50000],
        'tfidf__min_df': [2, 3, 5],
        'clf__alpha': [0.01, 0.1, 0.5, 1.0],
    }
else:  # Random Forest
    param_grid = {
        'tfidf__max_features': [10000, 20000],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 30],
    }

grid_size = int(np.prod([len(v) for v in param_grid.values()]))
print(f"  Grid size: {grid_size} combinations × 3 folds = {grid_size * 3} fits")

grid_search = GridSearchCV(
    final_best_pipeline,
    param_grid,
    cv=tuning_cv,
    scoring=restaurant_level_f1_scorer,
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_train, y_train, groups=groups_train)

print(f"\n Best params: {grid_search.best_params_}")
print(f"  Best CV F1:  {grid_search.best_score_:.4f}")

# Save the full GridSearchCV results 
cv_results = pd.DataFrame(grid_search.cv_results_)
fold_cols = [c for c in cv_results.columns if c.startswith('split') and c.endswith('_test_score')]
keep_cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score'] + fold_cols
cv_results_slim = cv_results[keep_cols].copy()
cv_results_slim['params'] = cv_results_slim['params'].astype(str)
cv_results_slim = cv_results_slim.sort_values('rank_test_score')
cv_results_slim.to_csv('ml_grid_search_results.csv', index=False)
print(f"  Saved: ml_grid_search_results.csv ({len(cv_results_slim)} combinations × {len(fold_cols)} folds)")

# Evaluate the tuned model on the held-out test set
tuned_pipeline = grid_search.best_estimator_
y_train_pred_tuned = tuned_pipeline.predict(X_train)
y_pred_tuned = tuned_pipeline.predict(X_test)

tuned_train_acc = accuracy_score(y_train, y_train_pred_tuned)
tuned_test_acc  = accuracy_score(y_test,  y_pred_tuned)
tuned_report    = classification_report(
    y_test, y_pred_tuned,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)
tuned_f1   = tuned_report['weighted avg']['f1-score']
untuned_f1 = phase3_results[final_best_model_name]['report']['weighted avg']['f1-score']

print(f"\n  Tuned train acc:  {tuned_train_acc:.4f}")
print(f"  Tuned test acc:   {tuned_test_acc:.4f}")
print(f"  Tuned F1:         {tuned_f1:.4f}")
print(f"  Improvement over untuned: {tuned_f1 - untuned_f1:+.4f}")

# save the before for comparison
untuned_phase3_result = dict(phase3_results[final_best_model_name]) 

# Build a tuned result object regardless of which one wins — we want both for the dashboard
tuned_phase3_result = {
    'accuracy': tuned_test_acc,
    'train_acc': tuned_train_acc,
    'overfit_gap': tuned_train_acc - tuned_test_acc,
    'report': tuned_report,
    'y_pred':  y_pred_tuned,
}

# Only use tuned model if it actually improved 
# finds params that win on CV but lose on held-out test
if tuned_f1 > untuned_f1:
    print(f"  -> Using TUNED model for final predictions")
    final_best_pipeline = tuned_pipeline
    phase3_results[final_best_model_name] = tuned_phase3_result
    phase3_predictions[final_best_model_name] = y_pred_tuned
    tuning_improved = True
else:
    print(f"  -> Keeping UNTUNED model (tuning did not improve held-out F1)")
    tuning_improved = False


pd.DataFrame({
    'param': list(grid_search.best_params_.keys()),
    'best_value':list(grid_search.best_params_.values())
}).to_csv('ml_tuning_best_params.csv', index=False)
print("  Saved: ml_tuning_best_params.csv")

# PHASE 4 same model, with and without balancing comparison
print(f"PHASE 4 — Before vs After Balancing ({final_best_model_name})")
before = phase1_results[final_best_model_name]
after  = phase3_results[final_best_model_name]

before_f1 = before['report']['weighted avg']['f1-score']
after_f1  = after['report']['weighted avg']['f1-score']
before_acc = before['accuracy']
after_acc = after['accuracy']

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
    # Mark classes that had zero test examples — their F1 is meaningless
    test_support = (y_test == np.where(le.classes_ == cls)[0][0]).sum() if cls in le.classes_ else 0
    before_after_rows.append({
        'cuisine': cls,
        'f1_before': b,
        'f1_after': a,
        'f1_change': round(a - b, 3),
        'test_support': int(test_support),
        'note': 'no test examples' if test_support == 0 else ''
    })

df_before_after = pd.DataFrame(before_after_rows).sort_values('f1_change', ascending=False)
df_before_after.to_csv('ml_before_after.csv', index=False)
print("  Saved: ml_before_after.csv")

#BEFORE vs AFTER HYPERPARAMETER TUNING
print(f"\nBefore vs After Hyperparameter Tuning ({final_best_model_name})")

untuned_f1_= untuned_phase3_result['report']['weighted avg']['f1-score']
tuned_f1_ = tuned_phase3_result['report']['weighted avg']['f1-score']
untuned_acc = untuned_phase3_result['accuracy']
tuned_acc = tuned_phase3_result['accuracy']
untuned_gap  = untuned_phase3_result['overfit_gap']
tuned_gap  = tuned_phase3_result['overfit_gap']

print(f"  {'Metric':<20} {'Untuned':>10} {'Tuned':>10} {'Change':>10}")
print(f"  {'-'*50}")
print(f"  {'Accuracy':<20} {untuned_acc:>10.4f} {tuned_acc:>10.4f} {tuned_acc - untuned_acc:>+10.4f}")
print(f"  {'Weighted F1':<20} {untuned_f1_:>10.4f} {tuned_f1_:>10.4f} {tuned_f1_ - untuned_f1_:>+10.4f}")
print(f"  {'Overfit Gap':<20} {untuned_gap:>10.4f} {tuned_gap:>10.4f} {tuned_gap - untuned_gap:>+10.4f}")

# Per-class F1 comparison
untuned_per_class = {
    cls: round(untuned_phase3_result['report'][cls]['f1-score'], 3)
    for cls in le.classes_ if cls in untuned_phase3_result['report']
}
tuned_per_class = {
    cls: round(tuned_phase3_result['report'][cls]['f1-score'], 3)
    for cls in le.classes_ if cls in tuned_phase3_result['report']
}

tuning_rows = []
for cls in le.classes_:
    u = untuned_per_class.get(cls, 0)
    t = tuned_per_class.get(cls, 0)
    tuning_rows.append({
        'cuisine':     cls,
        'f1_untuned':  u,
        'f1_tuned':    t,
        'f1_change':   round(t - u, 3)
    })

df_tuning_compare = pd.DataFrame(tuning_rows).sort_values('f1_change', ascending=False)
df_tuning_compare.to_csv('ml_tuning_before_after.csv', index=False)
print("  Saved: ml_tuning_before_after.csv")

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

# AGGREGATE TO RESTAURANT-LEVEL PREDICTIONS
# A cuisine is a property of a restaurant, not a review. Individual reviews
# are noisy. Averaging probability vectors across all of a restaurant's reviews
# gives a much more stable prediction than voting one review at a time.
print("\n" + "=" * 60)
print("AGGREGATING TO RESTAURANT-LEVEL PREDICTIONS")

# Attach per-review probability vectors back to the unknown dataframe
unknown_with_probs = unknown.copy()
for i, class_name in enumerate(le.classes_):
    unknown_with_probs[f'prob_{class_name}'] = predicted_proba[:, i]

# Group by restaurant,average the probability for each cuisine across its reviews
prob_cols = [f'prob_{c}' for c in le.classes_]
restaurant_probs = unknown_with_probs.groupby('restaurant_id')[prob_cols].mean()

# Argmax over the averaged probabilities → one cuisine per restaurant
restaurant_probs['cuisine_restaurant_level']  = restaurant_probs[prob_cols].idxmax(axis=1).str.replace('prob_', '', regex=False)
restaurant_probs['restaurant_confidence']     = restaurant_probs[prob_cols].max(axis=1).round(3)
restaurant_probs['n_reviews_for_prediction']  = unknown_with_probs.groupby('restaurant_id').size()

# How often does per-review voting disagree with probability-averaging?
# Per-review majority vote, for comparison
per_review_majority = (
    unknown_with_probs.groupby('restaurant_id')['cuisine_predicted']
    .agg(lambda s: s.mode().iloc[0])
)
restaurant_probs['cuisine_majority_vote'] = per_review_majority

disagreements = (
    restaurant_probs['cuisine_restaurant_level'] != restaurant_probs['cuisine_majority_vote']
).sum()
print(f"Restaurants classified: {len(restaurant_probs)}")
print(f"Avg reviews per restaurant: {restaurant_probs['n_reviews_for_prediction'].mean():.1f}")
print(f"Avg restaurant-level confidence: {restaurant_probs['restaurant_confidence'].mean():.3f}")
print(f"Probability-averaging disagrees with majority vote on: {disagreements} restaurants "
      f"({disagreements/len(restaurant_probs)*100:.1f}%)")

# Confidence distribution at restaurant level vs review level
high_conf_restaurants = (restaurant_probs['restaurant_confidence'] >= 0.5).sum()
print(f"Restaurants with high-confidence (≥0.5) prediction: "
      f"{high_conf_restaurants} ({high_conf_restaurants/len(restaurant_probs)*100:.1f}%)")

# Save the restaurant-level output
restaurant_output = restaurant_probs[[
    'cuisine_restaurant_level', 'restaurant_confidence',
    'n_reviews_for_prediction', 'cuisine_majority_vote'
]].reset_index()
restaurant_output.to_csv('ml_restaurant_level_predictions.csv', index=False)
print("Saved: ml_restaurant_level_predictions.csv")

# EVALUATE RESTAURANT-LEVEL F1 ON THE TEST SET
print("RESTAURANT-LEVEL EVALUATION ON TEST SET")

# Predict probabilities for the test set, aggregate the same way, evaluate
test_proba = final_best_pipeline.predict_proba(X_test)
test_df = pd.DataFrame({'restaurant_id': groups[test_idx], 'y_true': y_test})
for i, class_name in enumerate(le.classes_):
    test_df[f'prob_{class_name}'] = test_proba[:, i]

# Each test restaurant has one true cuisine (all its reviews share the same label)
test_restaurant = test_df.groupby('restaurant_id').agg(
    **{col: (col, 'mean') for col in prob_cols},
    y_true=('y_true', 'first')
)
test_restaurant['y_pred'] = test_restaurant[prob_cols].values.argmax(axis=1)

restaurant_acc = accuracy_score(test_restaurant['y_true'], test_restaurant['y_pred'])
restaurant_report = classification_report(
    test_restaurant['y_true'], test_restaurant['y_pred'],
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)
restaurant_f1 = restaurant_report['weighted avg']['f1-score']

# Compare to review-level metrics
review_acc = phase3_results[final_best_model_name]['accuracy']
review_f1  = phase3_results[final_best_model_name]['report']['weighted avg']['f1-score']

print(f"  {'Level':<20} {'Accuracy':>10} {'F1':>10}")
print(f"  {'-'*42}")
print(f"  {'Per-review':<20} {review_acc:>10.4f} {review_f1:>10.4f}")
print(f"  {'Per-restaurant':<20} {restaurant_acc:>10.4f} {restaurant_f1:>10.4f}")
print(f"  {'Improvement':<20} {restaurant_acc - review_acc:>+10.4f} {restaurant_f1 - review_f1:>+10.4f}")

# Save restaurant-level per-class F1
restaurant_per_class_f1 = {
    cls: round(restaurant_report[cls]['f1-score'], 3)
    for cls in le.classes_ if cls in restaurant_report
}
pd.DataFrame([
    {
        'cuisine':cls,
        'f1_review_level': round(phase3_results[final_best_model_name]['report'].get(cls, {}).get('f1-score', 0), 3),
        'f1_restaurant_level': restaurant_per_class_f1.get(cls, 0),
    }
    for cls in le.classes_
]).to_csv('ml_restaurant_vs_review_f1.csv', index=False)
print("Saved: ml_restaurant_vs_review_f1.csv")

# BUILD ENRICHED MASTER REVIEWS FILE
# Use RESTAURANT-LEVEL predictions — all reviews of the same restaurant
# get the same cuisine label, which is how cuisine actually works.
known_out = known.copy()
known_out['cuisine_source'] = 'original'
known_out['prediction_confidence'] = np.nan

# Map each unknown review to its restaurant's aggregated cuisine and confidence
restaurant_lookup = restaurant_probs[['cuisine_restaurant_level', 'restaurant_confidence']].to_dict('index')

unknown_out = unknown.copy()
unknown_out['cuisine_primary'] = unknown_out['restaurant_id'].map(
    lambda rid: restaurant_lookup.get(rid, {}).get('cuisine_restaurant_level', 'Unknown')
)
unknown_out['prediction_confidence'] = unknown_out['restaurant_id'].map(
    lambda rid: restaurant_lookup.get(rid, {}).get('restaurant_confidence', np.nan)
)
unknown_out['cuisine_source'] = 'predicted'
# Keep the per-review prediction too, for transparency / debugging — different column
unknown_out = unknown_out.rename(columns={'cuisine_predicted': 'cuisine_predicted_per_review'})

master_enriched = pd.concat([known_out, unknown_out], ignore_index=True)
master_enriched.to_csv('master_reviews_enriched.csv', index=False)
print(f"Saved: master_reviews_enriched.csv")
print(f"  ({len(known_out)} original + {len(unknown_out)} predicted = {len(master_enriched)} total)")

# Sanity check: every restaurant in the enriched file should have exactly ONE cuisine
cuisines_per_restaurant = master_enriched.groupby('restaurant_id')['cuisine_primary'].nunique()
multi_cuisine_restaurants = (cuisines_per_restaurant > 1).sum()
print(f"Restaurants with multiple cuisine labels: {multi_cuisine_restaurants} (should be 0)")

# SAVE SUMMARY JSON
best_report = phase3_results[final_best_model_name]['report']

per_class_f1 = {
    cls: round(best_report[cls]['f1-score'], 3)
    for cls in le.classes_ if cls in best_report
}

cm_full = confusion_matrix(y_test, phase3_predictions[final_best_model_name])
cm_full_norm = cm_full.astype(float) / cm_full.sum(axis=1, keepdims=True)

# Save confusion matrix as CSV for dashboard rendering
cm_df = pd.DataFrame(
    cm_full_norm,
    index=le.classes_,
    columns=le.classes_
).round(3)
cm_df.to_csv('ml_confusion_matrix.csv')
print("  Saved: ml_confusion_matrix.csv")

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

   # Phase 4 — before vs after balancing
    'before_after': {
        'model': final_best_model_name,
        'accuracy_before': round(before_acc, 4),
        'accuracy_after':  round(after_acc, 4),
        'f1_before': round(before_f1, 4),
        'f1_after': round(after_f1, 4),
    },

    #before vs after hyperparameter tuning
    'tuning_before_after': {
        'model': final_best_model_name,
        'accuracy_untuned': round(untuned_acc, 4),
        'accuracy_tuned': round(tuned_acc, 4),
        'f1_untuned': round(untuned_f1_, 4),
        'f1_tuned': round(tuned_f1_, 4),
        'overfit_gap_untuned': round(untuned_gap, 4),
        'overfit_gap_tuned': round(tuned_gap, 4),
        'tuning_improved': bool(tuning_improved),
        'best_params':   {k: (float(v) if isinstance(v, (int, float)) else str(v))
                              for k, v in grid_search.best_params_.items()},
    },
    
    # Restaurant-level evaluation — the metric that actually matters
    'restaurant_level': {
        'n_restaurants_in_test': int(len(test_restaurant)),
        'n_restaurants_predicted':int(len(restaurant_probs)),
        'accuracy_review_level':round(review_acc, 4),
        'accuracy_restaurant_level':round(restaurant_acc, 4),
        'f1_review_level': round(review_f1, 4),
        'f1_restaurant_level':round(restaurant_f1, 4),
        'improvement_f1': round(restaurant_f1 - review_f1, 4),
        'avg_confidence':  round(float(restaurant_probs['restaurant_confidence'].mean()), 3),
        'high_confidence_pct': round(high_conf_restaurants/len(restaurant_probs)*100, 1),
        'per_class_f1':  restaurant_per_class_f1,
    },

    # Confusion matrix
    'confusion_matrix_note': f'Top 10 cuisines only (out of {len(le.classes_)} total classes)',
    'top_confusion_pairs': confusion_pairs,

    # Prediction outcomes
    'predictions_made': int(len(unknown)),
    'low_confidence_count': int(low_conf_count),
    'low_confidence_pct': round(low_conf_count / len(unknown) * 100, 1),
    'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
    'avg_prediction_confidence': round(float(predicted_confidences.mean()), 3),

    # Per-class F1 (final model)
    'per_class_f1': per_class_f1,

    # Enriched file stats
    'enriched_reviews_total': int(len(master_enriched)),
    'enriched_original_labels': int((master_enriched['cuisine_source'] == 'original').sum()),
    'enriched_predicted_labels': int((master_enriched['cuisine_source'] == 'predicted').sum()),
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