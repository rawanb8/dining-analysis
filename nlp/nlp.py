import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# SHARED HELPERS
custom_stopwords = list(ENGLISH_STOP_WORDS) + [
    'food', 'service', 'atmosphere', 'place', 'meal',
    'type', 'meal type', 'food service', 'service atmosphere',
    'dine', 'restaurant', 'good', 'great', 'nice', 'order',
    'ordered', 'time', 'went', 'came', 'got', 'like', 'just'
]

def run_nlp(reviews_path, label):
    """
    Runs the full NLP on a reviews CSV.
    Returns a dict with all results so we could use them in dashbaord
    label: 'ORIGINAL' or 'ENRICHED'
    """

    print(f"\n{'=' * 60}")
    print(f"NLP PIPELINE — {label}")
    print(f"{'=' * 60}")

    print("Loading reviews...")
    reviews = pd.read_csv(reviews_path)
    reviews = reviews[reviews['review_text_cleaned'].notna()]
    reviews = reviews[reviews['review_text_cleaned'].str.strip() != '']

    reviews_known = reviews[
        (reviews['area'] != 'Unknown') &
        (reviews['cuisine_primary'] != 'Unknown') &
        (reviews['price_category'] != 'Unknown')
    ].copy()

    reviews_for_tfidf = reviews[
        reviews['area'] != 'Unknown'
    ].copy()

    print(f"Total reviews loaded: {len(reviews)}")
    print(f"Reviews with full metadata: {len(reviews_known)}")
    print()

    # ── SENTIMENT BY AREA ──
    print(f"--------- SENTIMENT BY AREA ({label}) ---------")
    sentiment_by_area = (
        reviews_known.groupby('area')
        .agg(
            avg_sentiment=('sentiment_score', 'mean'),
            review_count=('sentiment_score', 'count'),
            pct_positive=('sentiment_category', lambda x: (x == 'Positive').mean() * 100),
            pct_negative=('sentiment_category', lambda x: (x == 'Negative').mean() * 100),
        )
        .round(3)
        .sort_values('avg_sentiment', ascending=False)
    )
    sentiment_by_area = sentiment_by_area[sentiment_by_area['review_count'] >= 10]
    print(sentiment_by_area.to_string())
    print()

    # ── SENTIMENT BY CUISINE ──
    print(f"------SENTIMENT BY CUISINE ({label})------")
    sentiment_by_cuisine = (
        reviews_known.groupby('cuisine_primary')
        .agg(
            avg_sentiment=('sentiment_score', 'mean'),
            review_count=('sentiment_score', 'count'),
            pct_positive=('sentiment_category', lambda x: (x == 'Positive').mean() * 100),
            pct_negative=('sentiment_category', lambda x: (x == 'Negative').mean() * 100),
        )
        .round(3)
        .sort_values('avg_sentiment', ascending=False)
    )
    sentiment_by_cuisine = sentiment_by_cuisine[sentiment_by_cuisine['review_count'] >= 10]
    print(sentiment_by_cuisine.to_string())
    print()

    # ── SENTIMENT BY PRICE ──
    print(f"--------- SENTIMENT BY PRICE TIER ({label}) ---------")
    sentiment_by_price = (
        reviews_known.groupby('price_category')
        .agg(
            avg_sentiment=('sentiment_score', 'mean'),
            review_count=('sentiment_score', 'count'),
            pct_positive=('sentiment_category', lambda x: (x == 'Positive').mean() * 100),
            pct_negative=('sentiment_category', lambda x: (x == 'Negative').mean() * 100),
        )
        .round(3)
        .sort_values('avg_sentiment', ascending=False)
    )
    print(sentiment_by_price.to_string())
    print()

    # ── TF-IDF KEYWORDS PER AREA ──
    print(f"------TF-IDF KEYWORDS PER AREA ({label})------")
    vectorizer = TfidfVectorizer(
        stop_words=custom_stopwords,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    area_texts = reviews_for_tfidf.groupby('area')['review_text_cleaned'].apply(
        lambda texts: ' '.join(texts.tolist())
    )
    tfidf_matrix = vectorizer.fit_transform(area_texts)
    feature_names = vectorizer.get_feature_names_out()

    top_keywords_per_area = {}
    for i, area in enumerate(area_texts.index):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[-10:][::-1]
        top_keywords_per_area[area] = [feature_names[j] for j in top_indices]

    for area, keywords in top_keywords_per_area.items():
        print(f"  {area:25s} → {', '.join(keywords)}")
    print()

    # ── TF-IDF KEYWORDS PER CUISINE ──
    print(f"------TF-IDF KEYWORDS PER CUISINE ({label})------")
    cuisine_texts = reviews_for_tfidf.groupby('cuisine_primary')['review_text_cleaned'].apply(
        lambda texts: ' '.join(texts.tolist())
    )
    cuisine_texts = cuisine_texts[cuisine_texts.str.split().str.len() >= 50]

    tfidf_matrix_cuisine = vectorizer.fit_transform(cuisine_texts)
    feature_names_cuisine = vectorizer.get_feature_names_out()

    top_keywords_per_cuisine = {}
    for i, cuisine in enumerate(cuisine_texts.index):
        row = tfidf_matrix_cuisine[i].toarray()[0]
        top_indices = row.argsort()[-10:][::-1]
        top_keywords_per_cuisine[cuisine] = [feature_names_cuisine[j] for j in top_indices]

    for cuisine, keywords in top_keywords_per_cuisine.items():
        print(f"  {cuisine:20s} → {', '.join(keywords)}")
    print()

    # ── SUMMARY ──
    nlp_summary = {
        "total_reviews_loaded":    int(len(reviews)),
        "reviews_used":            int(len(reviews_known)),
        "reviews_excluded":        int(len(reviews) - len(reviews_known)),
        "neighborhoods_covered":   int(reviews_known['area'].nunique()),
        "cuisines_covered":        int(reviews_known['cuisine_primary'].nunique()),
        "avg_sentiment":           round(float(reviews_known['sentiment_score'].mean()), 3),
        "pct_positive":            round(float((reviews_known['sentiment_category'] == 'Positive').mean() * 100), 1),
        "pct_negative":            round(float((reviews_known['sentiment_category'] == 'Negative').mean() * 100), 1),
        "pct_neutral":             round(float((reviews_known['sentiment_category'] == 'Neutral').mean() * 100), 1),
        "source_breakdown":        reviews_known['review_source'].value_counts().to_dict()
    }

    return {
        "summary":                 nlp_summary,
        "sentiment_by_area":       sentiment_by_area,
        "sentiment_by_cuisine":    sentiment_by_cuisine,
        "sentiment_by_price":      sentiment_by_price,
        "top_keywords_per_area":   top_keywords_per_area,
        "top_keywords_per_cuisine":top_keywords_per_cuisine,
    }

# RUN BOTH PIPELINES
nlp_original = run_nlp('../merged/master_reviews.csv',           'ORIGINAL')
nlp_enriched = run_nlp('../machine_learning/master_reviews_enriched.csv',  'ENRICHED')


#original clean output (no ml)
nlp_original['sentiment_by_area'].to_csv('sentiment_by_area.csv')
nlp_original['sentiment_by_cuisine'].to_csv('sentiment_by_cuisine.csv')
nlp_original['sentiment_by_price'].to_csv('sentiment_by_price.csv')
with open('area_keywords.json', 'w') as f:
    json.dump(nlp_original['top_keywords_per_area'], f)
with open('cuisine_keywords.json', 'w') as f:
    json.dump(nlp_original['top_keywords_per_cuisine'], f)
with open('nlp_summary.json', 'w') as f:
    json.dump(nlp_original['summary'], f)

#enriched ml outputs
nlp_enriched['sentiment_by_area'].to_csv('sentiment_by_area_enriched.csv')
nlp_enriched['sentiment_by_cuisine'].to_csv('sentiment_by_cuisine_enriched.csv')
nlp_enriched['sentiment_by_price'].to_csv('sentiment_by_price_enriched.csv')
with open('area_keywords_enriched.json', 'w') as f:
    json.dump(nlp_enriched['top_keywords_per_area'], f)
with open('cuisine_keywords_enriched.json', 'w') as f:
    json.dump(nlp_enriched['top_keywords_per_cuisine'], f)
with open('nlp_summary_enriched.json', 'w') as f:
    json.dump(nlp_enriched['summary'], f)

print("All outputs saved.")