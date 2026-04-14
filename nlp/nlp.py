import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# part 1: load review csv files into pandas dataframes to analyze everything at once
print("Loading reviews...")
reviews = pd.read_csv('../merged/master_reviews.csv')

reviews = reviews[reviews['review_text_cleaned'].notna()]
reviews = reviews[reviews['review_text_cleaned'].str.strip() != '']

# Remove rows where area/cuisine/price is Unknown
# we might remove??
reviews_known = reviews[
    (reviews['area'] != 'Unknown') &
    (reviews['cuisine_primary'] != 'Unknown') &
    (reviews['price_category'] != 'Unknown')
].copy()

# For TF-IDF, only require area to be known (more data = better keywords)
reviews_for_tfidf = reviews[
    reviews['area'] != 'Unknown'
].copy()

print(f"Total reviews loaded: {len(reviews)}")
print(f"Reviews with full metadata: {len(reviews_known)}")
print()

# step 2: sentiment aggregation. group by area, cuisine, price and calculate average sentiment score
# which neighborhoods/cities get the happiest reviews

print("--------- SENTIMENT BY AREA (neighborhood) --------- ")

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

# Only show areas with enough reviews to be meaningful
sentiment_by_area = sentiment_by_area[sentiment_by_area['review_count'] >= 10]
print(sentiment_by_area.to_string())
print()

# CHART: AVERAGE SENTIMENT BY AREA
plt.figure(figsize=(12, 6))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in sentiment_by_area['avg_sentiment']]
plt.bar(sentiment_by_area.index, sentiment_by_area['avg_sentiment'], color=colors)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title('Average Review Sentiment by Neighborhood', fontsize=14)
plt.ylabel('Sentiment Score (-1 = negative, +1 = positive)')
plt.tight_layout()
plt.savefig('sentiment_by_area.png', dpi=150)
plt.show()
print("Chart saved: sentiment_by_area.png")
print()


print("------SENTIMENT BY CUISINE------")

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

plt.figure(figsize=(12, 6))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in sentiment_by_cuisine['avg_sentiment']]
plt.bar(sentiment_by_cuisine.index, sentiment_by_cuisine['avg_sentiment'], color=colors)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title('Average Review Sentiment by Cuisine Type', fontsize=14)
plt.ylabel('Sentiment Score (-1 = negative, +1 = positive)')
plt.tight_layout()
plt.savefig('sentiment_by_cuisine.png', dpi=150)
plt.show()
print("Chart saved: sentiment_by_cuisine.png")
print()

print("--------- SENTIMENT BY PRICE TIER ---------")

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

plt.figure(figsize=(7, 5))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in sentiment_by_price['avg_sentiment']]
plt.bar(sentiment_by_price.index, sentiment_by_price['avg_sentiment'], color=colors)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Average Review Sentiment by Price Tier', fontsize=14)
plt.ylabel('Sentiment Score (-1 = negative, +1 = positive)')
plt.tight_layout()
plt.savefig('sentiment_by_price.png', dpi=150)
plt.show()
print("Chart saved: sentiment_by_price.png")
print()

# save sentiment outputs as CSVs so the dashboard can load them directly
# without rerunning the heavy NLP computation every time
sentiment_by_area.to_csv('sentiment_by_area.csv')
sentiment_by_cuisine.to_csv('sentiment_by_cuisine.csv')
sentiment_by_price.to_csv('sentiment_by_price.csv')
print("Sentiment CSVs saved.")
print()

# step 3: TF-IDF keyword extraction per area
# these are words that uniquely describe each area vs all others
# stop_words='english': removes "the", "is", "a", "and", etc.
# ngram_range=(1,2): captures single words AND two-word phrases like "good food"
# max_features=5000: only considers the 5000 most common words across all areas
# min_df=2: a word must appear in at least 2 areas to be considered
# custom_stopwords: removes generic review words like "food", "service" that appear everywhere
#                   and don't tell us anything distinctive about a specific area or cuisine
custom_stopwords = list(ENGLISH_STOP_WORDS) + [
    'food', 'service', 'atmosphere', 'place', 'meal',
    'type', 'meal type', 'food service', 'service atmosphere',
    'dine', 'restaurant', 'good', 'great', 'nice', 'order',
    'ordered', 'time', 'went', 'came', 'got', 'like', 'just'
]

vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

print("------TF-IDF KEYWORDS PER AREA------")

# Group all reviews per area into one big text blob per area
area_texts = reviews_known.groupby('area')['review_text_cleaned'].apply(
    lambda texts: ' '.join(texts.tolist())
)

# Only keep areas with enough text to be meaningful
# area_texts = area_texts[area_texts.str.split().str.len() >= 50]
area_texts = reviews_for_tfidf.groupby('area')['review_text_cleaned'].apply(
    lambda texts: ' '.join(texts.tolist())
)


tfidf_matrix = vectorizer.fit_transform(area_texts)
feature_names = vectorizer.get_feature_names_out()

# For each area, find the top 10 words with the highest TF-IDF score
top_keywords_per_area = {}
for i, area in enumerate(area_texts.index):
    row = tfidf_matrix[i].toarray()[0]
    top_indices = row.argsort()[-10:][::-1]
    top_keywords_per_area[area] = [feature_names[j] for j in top_indices]

print("Top keywords per neighborhood:")
for area, keywords in top_keywords_per_area.items():
    print(f"  {area:25s} → {', '.join(keywords)}")
print()

# SAME THING FOR CUISINES
print("------TF-IDF KEYWORDS PER CUISINE------")

# cuisine_texts = reviews_known.groupby('cuisine_primary')['review_text_cleaned'].apply(
#     lambda texts: ' '.join(texts.tolist())
# )
cuisine_texts = reviews_for_tfidf.groupby('cuisine_primary')['review_text_cleaned'].apply(
    lambda texts: ' '.join(texts.tolist())
)
cuisine_texts = cuisine_texts[cuisine_texts.str.split().str.len() >= 50]

# refit vectorizer on cuisine texts
tfidf_matrix_cuisine = vectorizer.fit_transform(cuisine_texts)
feature_names_cuisine = vectorizer.get_feature_names_out()

top_keywords_per_cuisine = {}
for i, cuisine in enumerate(cuisine_texts.index):
    row = tfidf_matrix_cuisine[i].toarray()[0]
    top_indices = row.argsort()[-10:][::-1]
    top_keywords_per_cuisine[cuisine] = [feature_names_cuisine[j] for j in top_indices]

print("Top keywords per cuisine:")
for cuisine, keywords in top_keywords_per_cuisine.items():
    print(f"  {cuisine:20s} → {', '.join(keywords)}")
print()

# save keyword outputs as JSON so the dashboard can load them directly
with open('area_keywords.json', 'w') as f:
    json.dump(top_keywords_per_area, f)
with open('cuisine_keywords.json', 'w') as f:
    json.dump(top_keywords_per_cuisine, f)
print("Keyword JSONs saved.")
print()

# STEP 4: QUICK SUMMARY
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Total reviews analyzed: {len(reviews_known)}")
print(f"Neighborhoods covered:  {reviews_known['area'].nunique()}")
print(f"Cuisine types covered:  {reviews_known['cuisine_primary'].nunique()}")
print(f"Overall avg sentiment:  {reviews_known['sentiment_score'].mean():.3f}")
print(f"% Positive reviews:     {(reviews_known['sentiment_category'] == 'Positive').mean()*100:.1f}%")
print(f"% Negative reviews:     {(reviews_known['sentiment_category'] == 'Negative').mean()*100:.1f}%")

print(f"Total reviews loaded: {len(reviews)}")
print(f"Reviews with full metadata: {len(reviews_known)}")
print(f"Unique areas in reviews_known: {reviews_known['area'].nunique()}")
print(f"Unique cuisines in reviews_known: {reviews_known['cuisine_primary'].nunique()}")
print(f"Source breakdown:")
print(reviews_known['review_source'].value_counts())
print()