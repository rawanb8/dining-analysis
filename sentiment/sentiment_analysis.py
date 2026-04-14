import pandas as pd
import numpy as np
import os
from textblob import TextBlob
import nltk
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)

CLEANED_PATH = '../cleaned'

print("-------------------SENTIMENT ANALYSIS-------------------------")

def calculate_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return np.nan, np.nan
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return np.nan, np.nan

df = pd.read_csv('../merged/master_reviews.csv')
print(f"Loaded {len(df)} reviews")

df[['sentiment_score', 'sentiment_subjectivity']] = df['review_text'].apply(
lambda x: pd.Series(calculate_sentiment(x)) )

df['sentiment_category'] = pd.cut(
    df['sentiment_score'],
    bins=[-1, -0.1, 0.1, 1],
    labels=['Negative', 'Neutral', 'Positive']
)

print(f"Sentiment distribution:")
for label, count in df['sentiment_category'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")
    
reviews_with_sentiment = df['sentiment_score'].notna().sum()
reviews_with_full_metadata = len(df[
    (df['sentiment_score'].notna()) &
    (df['area'] != 'Unknown') &
    (df['cuisine_primary'] != 'Unknown') &
    (df['price_category'] != 'Unknown')
])

print(f"Reviews analyzed:")
print(f"  Total reviews: {len(df)}")
print(f"  Reviews with sentiment scores: {reviews_with_sentiment}")
print(f"  Reviews with full metadata (area + cuisine + price): {reviews_with_full_metadata}")
df.to_csv('../merged/master_reviews.csv', index=False)

print("✅ SENTIMENT ANALYSIS COMPLETE!")