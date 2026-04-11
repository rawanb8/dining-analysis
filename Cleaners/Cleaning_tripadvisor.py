import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

import nltk
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️  TextBlob not installed. Run: pip install textblob")

INPUT_FILE = '../restaurants_with_reviews.csv'
OUTPUT_RESTAURANTS = '../cleaned/Tripadvisor_restaurants_clean.csv'
OUTPUT_REVIEWS = '../cleaned/Tripadvisor_reviews_clean.csv'

os.makedirs('../cleaned', exist_ok=True)

print("cleaning tripadvisor")
df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows ({df['restaurant_name'].nunique()} unique restaurants)")

#split restaurant info from reviews

restaurant_df = (
    df.groupby('location_id')
    .agg({
        'restaurant_name': 'first',
        'city': 'first',
        'address': 'first',
        'website': 'first',
        'tripadvisor_url': 'first',
        'rating': 'first',
        'num_reviews': 'first',
    })
    .reset_index()
)
#uniqye restaurant ids
restaurant_df = restaurant_df.sort_values('restaurant_name').reset_index(drop=True)
restaurant_df['restaurant_id'] = ['src3_' + str(i + 1).zfill(3) for i in range(len(restaurant_df))]

# Build mapping for reviews
loc_to_id = dict(zip(restaurant_df['location_id'], restaurant_df['restaurant_id']))

#restaurant names
restaurant_df['name'] = restaurant_df['restaurant_name'].str.strip()

def parse_city_field(city_raw):
    """Parse 'Beirut Lebanon' → city='Beirut', country='Lebanon'"""
    if pd.isna(city_raw):
        return 'Unknown', 'Lebanon'
    parts = str(city_raw).strip().rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return str(city_raw).strip(), 'Lebanon'

restaurant_df[['city_clean', 'country']] = restaurant_df['city'].apply(
    lambda x: pd.Series(parse_city_field(x))
)

def parse_address(row):
    """Extract area from the full address string."""
    addr = row['address']
    if pd.isna(addr):
        return {'address_full': '', 'area': 'Unknown'}

    addr = str(addr).strip()
    parts = [p.strip() for p in addr.split(',')]

    # Try to find an area name (not the country or city already known)
    city_lower = str(row['city_clean']).lower()
    area = 'Unknown'
    for part in parts:
        part_lower = part.lower()
        if ('lebanon' not in part_lower
                and city_lower not in part_lower
                and not re.match(r'^\d+$', part.strip())):
            area = part.strip()
            break

    return {'address_full': addr, 'area': area}

addr_parsed = restaurant_df.apply(parse_address, axis=1).apply(pd.Series)
restaurant_df['address_full'] = addr_parsed['address_full']
restaurant_df['area'] = addr_parsed['area']
restaurant_df['city'] = restaurant_df['city_clean']

unique_areas = restaurant_df['area'].nunique()
unique_cities = restaurant_df['city'].nunique()
print(f"   ✓ Found {unique_cities} cities, {unique_areas} areas")
print()

#cuisine and phone nb not available
restaurant_df['cuisine_primary'] = 'Unknown'
restaurant_df['cuisine_tags'] = 'Unknown'
restaurant_df['phone'] = np.nan

restaurant_df['rating_tripadvisor'] = pd.to_numeric(restaurant_df['rating'], errors='coerce')
restaurant_df['rating_google'] = np.nan
restaurant_df['rating_overall'] = restaurant_df['rating_tripadvisor']

restaurant_df['review_count_tripadvisor'] = pd.to_numeric(restaurant_df['num_reviews'], errors='coerce').fillna(0).astype(int)
restaurant_df['review_count_google'] = 0
restaurant_df['review_count_total'] = restaurant_df['review_count_tripadvisor']


star_dist = (
    df.groupby('location_id')['review_rating']
    .value_counts()
    .unstack(fill_value=0)
)

for star in range(1, 6):
    col = float(star)
    if col not in star_dist.columns:
        star_dist[col] = 0

star_dist = star_dist.rename(columns={float(i): f'star_{i}_count' for i in range(1, 6)})
star_cols = [f'star_{i}_count' for i in range(1, 6)]
star_dist = star_dist[star_cols].reset_index()

restaurant_df = restaurant_df.merge(star_dist, on='location_id', how='left')

for i in range(1, 6):
    restaurant_df[f'star_{i}_count'] = restaurant_df[f'star_{i}_count'].fillna(0).astype(int)

total_stars = restaurant_df[star_cols].sum(axis=1)
for i in range(5, 0, -1):
    restaurant_df[f'star_{i}_percent'] = (
        (restaurant_df[f'star_{i}_count'] / total_stars.replace(0, np.nan)) * 100
    ).round(1).fillna(0)

print(f"   ✓ Average 5-star: {restaurant_df['star_5_percent'].mean():.1f}%")
print(f"   ✓ Average 1-star: {restaurant_df['star_1_percent'].mean():.1f}%")
print()

#price category, working hours, and features not available
restaurant_df['price_category'] = 'Unknown'

for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
    restaurant_df[f'hours_{day}'] = 'Unknown'
print(f"   ✓ Not available in source data")
print()


restaurant_df['delivery_available'] = 'Unknown'
restaurant_df['outdoor_seating'] = 'Unknown'
restaurant_df['reservation_required'] = 'Unknown'


restaurant_df['description'] = ''
restaurant_df['menu_items'] = ''
restaurant_df['menu_link'] = np.nan
restaurant_df['why_to_go'] = ''
restaurant_df['reviews_summary'] = ''
restaurant_df['tips'] = ''

restaurant_df['data_source'] = 'source3'
restaurant_df['source_url'] = restaurant_df['tripadvisor_url']
restaurant_df['website'] = restaurant_df['website']
restaurant_df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
restaurant_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')


restaurants_columns = [
    'restaurant_id', 'name',
    'cuisine_primary', 'cuisine_tags',
    'address_full', 'area', 'city', 'country', 'phone',
    'rating_overall', 'rating_google', 'rating_tripadvisor',
    'review_count_total', 'review_count_google', 'review_count_tripadvisor',
    'star_5_count', 'star_4_count', 'star_3_count', 'star_2_count', 'star_1_count',
    'star_5_percent', 'star_4_percent', 'star_3_percent', 'star_2_percent', 'star_1_percent',
    'price_category',
    'hours_monday', 'hours_tuesday', 'hours_wednesday', 'hours_thursday',
    'hours_friday', 'hours_saturday', 'hours_sunday',
    'delivery_available', 'outdoor_seating', 'reservation_required',
    'description', 'menu_items', 'menu_link', 'why_to_go', 'reviews_summary', 'tips', 'website',
    'data_source', 'source_url', 'scraped_date', 'last_updated'
]

restaurants_final = restaurant_df[restaurants_columns].copy()

print(f"   ✓ Created table: {len(restaurants_final)} rows × {len(restaurants_columns)} columns")
print()

#EXTRACT & CLEAN INDIVIDUAL REVIEWS
def clean_review_text(text):
    """Clean review text: fix encoding, whitespace, etc."""
    if pd.isna(text):
        return ''
    text = str(text)

    encoding_fixes = {
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': '—', 'Ã©': 'é', 'Ã¨': 'è',
        'Ã ': 'à', 'Â ': ' '
    }
    for wrong, right in encoding_fixes.items():
        text = text.replace(wrong, right)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

reviews_list = []
for idx, row in df.iterrows():
    rid = loc_to_id.get(row['location_id'], 'src3_unknown')
    rname = str(row['restaurant_name']).strip()

    raw_text = row.get('review_text', '')
    title = row.get('review_title', '')
    rating = row.get('review_rating', np.nan)

    cleaned = clean_review_text(raw_text)
    title_clean = clean_review_text(title)

    # Combine title + text for full review
    full_text = f"{title_clean}: {cleaned}" if title_clean and cleaned else (cleaned or title_clean)

    if not full_text:
        continue

    reviews_list.append({
        'review_id': f'rev_src3_{len(reviews_list) + 1:05d}',
        'restaurant_id': rid,
        'restaurant_name': rname,
        'review_text': f"[{title_clean}] {raw_text}" if title_clean else raw_text,
        'review_text_cleaned': full_text,
        'rating': float(rating) if pd.notna(rating) else np.nan,
        'review_date': np.nan, 
        'review_source': 'TripAdvisor',
        'word_count': len(full_text.split()),
    })

reviews_df = pd.DataFrame(reviews_list)

print("😊 STEP 17: Analyzing sentiment...")

if SENTIMENT_AVAILABLE:
    def calculate_sentiment(text):
        if pd.isna(text) or text == '':
            return np.nan, np.nan
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return np.nan, np.nan

    reviews_df[['sentiment_score', 'sentiment_subjectivity']] = reviews_df['review_text_cleaned'].apply(
        lambda x: pd.Series(calculate_sentiment(x))
    )

    reviews_df['sentiment_category'] = pd.cut(
        reviews_df['sentiment_score'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )

    print(f"Sentiment distribution:")
    for sentiment, count in reviews_df['sentiment_category'].value_counts().items():
        pct = (count / len(reviews_df)) * 100
        print(f"      {sentiment}: {count} ({pct:.1f}%)")
else:
    reviews_df['sentiment_score'] = np.nan
    reviews_df['sentiment_subjectivity'] = np.nan
    reviews_df['sentiment_category'] = 'Unknown'
   


restaurants_final.to_csv(OUTPUT_RESTAURANTS, index=False)

reviews_df.to_csv(OUTPUT_REVIEWS, index=False)

# FINAL SUMMARY
print("✅ CLEANING COMPLETE!")
print(f"""
📊 SUMMARY:
   • Restaurants: {len(restaurants_final)}
   • Reviews: {len(reviews_df)}
   • Cities: {restaurant_df['city'].nunique()}
   • Average rating: {restaurants_final['rating_overall'].mean():.2f}/5.0
   • Total review count (metadata): {restaurants_final['review_count_total'].sum():,}

""")
