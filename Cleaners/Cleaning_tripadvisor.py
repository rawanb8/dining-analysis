import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️  TextBlob not installed. Run: pip install textblob")

INPUT_FILE = '../data/restaurants_with_reviews.csv'
OUTPUT_RESTAURANTS = '../cleaned/Tripadvisor_restaurants_clean.csv'
OUTPUT_REVIEWS = '../cleaned/Tripadvisor_reviews_clean.csv'

os.makedirs('../cleaned', exist_ok=True)

print("="*70)
print("🧹 SOURCE 3: TRIPADVISOR DATA CLEANING")
print("="*70)
print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_RESTAURANTS}")
print(f"        {OUTPUT_REVIEWS}\n")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Loaded {len(df)} rows ({df['restaurant_name'].nunique()} unique restaurants)\n")

# STEP 1: SPLIT RESTAURANT INFO FROM REVIEWS

print("🏪 STEP 1: Aggregating restaurant info...")

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

restaurant_df = restaurant_df.sort_values('restaurant_name').reset_index(drop=True)
restaurant_df['restaurant_id'] = ['src3_' + str(i + 1).zfill(3) for i in range(len(restaurant_df))]

# Build mapping for reviews
loc_to_id = dict(zip(restaurant_df['location_id'], restaurant_df['restaurant_id']))

print(f"✓ Found {len(restaurant_df)} unique restaurants\n")

# STEP 2: CLEAN BASIC FIELDS

print("🏷️  STEP 2: Cleaning basic fields...")

restaurant_df['name'] = restaurant_df['restaurant_name'].str.strip()

def parse_city_field(city_raw):
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
print(f"✓ Found {unique_cities} cities, {unique_areas} areas\n")

# STEP 3: SET UNAVAILABLE FIELDS

print("📋 STEP 3: Setting unavailable fields...")

# Cuisine and phone not available in TripAdvisor data
restaurant_df['cuisine_primary'] = 'Unknown'
restaurant_df['cuisine_tags'] = 'Unknown'
restaurant_df['phone'] = np.nan

print(f"✓ Cuisine and phone: Not available\n")

# STEP 4: PROCESS RATINGS

print("⭐ STEP 4: Processing ratings...")

restaurant_df['rating_tripadvisor'] = pd.to_numeric(restaurant_df['rating'], errors='coerce')
restaurant_df['rating_google'] = np.nan
restaurant_df['rating_overall'] = restaurant_df['rating_tripadvisor']

restaurant_df['review_count_tripadvisor'] = pd.to_numeric(restaurant_df['num_reviews'], errors='coerce').fillna(0).astype(int)
restaurant_df['review_count_google'] = 0
restaurant_df['review_count_total'] = restaurant_df['review_count_tripadvisor']

print(f"✓ Average rating: {restaurant_df['rating_overall'].mean():.2f}/5.0")
print(f"✓ Total reviews: {restaurant_df['review_count_total'].sum():,}\n")

# STEP 5: CALCULATE STAR DISTRIBUTION

print("🌟 STEP 5: Calculating star distribution...")

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

print(f"✓ Average 5-star: {restaurant_df['star_5_percent'].mean():.1f}%")
print(f"✓ Average 1-star: {restaurant_df['star_1_percent'].mean():.1f}%\n")

# STEP 6: SET PRICE & HOURS

print("💰 STEP 6: Setting price and hours...")

restaurant_df['price_category'] = 'Unknown'

for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
    restaurant_df[f'hours_{day}'] = 'Unknown'

print(f"✓ Not available in source data\n")

# STEP 7: EXTRACT FEATURES FROM REVIEWS

print("💡 STEP 7: Extracting features from reviews...")

# Collect all review text for each restaurant
restaurant_reviews = df.groupby('location_id')['review_text'].apply(
    lambda texts: ' '.join([str(t).lower() for t in texts if pd.notna(t)])
).to_dict()

def extract_features_from_reviews(location_id):
    """Extract features from aggregated review text"""
    reviews_text = restaurant_reviews.get(location_id, '')
    
    features = {}
    
    # Basic operational features
    features['delivery_available'] = 'TRUE' if 'delivery' in reviews_text else 'Unknown'
    features['outdoor_seating'] = 'TRUE' if any(word in reviews_text for word in ['outdoor', 'terrace', 'patio', 'rooftop', 'garden']) else 'Unknown'
    features['reservation_required'] = 'TRUE' if any(word in reviews_text for word in ['reservation', 'book', 'reserv']) else 'Unknown'
    
    # Payment options
    features['cash_only'] = 'TRUE' if 'cash only' in reviews_text else 'Unknown'
    features['credit_cards_accepted'] = 'TRUE' if any(word in reviews_text for word in ['credit card', 'cards accepted', 'accept cards']) else 'Unknown'
    
    # Amenities
    features['wifi_available'] = 'TRUE' if any(word in reviews_text for word in ['wifi', 'wi-fi', 'internet', 'free wifi']) else 'Unknown'
    features['parking_available'] = 'TRUE' if any(word in reviews_text for word in ['parking', 'valet', 'park available', 'free parking']) else 'Unknown'
    features['wheelchair_accessible'] = 'TRUE' if any(word in reviews_text for word in ['wheelchair', 'accessible', 'handicap']) else 'Unknown'
    features['takeaway_available'] = 'TRUE' if any(word in reviews_text for word in ['takeaway', 'take away', 'take-away', 'to go', 'to-go']) else 'Unknown'
    
    # Experience features
    features['live_music'] = 'TRUE' if any(word in reviews_text for word in ['live music', 'live band', 'music performance', 'dj', 'piano', 'singer', 'belly dancer', 'dancers']) else 'Unknown'
    features['pet_friendly'] = 'TRUE' if any(word in reviews_text for word in ['pet friendly', 'pet-friendly', 'dog friendly', 'dogs allowed', 'pets welcome']) else 'Unknown'
    features['kids_friendly'] = 'TRUE' if any(word in reviews_text for word in ['kids', 'children', 'family-friendly', 'playground', 'kids menu', 'child-friendly']) else 'Unknown'
    
    return features

features_df = restaurant_df['location_id'].apply(extract_features_from_reviews).apply(pd.Series)
restaurant_df = pd.concat([restaurant_df, features_df], axis=1)

for feature in features_df.columns:
    count = (features_df[feature] == 'TRUE').sum()
    if count > 0:
        print(f"   ✓ {feature.replace('_', ' ')}: {count}")

print()

# STEP 8: SET REMAINING FIELDS

print("📋 STEP 8: Setting remaining fields...")

restaurant_df['menu_items'] = ''
restaurant_df['menu_link'] = np.nan

restaurant_df['data_source'] = 'source3'
restaurant_df['source_url'] = restaurant_df['tripadvisor_url']
restaurant_df['website'] = restaurant_df['website']
restaurant_df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
restaurant_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')

print(f"✓ Metadata added\n")

# STEP 9: CREATE FINAL RESTAURANTS TABLE

print("📊 STEP 9: Creating final restaurants table...")

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
    'cash_only', 'credit_cards_accepted', 'wifi_available', 'wheelchair_accessible', 'takeaway_available',
    'parking_available', 'live_music', 'pet_friendly', 'kids_friendly',
    'menu_items', 'menu_link', 'website',
    'data_source', 'source_url', 'scraped_date', 'last_updated'
]

restaurants_final = restaurant_df[restaurants_columns].copy()

print(f"✓ Created table: {len(restaurants_final)} rows × {len(restaurants_columns)} columns\n")

# STEP 10: EXTRACT INDIVIDUAL REVIEWS

print("💬 STEP 10: Extracting individual reviews...")

def clean_review_text(text):
    """Clean review text: fix encoding, whitespace, etc."""
    if pd.isna(text):
        return ''
    text = str(text)

    encoding_fixes = {
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': '—', 'Ã©': 'é', 'Ã¨': 'è',
        'Ã ': 'à', 'Â ': ' ', 'ðŸ': ''
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
        'review_id': f'rev_{len(reviews_list) + 1:05d}',
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

print(f"✓ Extracted {len(reviews_df)} reviews")
print(f"✓ Avg review length: {reviews_df['word_count'].mean():.0f} words\n")

reviews_df['sentiment_score'] = np.nan
reviews_df['sentiment_subjectivity'] = np.nan
reviews_df['sentiment_category'] = 'Unknown'

# Attach restaurant metadata to reviews
restaurant_meta = restaurants_final[['restaurant_id', 'area', 'cuisine_primary', 'price_category']].copy()
reviews_df = reviews_df.merge(restaurant_meta, on='restaurant_id', how='left')
reviews_df['area'] = reviews_df['area'].fillna('Unknown')
reviews_df['cuisine_primary'] = reviews_df['cuisine_primary'].fillna('Unknown')
reviews_df['price_category'] = reviews_df['price_category'].fillna('Unknown')

# STEP 12: SAVE CLEANED DATA

print("💾 STEP 12: Saving cleaned data...")

restaurants_final.to_csv(OUTPUT_RESTAURANTS, index=False)
print(f"✓ Saved: {OUTPUT_RESTAURANTS}")
print(f"  ({len(restaurants_final)} restaurants)\n")

reviews_df.to_csv(OUTPUT_REVIEWS, index=False)
print(f"✓ Saved: {OUTPUT_REVIEWS}")
print(f"  ({len(reviews_df)} reviews)\n")

# FINAL SUMMARY

print("="*70)
print("✅ SOURCE 3 CLEANING COMPLETE!")
print("="*70)
print(f"""
📊 SUMMARY:
   • Restaurants: {len(restaurants_final)}
   • Reviews: {len(reviews_df)}
   • Cities: {restaurant_df['city'].nunique()}
   • Average rating: {restaurants_final['rating_overall'].mean():.2f}/5.0
   • Total review count: {restaurants_final['review_count_total'].sum():,}
   • Data completeness: {((restaurants_final['rating_overall'].notna().sum() + restaurants_final['area'].notna().sum()) / (2 * len(restaurants_final)) * 100):.0f}%

""")
print("="*70)