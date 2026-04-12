import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

SENTIMENT_AVAILABLE = False
try:
    from textblob import TextBlob
    # Test if corpora is downloaded
    test_blob = TextBlob("test")
    _ = test_blob.sentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("⚠️  TextBlob not installed. Run: pip install textblob")
except LookupError:
    print("⚠️  TextBlob corpora not downloaded. Downloading now...")
    try:
        import nltk
        nltk.download('brown', quiet=True)
        nltk.download('punkt', quiet=True)
        from textblob import TextBlob
        SENTIMENT_AVAILABLE = True
        print("✅  TextBlob corpora downloaded successfully!")
    except:
        print("⚠️  Could not download corpora. Run: python -m textblob.download_corpora")


INPUT_FILE = '../data/wandor_restaurants.csv'  
OUTPUT_RESTAURANTS = '../cleaned/Wandorlog_restaurants_clean.csv'  
OUTPUT_REVIEWS = '../cleaned/Wandorlog_reviews_clean.csv'   

os.makedirs('../cleaned', exist_ok=True)

print("="*80)
print("🧹 SOURCE 1 DATA CLEANING - BEIRUT RESTAURANTS")
print("="*80)
print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_RESTAURANTS}")
print(f"        {OUTPUT_REVIEWS}")
print()

# STEP 1: LOAD DATA

print("📂 STEP 1: Loading data...")

df = pd.read_csv(INPUT_FILE)

print(f"   ✓ Loaded {len(df)} restaurants")
print(f"   ✓ Original columns: {len(df.columns)}")
print()

# STEP 2: CREATE UNIQUE RESTAURANT IDs

print("🔑 STEP 2: Creating unique restaurant IDs...")

# Format: src1_001, src1_002, etc.
df['restaurant_id'] = ['src1_' + str(i+1).zfill(3) for i in range(len(df))]

print(f"   ✓ Created IDs from src1_001 to src1_{len(df):03d}")
print()


# STEP 3: CLEAN BASIC FIELDS

print("🏷️  STEP 3: Cleaning basic fields...")

# Name - trim whitespace
df['name'] = df['name'].str.strip()


# STEP 4: PARSE CUISINE

print("🍽️  STEP 4: Parsing cuisine information...")

def parse_cuisine(cuisine):
    """
    Split cuisine into primary (first item) and tags (all items)
    Example: "Lebanese, Mediterranean" → primary: Lebanese, tags: Lebanese,Mediterranean
    """
    if pd.isna(cuisine):
        return 'Unknown', 'Unknown'
    
    cuisines = [c.strip() for c in str(cuisine).split(',')]
    primary = cuisines[0]
    tags = ','.join(cuisines)
    
    return primary, tags

# Create new columns
df[['cuisine_primary', 'cuisine_tags']] = df['cuisine'].apply(
    lambda x: pd.Series(parse_cuisine(x))
)

unique_cuisines = df['cuisine_primary'].nunique()
print(f"   ✓ Found {unique_cuisines} unique cuisine types")
print(f"   ✓ Most common: {df['cuisine_primary'].mode()[0]}")
print()


# STEP 5: PARSE ADDRESS

print("📍 STEP 5: Parsing addresses...")

def parse_address(address):
    """
    Extract components from address string
    Example: "Hamra Street, Beirut, Lebanon" → area: Hamra Street, city: Beirut
    """
    if pd.isna(address):
        return {
            'address_full': '',
            'area': 'Unknown',
            'city': 'Beirut',
            'country': 'Lebanon'
        }
    
    parts = [p.strip() for p in str(address).split(',')]
    
    if len(parts) >= 3:
        return {
            'address_full': address,
            'area': parts[-3],  # Third from end
            'city': parts[-2],  # Second from end
            'country': parts[-1]  # Last
        }
    elif len(parts) == 2:
        return {
            'address_full': address,
            'area': parts[0],
            'city': parts[1] if 'Beirut' in parts[1] else 'Beirut',
            'country': 'Lebanon'
        }
    else:
        return {
            'address_full': address,
            'area': 'Unknown',
            'city': 'Beirut',
            'country': 'Lebanon'
        }

# Apply parsing
address_df = df['address'].apply(parse_address).apply(pd.Series)
df = pd.concat([df, address_df], axis=1)

unique_areas = df['area'].nunique()
print(f"   ✓ Extracted {unique_areas} unique areas/neighborhoods")
print()


# STEP 6: CLEAN PHONE NUMBERS

print("📞 STEP 6: Standardizing phone numbers...")

def clean_phone(phone):
    """
    Standardize to +961XXXXXXXX format (Lebanon country code)
    """
    if pd.isna(phone):
        return np.nan
    
    # Remove all non-digit characters except +
    phone = re.sub(r'[^\d+]', '', str(phone))
    
    # Already correct format
    if phone.startswith('+961'):
        return phone
    
    # Has 961 but missing +
    if phone.startswith('961'):
        return '+' + phone
    
    # Local number - add country code
    if len(phone) >= 7:
        return '+961' + phone.lstrip('0')
    
    return np.nan

df['phone'] = df['phone'].apply(clean_phone)

valid_phones = df['phone'].notna().sum()
print(f"   ✓ Standardized {valid_phones}/{len(df)} phone numbers")
print()


# STEP 7: PROCESS RATINGS

print("⭐ STEP 7: Processing ratings...")

# Ratings are already on 0-5 scale, just ensure float type
df['rating_google'] = pd.to_numeric(df['google_rating'], errors='coerce')
df['rating_tripadvisor'] = pd.to_numeric(df['tripadvisor_rating'], errors='coerce')

# Calculate weighted average rating
def calc_weighted_rating(row):
    """
    Calculate overall rating as weighted average of Google and TripAdvisor
    Weight by number of reviews
    """
    ratings = []
    weights = []
    
    if pd.notna(row['rating_google']):
        ratings.append(row['rating_google'])
        weights.append(row['google_reviews'] if pd.notna(row['google_reviews']) else 1)
    
    if pd.notna(row['rating_tripadvisor']):
        ratings.append(row['rating_tripadvisor'])
        weights.append(row['tripadvisor_reviews'] if pd.notna(row['tripadvisor_reviews']) else 1)
    
    if not ratings:
        return np.nan
    
    return round(np.average(ratings, weights=weights), 2)

df['rating_overall'] = df.apply(calc_weighted_rating, axis=1)

# Review counts
df['review_count_google'] = pd.to_numeric(df['google_reviews'], errors='coerce').fillna(0).astype(int)
df['review_count_tripadvisor'] = pd.to_numeric(df['tripadvisor_reviews'], errors='coerce').fillna(0).astype(int)
df['review_count_total'] = df['review_count_google'] + df['review_count_tripadvisor']

print(f"   ✓ Average overall rating: {df['rating_overall'].mean():.2f}/5.0")
print(f"   ✓ Total reviews: {df['review_count_total'].sum():,}")
print()


# STEP 8: STAR DISTRIBUTION

print("🌟 STEP 8: Calculating star distribution...")

# Convert to numeric
df['star_5_count'] = pd.to_numeric(df['star_5'], errors='coerce').fillna(0).astype(int)
df['star_4_count'] = pd.to_numeric(df['star_4'], errors='coerce').fillna(0).astype(int)
df['star_3_count'] = pd.to_numeric(df['star_3'], errors='coerce').fillna(0).astype(int)
df['star_2_count'] = pd.to_numeric(df['star_2'], errors='coerce').fillna(0).astype(int)
df['star_1_count'] = pd.to_numeric(df['star_1'], errors='coerce').fillna(0).astype(int)

# Calculate total
star_cols = ['star_5_count', 'star_4_count', 'star_3_count', 'star_2_count', 'star_1_count']
df['total_star_reviews'] = df[star_cols].sum(axis=1)

# Calculate percentages
for i in range(5, 0, -1):
    df[f'star_{i}_percent'] = (
        (df[f'star_{i}_count'] / df['total_star_reviews'].replace(0, np.nan)) * 100
    ).round(1).fillna(0)

print(f"   ✓ Average 5-star: {df['star_5_percent'].mean():.1f}%")
print(f"   ✓ Average 1-star: {df['star_1_percent'].mean():.1f}%")
print()


# STEP 9: PARSE WORKING HOURS

print("🕐 STEP 9: Parsing working hours...")

def parse_hours(hours_text):
    """
    Parse working hours into individual day columns
    Example: "Monday: 9AM-11PM | Tuesday: 9AM-11PM" → dict of hours per day
    """
    days = {
        'hours_monday': 'Unknown',
        'hours_tuesday': 'Unknown',
        'hours_wednesday': 'Unknown',
        'hours_thursday': 'Unknown',
        'hours_friday': 'Unknown',
        'hours_saturday': 'Unknown',
        'hours_sunday': 'Unknown'
    }
    
    if pd.isna(hours_text):
        return days
    
    # Find pattern: DayName: hours
    pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday):\s*([^|]+)'
    matches = re.findall(pattern, str(hours_text), re.IGNORECASE)
    
    for day, hours in matches:
        day_key = f'hours_{day.lower()}'
        hours_clean = hours.strip().replace('–', '-').replace('â€"', '-')
        days[day_key] = hours_clean
    
    return days

hours_df = df['working_hours'].apply(parse_hours).apply(pd.Series)
df = pd.concat([df, hours_df], axis=1)

valid_hours = (hours_df != 'Unknown').any(axis=1).sum()
print(f"   ✓ Parsed hours for {valid_hours}/{len(df)} restaurants")
print()


# STEP 10: INFER PRICE CATEGORY

print("💰 STEP 10: Inferring price categories...")

def infer_price(description, why_go):
    """
    Infer price category from description keywords
    Categories: Budget, Mid-Range, High-End
    """
    text = str(description) + ' ' + str(why_go)
    text = text.lower()
    
    # High-end keywords
    high_end_keywords = [
        'upscale', 'fine-dining', 'fine dining', 'luxury', 'luxurious',
        'elegant', 'sophisticated', 'expensive', 'high-end', 'premium'
    ]
    
    # Budget keywords
    budget_keywords = [
        'cheap', 'affordable', 'budget', 'inexpensive', 'economical',
        'value for money', 'good value', 'reasonably priced'
    ]
    
    high_end_score = sum(1 for keyword in high_end_keywords if keyword in text)
    budget_score = sum(1 for keyword in budget_keywords if keyword in text)
    
    if high_end_score >= 2:
        return 'High-End'
    elif budget_score >= 2:
        return 'Budget'
    else:
        return 'Mid-Range'

df['price_category'] = df.apply(
    lambda row: infer_price(row['description'], row['why_to_go']), 
    axis=1
)

print(f"   ✓ Price distribution:")
for price, count in df['price_category'].value_counts().items():
    print(f"      {price}: {count}")
print()


# STEP 11: CLEAN TEXT FIELDS

print("📝 STEP 11: Cleaning text content...")

def clean_text(text):
    """
    Remove encoding issues and clean text
    """
    if pd.isna(text):
        return ''
    
    text = str(text)
    
    # Fix common encoding issues
    encoding_fixes = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'ðŸ': '',
        'Â ': ' '
    }
    
    for wrong, right in encoding_fixes.items():
        text = text.replace(wrong, right)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove "About\n" prefix
    text = re.sub(r'^About\s*\n', '', text)
    
    return text

# Apply to text columns
text_columns = ['description', 'why_to_go', 'reviews_summary', 'tips', 'reviews']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

print(f"   ✓ Cleaned {len(text_columns)} text columns")
print()


# STEP 12: PARSE MENU ITEMS

print("🍽️  STEP 12: Parsing menu items...")

def clean_menu_items(items):
    """Clean comma-separated menu items"""
    if pd.isna(items):
        return ''
    items_list = [item.strip() for item in str(items).split(',')]
    return ','.join(items_list)

df['menu_items'] = df['menu_items'].apply(clean_menu_items)

restaurants_with_menu = (df['menu_items'] != '').sum()
print(f"   ✓ {restaurants_with_menu}/{len(df)} restaurants have menu items")
print()


# STEP 13: EXTRACT FEATURES

print("💡 STEP 13: Extracting features from text and reviews...")

def extract_features(row):
    """Extract features from description, why_to_go, tips, reviews_summary, and reviews"""
    
    text_fields = [
        str(row.get('description', '')),
        str(row.get('why_to_go', '')),
        str(row.get('tips', '')),
        str(row.get('reviews_summary', '')),
        str(row.get('reviews', ''))
    ]
    full_text = ' '.join(text_fields).lower()
    
    features = {}
    
    features['delivery_available'] = 'TRUE' if 'delivery' in full_text else 'Unknown'
    features['outdoor_seating'] = 'TRUE' if any(word in full_text for word in ['outdoor', 'terrace', 'patio', 'rooftop', 'garden']) else 'Unknown'
    features['reservation_required'] = 'TRUE' if any(word in full_text for word in ['reservation', 'book', 'reserv']) else 'Unknown'
    features['cash_only'] = 'TRUE' if 'cash only' in full_text else 'Unknown'
    features['credit_cards_accepted'] = 'TRUE' if any(word in full_text for word in ['credit card', 'cards accepted', 'accept cards']) else 'Unknown'
    features['wifi_available'] = 'TRUE' if any(word in full_text for word in ['wifi', 'wi-fi', 'internet', 'free wifi']) else 'Unknown'
    features['parking_available'] = 'TRUE' if any(word in full_text for word in ['parking', 'valet', 'park available', 'free parking']) else 'Unknown'
    features['wheelchair_accessible'] = 'TRUE' if any(word in full_text for word in ['wheelchair', 'accessible', 'handicap']) else 'Unknown'
    features['takeaway_available'] = 'TRUE' if any(word in full_text for word in ['takeaway', 'take away', 'take-away', 'to go', 'to-go']) else 'Unknown'
    features['live_music'] = 'TRUE' if any(word in full_text for word in ['live music', 'live band', 'music performance', 'dj', 'piano', 'singer']) else 'Unknown'
    features['pet_friendly'] = 'TRUE' if any(word in full_text for word in ['pet friendly', 'pet-friendly', 'dog friendly', 'dogs allowed', 'pets welcome']) else 'Unknown'
    features['kids_friendly'] = 'TRUE' if any(word in full_text for word in ['kids', 'children', 'family-friendly', 'playground', 'kids menu', 'child-friendly']) else 'Unknown'
    
    return features

features_df = df.apply(extract_features, axis=1, result_type='expand')
df = pd.concat([df, features_df], axis=1)

for feature in features_df.columns:
    count = (features_df[feature] == 'TRUE').sum()
    if count > 0:
        print(f"   ✓ {feature.replace('_', ' ')}: {count}")

print()


# STEP 14: ADD METADATA

print("📋 STEP 14: Adding metadata...")

df['data_source'] = 'source1'
df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
df['source_url'] = df['address_link']

print(f"   ✓ Added metadata fields")
print()


# STEP 15: CREATE FINAL RESTAURANTS TABLE

print("📊 STEP 15: Creating final restaurants table...")

# Select columns in target schema order
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

restaurants_final = df[restaurants_columns].copy()

print(f"   ✓ Created table: {len(restaurants_final)} rows × {len(restaurants_columns)} columns")
print()


# STEP 16: EXTRACT INDIVIDUAL REVIEWS

print("💬 STEP 16: Extracting individual reviews...")

reviews_list = []

for idx, row in df.iterrows():
    restaurant_id = row['restaurant_id']
    restaurant_name = row['name']
    reviews_text = row['reviews']
    
    if pd.isna(reviews_text):
        continue
    
    # Split by "||"
    individual_reviews = str(reviews_text).split('||')
    
    for review in individual_reviews:
        review = review.strip()
        if not review:
            continue
        
        # Parse: [5/5] [May 18, 2025 from Google]: Review text...
        rating_match = re.search(r'\[(\d+)/(\d+)\]', review)
        date_match = re.search(r'\[(.*?)\s+from\s+(.*?)\]:', review)
        
        rating = np.nan
        review_date = np.nan
        review_source = 'Unknown'
        
        if rating_match:
            rating = float(rating_match.group(1))
        
        if date_match:
            try:
                date_str = date_match.group(1).strip()
                review_date = pd.to_datetime(date_str, format='%b %d, %Y').strftime('%Y-%m-%d')
            except:
                review_date = np.nan
            review_source = date_match.group(2).strip()
        
        # Extract clean text
        review_text_clean = re.sub(r'\[\d+/\d+\]\s*\[.*?\]:\s*', '', review).strip()
        
        reviews_list.append({
            'review_id': f'rev_{len(reviews_list)+1:05d}',
            'restaurant_id': restaurant_id,
            'restaurant_name': restaurant_name,
            'review_text': review,
            'review_text_cleaned': review_text_clean,
            'rating': rating,
            'review_date': review_date,
            'review_source': review_source,
            'word_count': len(review_text_clean.split())
        })

reviews_df = pd.DataFrame(reviews_list)

print(f"   ✓ Extracted {len(reviews_df)} individual reviews")
print(f"   ✓ Avg review length: {reviews_df['word_count'].mean():.0f} words")
print()


# STEP 17: SENTIMENT ANALYSIS

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
    
    print(f"   ✓ Sentiment distribution:")
    for sentiment, count in reviews_df['sentiment_category'].value_counts().items():
        pct = (count / len(reviews_df)) * 100
        print(f"      {sentiment}: {count} ({pct:.1f}%)")
else:
    reviews_df['sentiment_score'] = np.nan
    reviews_df['sentiment_subjectivity'] = np.nan
    reviews_df['sentiment_category'] = 'Unknown'
    print(f"   ⚠️  Sentiment analysis skipped (TextBlob not installed)")

print()


# STEP 18: SAVE CLEANED DATA

print("💾 STEP 18: Saving cleaned data...")

# Save restaurants
restaurants_final.to_csv(OUTPUT_RESTAURANTS, index=False)
print(f"   ✓ Saved: {OUTPUT_RESTAURANTS}")
print(f"      ({len(restaurants_final)} restaurants × {len(restaurants_columns)} columns)")

# Save reviews
reviews_df.to_csv(OUTPUT_REVIEWS, index=False)
print(f"   ✓ Saved: {OUTPUT_REVIEWS}")
print(f"      ({len(reviews_df)} reviews × {len(reviews_df.columns)} columns)")

print()


# FINAL SUMMARY

print("="*80)
print("✅ CLEANING COMPLETE!")
print("="*80)
print(f"""
📦 OUTPUT FILES:
   • {OUTPUT_RESTAURANTS}
   • {OUTPUT_REVIEWS}

📊 SUMMARY:
   • Restaurants: {len(restaurants_final)}
   • Reviews: {len(reviews_df)}
   • Average rating: {restaurants_final['rating_overall'].mean():.2f}/5.0
   • Total review count: {restaurants_final['review_count_total'].sum():,}
   • Data completeness: {((restaurants_final['rating_overall'].notna().sum() + restaurants_final['phone'].notna().sum()) / (2 * len(restaurants_final)) * 100):.0f}%

✨ NEXT STEPS:
   1. Review the output files
   2. Share with your team
   3. Merge with other sources
   4. Start analysis & visualization!
""")
print("="*80)