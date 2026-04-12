import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

try:
    from textblob import TextBlob
    test_blob = TextBlob("test")
    _ = test_blob.sentiment
    SENTIMENT_AVAILABLE = True
except:
    SENTIMENT_AVAILABLE = False

INPUT_FILE = '../data/guru.csv'
OUTPUT_RESTAURANTS = '../cleaned/Guru_restaurants_clean.csv'
OUTPUT_REVIEWS = '../cleaned/Guru_reviews_clean.csv'

os.makedirs('../cleaned', exist_ok=True)

print("="*70)
print("🧹 SOURCE 2: RESTAURANT GURU DATA CLEANING")
print("="*70)
print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_RESTAURANTS}")
print(f"        {OUTPUT_REVIEWS}\n")

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df)} rows\n")

# Assign IDs to ALL rows first (before filtering)
df['restaurant_id'] = ['src2_' + str(i+1).zfill(3) for i in range(len(df))]

# Extract reviews from ALL rows (including ones we'll drop)
print("📝 Extracting reviews from ALL rows (including incomplete restaurants)...")
all_reviews_list = []

for idx, row in df.iterrows():
    restaurant_id = row['restaurant_id']
    restaurant_name = row['name'] if pd.notna(row['name']) else 'Unknown Restaurant'
    reviews_text = row['all_reviews']
    
    if pd.isna(reviews_text):
        continue
    
    individual_reviews = str(reviews_text).split('||')
    
    for review in individual_reviews:
        review = review.strip()
        if not review:
            continue
        
        rating_match = re.search(r'\[(\d+\.?\d*)\]', review)
        date_source_match = re.search(r'\[(.*?)\s+on\s+(.*?)\]:', review)
        
        rating = np.nan
        review_date = np.nan
        review_source = 'Unknown'
        
        if rating_match:
            rating = float(rating_match.group(1))
        
        if date_source_match:
            date_str = date_source_match.group(1).strip()
            review_source = date_source_match.group(2).strip()
        
        review_text_clean = re.sub(r'\[\d+\.?\d*\]\s*\[.*?\]:\s*', '', review).strip()
        
        all_reviews_list.append({
            'review_id': f'rev_{len(all_reviews_list)+1:05d}',
            'restaurant_id': restaurant_id,
            'restaurant_name': restaurant_name,
            'review_text': review,
            'review_text_cleaned': review_text_clean,
            'rating': rating,
            'review_date': review_date,
            'review_source': review_source,
            'word_count': len(review_text_clean.split()),
            'from_incomplete_restaurant': False  # Will mark these
        })

print(f"✓ Extracted {len(all_reviews_list)} reviews from ALL restaurants\n")

# NOW filter restaurants for the restaurants table
df_before_filter = df.copy()
df = df[df['name'].notna()].copy()
df = df[df['location'].notna()].copy()
print(f"✓ Filtered to {len(df)} valid restaurants (from {len(df_before_filter)})\n")

# Mark which reviews came from incomplete restaurants
valid_restaurant_ids = set(df['restaurant_id'].unique())
for review in all_reviews_list:
    if review['restaurant_id'] not in valid_restaurant_ids:
        review['from_incomplete_restaurant'] = True

df = df.reset_index(drop=True)

df['name'] = df['name'].str.strip()

def parse_cuisine(cuisine):
    if pd.isna(cuisine) or str(cuisine).strip() == '':
        return 'Unknown', 'Unknown'
    cuisines = [c.strip() for c in str(cuisine).split(',')]
    cuisines = [c for c in cuisines if c and c.lower() != 'n/a']
    if not cuisines:
        return 'Unknown', 'Unknown'
    return cuisines[0], ','.join(cuisines)

df[['cuisine_primary', 'cuisine_tags']] = df['cuisines'].apply(lambda x: pd.Series(parse_cuisine(x)))

def parse_address(location):
    if pd.isna(location):
        return {'address_full': '', 'area': 'Unknown', 'city': 'Beirut', 'country': 'Lebanon'}
    parts = [p.strip() for p in str(location).split(',')]
    if len(parts) >= 3:
        return {'address_full': location, 'area': parts[-3], 'city': parts[-2], 'country': parts[-1]}
    elif len(parts) == 2:
        return {'address_full': location, 'area': parts[0], 'city': parts[1], 'country': 'Lebanon'}
    else:
        return {'address_full': location, 'area': 'Unknown', 'city': 'Beirut', 'country': 'Lebanon'}

address_df = df['location'].apply(parse_address).apply(pd.Series)
df = pd.concat([df, address_df], axis=1)

def clean_phone(phone):
    if pd.isna(phone):
        return np.nan
    phone = re.sub(r'[^\d+]', '', str(phone))
    if phone.startswith('+961'):
        return phone
    if phone.startswith('961'):
        return '+' + phone
    if len(phone) >= 7:
        return '+961' + phone.lstrip('0')
    return np.nan

df['phone'] = df['contact_info'].apply(clean_phone)

def parse_rating(rating):
    if pd.isna(rating):
        return np.nan
    try:
        r = float(rating)
        if r > 5:
            r = r / 2
        return round(r, 1)
    except:
        return np.nan

df['rating_overall'] = df['rating'].apply(parse_rating)
df['rating_google'] = df['rating_overall']
df['rating_tripadvisor'] = np.nan

# Initialize review counts (will calculate from actual reviews)
df['review_count_total'] = 0
df['review_count_google'] = 0
df['review_count_tripadvisor'] = 0

# Initialize star counts (will calculate from reviews)
df['star_5_count'] = 0
df['star_4_count'] = 0
df['star_3_count'] = 0
df['star_2_count'] = 0
df['star_1_count'] = 0
df['star_5_percent'] = 0.0
df['star_4_percent'] = 0.0
df['star_3_percent'] = 0.0
df['star_2_percent'] = 0.0
df['star_1_percent'] = 0.0

def parse_working_hours(hours_text):
    """Parse working hours from format: 'Sunday: 8AM-12AM | Monday: 8AM-1AM | ...'"""
    hours_dict = {
        'hours_monday': 'Unknown',
        'hours_tuesday': 'Unknown',
        'hours_wednesday': 'Unknown',
        'hours_thursday': 'Unknown',
        'hours_friday': 'Unknown',
        'hours_saturday': 'Unknown',
        'hours_sunday': 'Unknown'
    }
    
    if pd.isna(hours_text) or str(hours_text).strip() in ['', 'N/A', 'None']:
        return hours_dict
    
    try:
        day_mapping = {
            'monday': 'hours_monday',
            'tuesday': 'hours_tuesday',
            'wednesday': 'hours_wednesday',
            'thursday': 'hours_thursday',
            'friday': 'hours_friday',
            'saturday': 'hours_saturday',
            'sunday': 'hours_sunday'
        }
        
        parts = str(hours_text).split('|')
        
        for part in parts:
            part = part.strip()
            if ':' in part:
                day_part, time_part = part.split(':', 1)
                day_name = day_part.strip().lower()
                time_value = time_part.strip()
                
                if day_name in day_mapping:
                    hours_dict[day_mapping[day_name]] = time_value
        
        return hours_dict
    except:
        return hours_dict

print("⏰ Parsing working hours...")
hours_df = df['working_hours'].apply(parse_working_hours).apply(pd.Series)
df = pd.concat([df, hours_df], axis=1)

def infer_price(price_range):
    if pd.isna(price_range):
        return 'Mid-Range'
    try:
        p = int(price_range)
        if p == 1:
            return 'Budget'
        elif p >= 3:
            return 'High-End'
        else:
            return 'Mid-Range'
    except:
        return 'Mid-Range'

df['price_category'] = df['price_range'].apply(infer_price)

def parse_features(features, all_reviews=''):
    """Extract features from features column and reviews text"""
    
    # Combine features and reviews for comprehensive extraction
    combined_text = str(features).lower() if pd.notna(features) else ''
    reviews_text = str(all_reviews).lower() if pd.notna(all_reviews) else ''
    full_text = combined_text + ' ' + reviews_text
    
    # Extract from features column
    delivery = 'TRUE' if 'delivery' in combined_text else 'Unknown'
    outdoor = 'TRUE' if 'outdoor' in combined_text else 'Unknown'
    reservation = 'TRUE' if any(x in combined_text for x in ['booking', 'reservation']) else 'Unknown'
    
    # New: Extract from features column
    cash_only = 'TRUE' if 'cash only' in combined_text else 'Unknown'
    credit_cards = 'TRUE' if any(x in combined_text for x in ['credit card', 'cards accepted']) else 'Unknown'
    wifi = 'TRUE' if any(x in combined_text for x in ['wifi', 'wi-fi', 'internet']) else 'Unknown'
    wheelchair = 'TRUE' if 'wheelchair' in combined_text else 'Unknown'
    takeaway = 'TRUE' if 'takeaway' in combined_text else 'Unknown'
    
    # Extract from reviews text (mentions = likely available)
    parking = 'TRUE' if any(x in full_text for x in ['parking', 'park available', 'free parking', 'valet']) else 'Unknown'
    live_music = 'TRUE' if any(x in full_text for x in ['live music', 'live band', 'dj', 'piano music']) else 'Unknown'
    pet_friendly = 'TRUE' if any(x in full_text for x in ['pet friendly', 'pet-friendly', 'dog friendly', 'dogs allowed']) else 'Unknown'
    kids_friendly = 'TRUE' if any(x in full_text for x in ['kids', 'children', 'family-friendly', 'playground']) else 'Unknown'
    
    return {
        'delivery_available': delivery,
        'outdoor_seating': outdoor,
        'reservation_required': reservation,
        'cash_only': cash_only,
        'credit_cards_accepted': credit_cards,
        'wifi_available': wifi,
        'wheelchair_accessible': wheelchair,
        'takeaway_available': takeaway,
        'parking_available': parking,
        'live_music': live_music,
        'pet_friendly': pet_friendly,
        'kids_friendly': kids_friendly
    }

# Apply feature extraction with both features and reviews
print("🔍 Extracting features from features column and reviews...")
features_df = df.apply(lambda row: pd.Series(parse_features(row['features'], row.get('all_reviews', ''))), axis=1)
df = pd.concat([df, features_df], axis=1)

df['data_source'] = 'source2'
df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')
df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
df['source_url'] = df['url']

df['menu_items'] = ''
df['menu_link'] = ''
df['website'] = ''
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
print(f"✓ Created restaurants table: {len(restaurants_final)} rows\n")

# Use the reviews we already extracted (before filtering)
reviews_df = pd.DataFrame(all_reviews_list)
print(f"✓ Using previously extracted {len(reviews_df)} reviews\n")

# Count reviews for valid restaurants only
print("📊 Calculating review counts and star distributions...")
review_counts = reviews_df[reviews_df['from_incomplete_restaurant'] == False].groupby('restaurant_id').size()

for rest_id in df['restaurant_id']:
    # Get reviews for this restaurant
    rest_reviews = reviews_df[(reviews_df['restaurant_id'] == rest_id) & 
                              (reviews_df['from_incomplete_restaurant'] == False)]
    
    # Total review count
    count = len(rest_reviews)
    df.loc[df['restaurant_id'] == rest_id, 'review_count_total'] = count
    
    if count > 0:
        # Calculate star counts from reviews
        valid_ratings = rest_reviews[rest_reviews['rating'].notna()]['rating']
        
        if len(valid_ratings) > 0:
            star_5 = ((valid_ratings >= 4.5) & (valid_ratings <= 5.0)).sum()
            star_4 = ((valid_ratings >= 3.5) & (valid_ratings < 4.5)).sum()
            star_3 = ((valid_ratings >= 2.5) & (valid_ratings < 3.5)).sum()
            star_2 = ((valid_ratings >= 1.5) & (valid_ratings < 2.5)).sum()
            star_1 = ((valid_ratings >= 0.0) & (valid_ratings < 1.5)).sum()
            
            total_rated = len(valid_ratings)
            
            df.loc[df['restaurant_id'] == rest_id, 'star_5_count'] = star_5
            df.loc[df['restaurant_id'] == rest_id, 'star_4_count'] = star_4
            df.loc[df['restaurant_id'] == rest_id, 'star_3_count'] = star_3
            df.loc[df['restaurant_id'] == rest_id, 'star_2_count'] = star_2
            df.loc[df['restaurant_id'] == rest_id, 'star_1_count'] = star_1
            
            if total_rated > 0:
                df.loc[df['restaurant_id'] == rest_id, 'star_5_percent'] = (star_5 / total_rated) * 100
                df.loc[df['restaurant_id'] == rest_id, 'star_4_percent'] = (star_4 / total_rated) * 100
                df.loc[df['restaurant_id'] == rest_id, 'star_3_percent'] = (star_3 / total_rated) * 100
                df.loc[df['restaurant_id'] == rest_id, 'star_2_percent'] = (star_2 / total_rated) * 100
                df.loc[df['restaurant_id'] == rest_id, 'star_1_percent'] = (star_1 / total_rated) * 100

restaurants_final['review_count_total'] = df['review_count_total'].astype(int)
restaurants_final['review_count_google'] = 0  # Not available in Source 2
restaurants_final['review_count_tripadvisor'] = 0  # Not available in Source 2

# Copy star counts to final dataframe
restaurants_final['star_5_count'] = df['star_5_count'].astype(int)
restaurants_final['star_4_count'] = df['star_4_count'].astype(int)
restaurants_final['star_3_count'] = df['star_3_count'].astype(int)
restaurants_final['star_2_count'] = df['star_2_count'].astype(int)
restaurants_final['star_1_count'] = df['star_1_count'].astype(int)
restaurants_final['star_5_percent'] = df['star_5_percent']
restaurants_final['star_4_percent'] = df['star_4_percent']
restaurants_final['star_3_percent'] = df['star_3_percent']
restaurants_final['star_2_percent'] = df['star_2_percent']
restaurants_final['star_1_percent'] = df['star_1_percent']

if SENTIMENT_AVAILABLE and len(reviews_df) > 0:
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
    print(f"✓ Sentiment analysis complete\n")
else:
    reviews_df['sentiment_score'] = np.nan
    reviews_df['sentiment_subjectivity'] = np.nan
    reviews_df['sentiment_category'] = 'Unknown'

restaurants_final.to_csv(OUTPUT_RESTAURANTS, index=False)
print(f"✓ Saved: {OUTPUT_RESTAURANTS}")
print(f"  ({len(restaurants_final)} restaurants)\n")

reviews_df.to_csv(OUTPUT_REVIEWS, index=False)
print(f"✓ Saved: {OUTPUT_REVIEWS}")
print(f"  ({len(reviews_df)} reviews)\n")

print("="*70)
print("✅ SOURCE 2 CLEANING COMPLETE!")
print("="*70)

reviews_from_incomplete = reviews_df['from_incomplete_restaurant'].sum()
reviews_from_complete = len(reviews_df) - reviews_from_incomplete

print(f"""
📊 SUMMARY:
  • Restaurants: {len(restaurants_final)}
  • Reviews (total): {len(reviews_df)}
    - From valid restaurants: {reviews_from_complete}
    - Saved from incomplete restaurants: {reviews_from_incomplete} 💎
  • Avg rating: {restaurants_final['rating_overall'].mean():.2f}/5.0
  • Data completeness: {((restaurants_final['rating_overall'].notna().sum() + restaurants_final['phone'].notna().sum()) / (2 * len(restaurants_final)) * 100):.0f}%

💡 Smart extraction: We saved {reviews_from_incomplete} reviews from {len(df_before_filter) - len(df)} 
   restaurants that had no location but had valuable reviews!
""")
print("="*70)