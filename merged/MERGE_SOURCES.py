import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import os

# CONFIGURATION

SOURCE1_RESTAURANTS = '../cleaned/Wandorlog_restaurants_clean.csv'
SOURCE1_REVIEWS = '../cleaned/Wandorlog_reviews_clean.csv'

SOURCE2_RESTAURANTS = '../cleaned/Guru_restaurants_clean.csv'
SOURCE2_REVIEWS = '../cleaned/Guru_reviews_clean.csv'

SOURCE3_RESTAURANTS = '../cleaned/Tripadvisor_restaurants_clean.csv'
SOURCE3_REVIEWS = '../cleaned/Tripadvisor_reviews_clean.csv'

OUTPUT_DIR = '../merged'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMILARITY_THRESHOLD = 0.85  # 85% match = duplicate

# HELPER FUNCTIONS

def normalize_text(text):
    """Normalize text for comparison: lowercase, no special chars"""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return ' '.join(text.split())

def similarity_score(text1, text2):
    """Calculate similarity between two text strings (0-1)"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 or not norm2:
        return 0.0
    return SequenceMatcher(None, norm1, norm2).ratio()

def find_duplicates(df1, df2, threshold=0.85):
    """Find duplicate restaurants between two dataframes"""
    duplicates = []
    
    for idx1, row1 in df1.iterrows():
        for idx2, row2 in df2.iterrows():
            # Compare name and area
            name_sim = similarity_score(row1['name'], row2['name'])
            area_sim = similarity_score(row1['area'], row2['area'])
            
            # Average similarity
            avg_sim = (name_sim + area_sim) / 2
            
            if avg_sim >= threshold:
                duplicates.append({
                    'id1': row1['restaurant_id'],
                    'id2': row2['restaurant_id'],
                    'name1': row1['name'],
                    'name2': row2['name'],
                    'similarity': round(avg_sim, 3)
                })
    
    return duplicates

def merge_ratings(row):
    """Calculate weighted average rating from multiple sources"""
    ratings = []
    weights = []
    
    if pd.notna(row['rating_google']):
        ratings.append(row['rating_google'])
        weights.append(row['review_count_google'] or 1)
    
    if pd.notna(row['rating_tripadvisor']):
        ratings.append(row['rating_tripadvisor'])
        weights.append(row['review_count_tripadvisor'] or 1)
    
    if not ratings:
        return np.nan
    
    return round(np.average(ratings, weights=weights), 2)

def merge_features(row, feature_col):
    """Merge feature values: prioritize TRUE over Unknown"""
    # Collect all values for this feature from merged sources
    values = str(row.get(feature_col, '')).split('|')
    
    if 'TRUE' in values:
        return 'TRUE'
    elif 'FALSE' in values:
        return 'FALSE'
    else:
        return 'Unknown'

# LOAD DATA

print("="*70)
print("🔀 MERGING RESTAURANT DATA FROM 3 SOURCES")
print("="*70)
print()

print("📂 Loading cleaned data...")

src1_rest = pd.read_csv(SOURCE1_RESTAURANTS)
src1_rev = pd.read_csv(SOURCE1_REVIEWS)

src2_rest = pd.read_csv(SOURCE2_RESTAURANTS)
src2_rev = pd.read_csv(SOURCE2_REVIEWS)

src3_rest = pd.read_csv(SOURCE3_RESTAURANTS)
src3_rev = pd.read_csv(SOURCE3_REVIEWS)

print(f"   Source 1: {len(src1_rest)} restaurants, {len(src1_rev)} reviews")
print(f"   Source 2: {len(src2_rest)} restaurants, {len(src2_rev)} reviews")
print(f"   Source 3: {len(src3_rest)} restaurants, {len(src3_rev)} reviews")
print()

# FIND DUPLICATES

print("🔍 Detecting duplicate restaurants...")

# Find duplicates between sources
dups_1_2 = find_duplicates(src1_rest, src2_rest, SIMILARITY_THRESHOLD)
dups_1_3 = find_duplicates(src1_rest, src3_rest, SIMILARITY_THRESHOLD)
dups_2_3 = find_duplicates(src2_rest, src3_rest, SIMILARITY_THRESHOLD)

total_dups = len(dups_1_2) + len(dups_1_3) + len(dups_2_3)

print(f"   Source 1 ↔ Source 2: {len(dups_1_2)} duplicates")
print(f"   Source 1 ↔ Source 3: {len(dups_1_3)} duplicates")
print(f"   Source 2 ↔ Source 3: {len(dups_2_3)} duplicates")
print(f"   Total duplicates: {total_dups}")
print()

# MERGE DUPLICATES

print("🔗 Merging duplicate records...")

# Create duplicate mapping
dup_map = {}  # Maps restaurant_id -> list of duplicate IDs

for dup in dups_1_2 + dups_1_3 + dups_2_3:
    id1, id2 = dup['id1'], dup['id2']
    
    # Group all duplicates together
    if id1 not in dup_map and id2 not in dup_map:
        dup_map[id1] = [id1, id2]
    elif id1 in dup_map:
        if id2 not in dup_map[id1]:
            dup_map[id1].append(id2)
    elif id2 in dup_map:
        if id1 not in dup_map[id2]:
            dup_map[id2].append(id1)

merged_restaurants = []
processed_ids = set()

# Process duplicates
for primary_id, duplicate_ids in dup_map.items():
    if primary_id in processed_ids:
        continue
    
    # Get all duplicate records
    dup_records = []
    for dup_id in duplicate_ids:
        if dup_id.startswith('src1_'):
            dup_records.append(src1_rest[src1_rest['restaurant_id'] == dup_id].iloc[0])
        elif dup_id.startswith('src2_'):
            dup_records.append(src2_rest[src2_rest['restaurant_id'] == dup_id].iloc[0])
        elif dup_id.startswith('src3_'):
            dup_records.append(src3_rest[src3_rest['restaurant_id'] == dup_id].iloc[0])
    
    # Merge into single record
    merged_record = dup_records[0].copy()
    merged_record['restaurant_id'] = f"merged_{'_'.join(duplicate_ids)}"
    
    # Merge ratings
    merged_record['rating_overall'] = merge_ratings(pd.concat(dup_records, axis=1).T.iloc[0])
    merged_record['review_count_total'] = sum([r.get('review_count_total', 0) for r in dup_records])
    merged_record['review_count_google'] = sum([r.get('review_count_google', 0) for r in dup_records])
    merged_record['review_count_tripadvisor'] = sum([r.get('review_count_tripadvisor', 0) for r in dup_records])
    
    # Merge star counts
    for i in range(1, 6):
        merged_record[f'star_{i}_count'] = sum([r.get(f'star_{i}_count', 0) for r in dup_records])
    
    # Recalculate star percentages
    total_stars = sum([merged_record[f'star_{i}_count'] for i in range(1, 6)])
    if total_stars > 0:
        for i in range(1, 6):
            merged_record[f'star_{i}_percent'] = round((merged_record[f'star_{i}_count'] / total_stars) * 100, 1)
    
    # Merge features (TRUE > FALSE > Unknown)
    feature_cols = ['delivery_available', 'outdoor_seating', 'reservation_required',
                    'cash_only', 'credit_cards_accepted', 'wifi_available', 
                    'wheelchair_accessible', 'takeaway_available', 'parking_available',
                    'live_music', 'pet_friendly', 'kids_friendly']
    
    for feature in feature_cols:
        values = [str(r.get(feature, 'Unknown')) for r in dup_records]
        if 'TRUE' in values:
            merged_record[feature] = 'TRUE'
        elif 'FALSE' in values:
            merged_record[feature] = 'FALSE'
        else:
            merged_record[feature] = 'Unknown'
    
    # Merge URLs (combine)
    urls = [r.get('source_url', '') for r in dup_records if pd.notna(r.get('source_url', ''))]
    merged_record['source_url'] = ' | '.join(urls) if urls else ''
    
    merged_restaurants.append(merged_record)
    processed_ids.update(duplicate_ids)

print(f"   ✓ Merged {len(merged_restaurants)} duplicate groups")
print()

# COMBINE ALL RESTAURANTS

print("📊 Combining all restaurants...")

# Add non-duplicate restaurants
for df in [src1_rest, src2_rest, src3_rest]:
    for idx, row in df.iterrows():
        if row['restaurant_id'] not in processed_ids:
            merged_restaurants.append(row)

master_restaurants = pd.DataFrame(merged_restaurants)

print(f"   ✓ Total unique restaurants: {len(master_restaurants)}")
print(f"      - Merged: {len(merged_restaurants) - (len(src1_rest) + len(src2_rest) + len(src3_rest) - len(processed_ids))}")
print(f"      - Unique: {len(master_restaurants) - len(dup_map)}")
print()

# COMBINE ALL REVIEWS

print("💬 Combining all reviews...")

# Update review restaurant_id for merged restaurants
review_id_mapping = {}
for merged_id, dup_ids in dup_map.items():
    new_id = f"merged_{'_'.join(dup_ids)}"
    for dup_id in dup_ids:
        review_id_mapping[dup_id] = new_id

# Combine all reviews
all_reviews = pd.concat([src1_rev, src2_rev, src3_rev], ignore_index=True)

# Update restaurant IDs for merged records
all_reviews['restaurant_id'] = all_reviews['restaurant_id'].apply(
    lambda x: review_id_mapping.get(x, x)
)

# Create unique review IDs
all_reviews['review_id'] = [f'rev_{i+1:06d}' for i in range(len(all_reviews))]

print(f"   ✓ Total reviews: {len(all_reviews)}")
print()

# DATA QUALITY SUMMARY

print("📈 Data Quality Summary...")

# Overall statistics
total_restaurants = len(master_restaurants)
total_reviews = len(all_reviews)
avg_rating = master_restaurants['rating_overall'].mean()
avg_reviews_per_restaurant = total_reviews / total_restaurants

# Data completeness by field
completeness = {
    'phone': (master_restaurants['phone'].notna().sum() / total_restaurants) * 100,
    'cuisine': ((master_restaurants['cuisine_primary'] != 'Unknown').sum() / total_restaurants) * 100,
    'hours': ((master_restaurants['hours_monday'] != 'Unknown').sum() / total_restaurants) * 100,
    'price': ((master_restaurants['price_category'] != 'Unknown').sum() / total_restaurants) * 100,
}

print(f"   Overall rating: {avg_rating:.2f}/5.0")
print(f"   Reviews per restaurant: {avg_reviews_per_restaurant:.1f}")
print(f"   Data completeness:")
print(f"      - Phone: {completeness['phone']:.0f}%")
print(f"      - Cuisine: {completeness['cuisine']:.0f}%")
print(f"      - Hours: {completeness['hours']:.0f}%")
print(f"      - Price: {completeness['price']:.0f}%")
print()

# Feature availability
feature_cols = ['delivery_available', 'outdoor_seating', 'reservation_required',
                'cash_only', 'credit_cards_accepted', 'wifi_available', 
                'wheelchair_accessible', 'takeaway_available', 'parking_available',
                'live_music', 'pet_friendly', 'kids_friendly']

print(f"   Feature detection:")
for feature in feature_cols:
    count = (master_restaurants[feature] == 'TRUE').sum()
    pct = (count / total_restaurants) * 100
    if count > 0:
        print(f"      - {feature.replace('_', ' ').title()}: {count} ({pct:.0f}%)")

print()

# SAVE MERGED DATA

print("💾 Saving merged data...")

master_restaurants.to_csv(f'{OUTPUT_DIR}/master_restaurants.csv', index=False)
print(f"   ✓ Saved: {OUTPUT_DIR}/master_restaurants.csv")
print(f"      ({len(master_restaurants)} restaurants × {len(master_restaurants.columns)} columns)")

all_reviews.to_csv(f'{OUTPUT_DIR}/master_reviews.csv', index=False)
print(f"   ✓ Saved: {OUTPUT_DIR}/master_reviews.csv")
print(f"      ({len(all_reviews)} reviews × {len(all_reviews.columns)} columns)")

# Save duplicate report
if dups_1_2 or dups_1_3 or dups_2_3:
    dup_report = pd.DataFrame(dups_1_2 + dups_1_3 + dups_2_3)
    dup_report.to_csv(f'{OUTPUT_DIR}/duplicate_report.csv', index=False)
    print(f"   ✓ Saved: {OUTPUT_DIR}/duplicate_report.csv")
    print(f"      ({len(dup_report)} duplicates detected)")

print()

# FINAL SUMMARY

print("="*70)
print("✅ MERGE COMPLETE!")
print("="*70)
print(f"""
📦 OUTPUT FILES:
   • {OUTPUT_DIR}/master_restaurants.csv
   • {OUTPUT_DIR}/master_reviews.csv
   • {OUTPUT_DIR}/duplicate_report.csv

📊 SUMMARY:
   • Total unique restaurants: {len(master_restaurants)}
   • Total reviews: {len(all_reviews)}
   • Duplicates merged: {total_dups}
   • Average rating: {avg_rating:.2f}/5.0
   • Reviews per restaurant: {avg_reviews_per_restaurant:.1f}
   
🎯 TOP CUISINES:
""")

# Top 5 cuisines
cuisine_counts = master_restaurants[master_restaurants['cuisine_primary'] != 'Unknown']['cuisine_primary'].value_counts().head(5)
for cuisine, count in cuisine_counts.items():
    pct = (count / len(master_restaurants)) * 100
    print(f"   {cuisine}: {count} ({pct:.1f}%)")

print(f"""
🏆 TOP RATED (5 stars):
""")

# Top 5 rated restaurants
top_rated = master_restaurants.nlargest(5, 'rating_overall')[['name', 'rating_overall', 'review_count_total', 'area']]
for idx, row in top_rated.iterrows():
    print(f"   {row['name']} - {row['rating_overall']}/5.0 ({row['review_count_total']} reviews) - {row['area']}")

print()
print("="*70)