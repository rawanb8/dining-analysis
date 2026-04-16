import pandas as pd
import requests
import time
import os

# CONFIGURATION
INPUT_FILE = '../merged/master_restaurants.csv'
OUTPUT_FILE = '../merged/master_restaurants_geocoded.csv'
GEOCODED_BACKUP = '../merged/geocoding_backup.csv'  # Backup of previous geocoding

# Rate limit: 1 request per second (respect Nominatim usage policy)
REQUEST_DELAY = 1.1  # seconds between requests

# GEOCODING FUNCTION
def geocode_address(address, city, country):
    """
    Geocode an address using Nominatim (OpenStreetMap) API.
    
    Parameters:
        address (str): Full address or area name
        city (str): City name
        country (str): Country name
    
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    try:
        # Build search query
        query = f"{address}, {city}, {country}"
        
        # Nominatim API endpoint
        url = "https://nominatim.openstreetmap.org/search"
        
        # Request parameters
        params = {
            'q': query,
            'format': 'json',
            'limit': 1
        }
        
        # Headers (required by Nominatim)
        headers = {
            'User-Agent': 'BeirutRestaurantAnalysis/1.0'
        }
        
        # Make API request
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Check if we got results
            if data and len(data) > 0:
                latitude = float(data[0]['lat'])
                longitude = float(data[0]['lon'])
                return latitude, longitude
        
        # Return None if not found
        return None, None
        
    except Exception as e:
        print(f"   Error: {str(e)}")
        return None, None

# MAIN PROGRAM

print("="*70)
print("INCREMENTAL GEOCODING - ONLY NEW RESTAURANTS")
print("="*70)
print()

# Step 1: Load current restaurant data
print("Step 1: Loading current restaurant data...")
df_current = pd.read_csv(INPUT_FILE)
print(f"   Loaded {len(df_current)} restaurants from current merge")
print()

# Step 2: Load previous geocoding data if it exists
print("Step 2: Checking for previous geocoding work...")
if os.path.exists(OUTPUT_FILE):
    df_previous = pd.read_csv(OUTPUT_FILE)
    print(f"   ✓ Found previous geocoding file: {len(df_previous)} restaurants")
    
    # Create lookup dictionary: restaurant_id -> (lat, lon, geocoded)
    geocoding_lookup = {}
    for _, row in df_previous.iterrows():
        if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
            geocoding_lookup[row['restaurant_id']] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'geocoded': True
            }
    
    print(f"   ✓ Loaded {len(geocoding_lookup)} previously geocoded locations")
else:
    print("   ⚠️ No previous geocoding file found - will geocode all restaurants")
    geocoding_lookup = {}
print()

# Step 3: Add geocoding columns to current data
print("Step 3: Preparing columns...")
if 'latitude' not in df_current.columns:
    df_current['latitude'] = None
if 'longitude' not in df_current.columns:
    df_current['longitude'] = None
if 'geocoded' not in df_current.columns:
    df_current['geocoded'] = False
print("   Columns ready")
print()

# Step 4: Apply previous geocoding data to matching restaurants
print("Step 4: Applying previous geocoding data...")
matched_count = 0
for idx, row in df_current.iterrows():
    restaurant_id = row['restaurant_id']
    
    if restaurant_id in geocoding_lookup:
        # Found previous geocoding - reuse it!
        df_current.at[idx, 'latitude'] = geocoding_lookup[restaurant_id]['latitude']
        df_current.at[idx, 'longitude'] = geocoding_lookup[restaurant_id]['longitude']
        df_current.at[idx, 'geocoded'] = True
        matched_count += 1

print(f"   ✓ Reused geocoding for {matched_count} restaurants")
print()

# Step 5: Identify new restaurants that need geocoding
to_geocode = df_current[df_current['geocoded'] == False]
already_done = len(df_current) - len(to_geocode)

print("Step 5: Checking what needs geocoding...")
print(f"   Already geocoded: {already_done}")
print(f"   New restaurants to geocode: {len(to_geocode)}")
print()

if len(to_geocode) == 0:
    print("✅ All restaurants already geocoded! Nothing to do.")
    print()
    print("Saving updated file...")
    df_current.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved to: {OUTPUT_FILE}")
    print()
    exit(0)

# Step 6: Geocode new restaurants only
print("Step 6: Geocoding NEW restaurants only...")
estimated_time = (len(to_geocode) * REQUEST_DELAY) / 60
print(f"   Estimated time: ~{estimated_time:.1f} minutes")
print(f"   Rate limit: {REQUEST_DELAY} seconds per request")
print()

success_count = 0
fallback_count = 0
failed_count = 0

# Loop through new restaurants only
for idx, row in to_geocode.iterrows():
    restaurant_name = row['name']
    address = row['address_full']
    area = row['area']
    city = row['city']
    country = row['country']
    
    # Handle missing city/country
    if pd.isna(city) or city == 'Unknown':
        city = 'Beirut'
    if pd.isna(country) or country == 'Unknown':
        country = 'Lebanon'
    
    # Progress counter
    current = success_count + fallback_count + failed_count + 1
    total = len(to_geocode)
    
    # Show progress
    print(f"[{current}/{total}] {restaurant_name[:40]}...", end='')
    
    # Try geocoding with full address first
    lat = None
    lon = None
    
    if pd.notna(address) and address != 'Unknown' and len(str(address)) > 5:
        lat, lon = geocode_address(address, city, country)
        time.sleep(REQUEST_DELAY)
    
    # Fallback: Try just the area/neighborhood
    if lat is None and pd.notna(area) and area != 'Unknown':
        print(" (trying area)...", end='')
        lat, lon = geocode_address(area, city, country)
        time.sleep(REQUEST_DELAY)
        
        if lat is not None:
            fallback_count += 1
    
    # Save results
    if lat is not None and lon is not None:
        df_current.at[idx, 'latitude'] = lat
        df_current.at[idx, 'longitude'] = lon
        df_current.at[idx, 'geocoded'] = True
        
        if fallback_count == 0 or current == 1:
            success_count += 1
        
        print(f" SUCCESS ({lat:.4f}, {lon:.4f})")
    else:
        failed_count += 1
        print(" FAILED")

print()
print("="*70)
print("GEOCODING SUMMARY")
print("="*70)
print(f"Previously geocoded (reused): {already_done}")
print(f"Newly geocoded: {success_count}")
print(f"Using area fallback: {fallback_count}")
print(f"Failed to geocode: {failed_count}")
print(f"Total geocoded: {already_done + success_count + fallback_count} / {len(df_current)}")

if len(df_current) > 0:
    success_rate = ((already_done + success_count + fallback_count) / len(df_current)) * 100
    print(f"Overall success rate: {success_rate:.1f}%")
print()

# Step 7: Save results
print("Step 7: Saving geocoded data...")
df_current.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Saved to: {OUTPUT_FILE}")

# Also save backup for next run
df_current.to_csv(GEOCODED_BACKUP, index=False)
print(f"   ✓ Backup saved to: {GEOCODED_BACKUP}")
print()

# Step 8: Show statistics
print("="*70)
print("FINAL STATISTICS")
print("="*70)

total_geocoded = df_current['geocoded'].sum()
total_restaurants = len(df_current)
geocoded_pct = (total_geocoded / total_restaurants) * 100

print(f"Total restaurants: {total_restaurants}")
print(f"Geocoded: {total_geocoded} ({geocoded_pct:.1f}%)")
print(f"Not geocoded: {total_restaurants - total_geocoded}")
print()

# Show coordinate ranges
geocoded_df = df_current[df_current['geocoded'] == True]
if len(geocoded_df) > 0:
    print("Coordinate Ranges:")
    print(f"   Latitude:  {geocoded_df['latitude'].min():.4f} to {geocoded_df['latitude'].max():.4f}")
    print(f"   Longitude: {geocoded_df['longitude'].min():.4f} to {geocoded_df['longitude'].max():.4f}")
    print()

print("="*70)
print("GEOCODING COMPLETE!")
print("="*70)
print()
print("Next step: Run the Streamlit dashboard to see maps!")
print("   streamlit run streamlit_dashboard.py")
print("="*70)