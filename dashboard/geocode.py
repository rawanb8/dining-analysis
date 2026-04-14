import pandas as pd
import requests
import time

# CONFIGURATION

INPUT_FILE = '../merged/master_restaurants.csv'
OUTPUT_FILE = '../merged/master_restaurants_geocoded.csv'

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
print("GEOCODING BEIRUT RESTAURANTS")
print("="*70)
print()

# Step 1: Load data
print("Step 1: Loading restaurant data...")
df = pd.read_csv(INPUT_FILE)
print(f"   Loaded {len(df)} restaurants")
print()

# Step 2: Add geocoding columns if they don't exist
print("Step 2: Preparing columns...")
if 'latitude' not in df.columns:
    df['latitude'] = None
if 'longitude' not in df.columns:
    df['longitude'] = None
if 'geocoded' not in df.columns:
    df['geocoded'] = False
print("   Columns ready")
print()

# Step 3: Check how many need geocoding
to_geocode = df[df['geocoded'] == False]
already_done = len(df) - len(to_geocode)

print("Step 3: Checking progress...")
print(f"   Restaurants to geocode: {len(to_geocode)}")
print(f"   Already geocoded: {already_done}")
print()

if len(to_geocode) == 0:
    print("All restaurants already geocoded!")
    print()
else:
    # Step 4: Start geocoding
    print("Step 4: Starting geocoding process...")
    estimated_time = (len(to_geocode) * REQUEST_DELAY) / 60
    print(f"   Estimated time: ~{estimated_time:.1f} minutes")
    print(f"   Rate limit: {REQUEST_DELAY} seconds per request")
    print()
    
    success_count = 0
    fallback_count = 0
    failed_count = 0
    
    # Loop through restaurants
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
            df.at[idx, 'latitude'] = lat
            df.at[idx, 'longitude'] = lon
            df.at[idx, 'geocoded'] = True
            
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
    print(f"Successfully geocoded: {success_count}")
    print(f"Using area fallback: {fallback_count}")
    print(f"Failed to geocode: {failed_count}")
    print(f"Total geocoded: {success_count + fallback_count} / {len(to_geocode)}")
    
    if len(to_geocode) > 0:
        success_rate = ((success_count + fallback_count) / len(to_geocode)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    print()

# Step 5: Save results
print("Step 5: Saving geocoded data...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"   Saved to: {OUTPUT_FILE}")
print()

# Step 6: Show statistics
print("="*70)
print("FINAL STATISTICS")
print("="*70)

total_geocoded = df['geocoded'].sum()
total_restaurants = len(df)
geocoded_pct = (total_geocoded / total_restaurants) * 100

print(f"Total restaurants: {total_restaurants}")
print(f"Geocoded: {total_geocoded} ({geocoded_pct:.1f}%)")
print(f"Not geocoded: {total_restaurants - total_geocoded}")
print()

# Show by area
print("Geocoding by Area (Top 10):")
print()

area_stats = df.groupby('area').agg({
    'geocoded': ['count', 'sum']
})
area_stats.columns = ['Total', 'Geocoded']
area_stats['Success Rate'] = (area_stats['Geocoded'] / area_stats['Total'] * 100).round(1)
area_stats = area_stats.sort_values('Total', ascending=False).head(10)

print(area_stats)
print()

# Show coordinate ranges
geocoded_df = df[df['geocoded'] == True]
if len(geocoded_df) > 0:
    print("Coordinate Ranges:")
    print(f"   Latitude:  {geocoded_df['latitude'].min():.4f} to {geocoded_df['latitude'].max():.4f}")
    print(f"   Longitude: {geocoded_df['longitude'].min():.4f} to {geocoded_df['longitude'].max():.4f}")
    print()
    print("Center Point:")
    print(f"   Latitude:  {geocoded_df['latitude'].mean():.4f}")
    print(f"   Longitude: {geocoded_df['longitude'].mean():.4f}")
    print()

print("="*70)
print("GEOCODING COMPLETE!")
print("="*70)
print()
print("Next step: Run the Streamlit dashboard to see maps!")
print("   streamlit run streamlit_dashboard.py")
print("="*70)