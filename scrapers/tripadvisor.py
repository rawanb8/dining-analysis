import requests
import time
import csv

API_KEY = "4B459D77904343F08A4CF45AD616EA38"
BASE_URL = "https://api.content.tripadvisor.com/api/v1"


def api_get(path, params, max_retries=4):
    url = f"{BASE_URL}{path}"
    wait_times = [30, 60, 120, 240]

    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=20)

        if r.status_code == 429:
            wait = wait_times[min(attempt, len(wait_times)-1)]
            print(f"429 for {url} -> waiting {wait}s")
            time.sleep(wait)
            continue

        r.raise_for_status()
        return r.json()

    raise Exception(f"Too many 429 responses for {url}")


def city_to_coords(city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json"}

    res = requests.get(url, params=params, headers={"User-Agent": "my-app"})
    res.raise_for_status()
    data = res.json()

    if not data:
        raise Exception(f"City not found: {city}")

    return float(data[0]["lat"]), float(data[0]["lon"])


def nearby_restaurants(lat, lon, language="en", radius=None, radius_unit="km"):
    params = {
        "key": API_KEY,
        "latLong": f"{lat},{lon}",
        "category": "restaurants",
        "language": language
    }

    if radius is not None:
        params["radius"] = radius
        params["radiusUnit"] = radius_unit

    return api_get("/location/nearby_search", params)


def get_details(location_id, language="en"):
    params = {
        "key": API_KEY,
        "language": language
    }
    return api_get(f"/location/{location_id}/details", params)


def get_reviews(location_id, language="en"):
    params = {
        "key": API_KEY,
        "language": language
    }
    return api_get(f"/location/{location_id}/reviews", params)


def generate_grid(lat, lon, steps=1, offset=0.005):
    points = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            points.append((lat + i * offset, lon + j * offset))
    return points


cities = [
    "Beirut Lebanon","Saida Lebanon"
]

restaurants = {}

for city in cities:
    try:
        print(f"\nProcessing {city}...")
        lat, lon = city_to_coords(city)

        grid_points = generate_grid(lat, lon, steps=1, offset=0.005)

        for idx, (g_lat, g_lon) in enumerate(grid_points, start=1):
            try:
                print(f"  Request {idx}/{len(grid_points)} for {city} at ({g_lat}, {g_lon})")
                result = nearby_restaurants(g_lat, g_lon, radius=2, radius_unit="km")

                for item in result.get("data", []):
                    loc_id = item.get("location_id")

                    if loc_id and loc_id not in restaurants:
                        restaurants[loc_id] = {
                            "city": city,
                            "name": item.get("name"),
                            "address": item.get("address_obj", {}).get("address_string")
                        }

                time.sleep(2)

            except Exception as e:
                print(f"Failed at {g_lat}, {g_lon} in {city}: {e}")

    except Exception as e:
        print(f"Error in {city}: {e}")

print(f"\nTotal unique restaurants found: {len(restaurants)}")

all_data = []
for i, (location_id, basic) in enumerate(restaurants.items(), start=1):
    try:
        print(f"Fetching details/reviews {i}/{len(restaurants)} for {basic['name']}")
        details = get_details(location_id)
        reviews = get_reviews(location_id)

        all_data.append({
            "city": basic["city"],
            "location_id": location_id,
            "name": details.get("name"),
            "rating": details.get("rating"),
            "num_reviews": details.get("num_reviews"),
            "address": details.get("address_obj", {}).get("address_string"),
            "website": details.get("website"),
            "tripadvisor_url": details.get("web_url"),
            "reviews": reviews.get("data", [])
        })

        time.sleep(5)

    except Exception as e:
        print(f"Failed for {location_id}: {e}")

print(f"Collected details for {len(all_data)} restaurants")

with open("restaurants_with_reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "city",
        "location_id",
        "address",
        "restaurant_name",
        "website",
        "tripadvisor_url",
        "rating",
        "num_reviews",
         "review_title",
         "review_text",
         "review_rating"
    ])

    for r in all_data:
         if r["reviews"]:
             for review in r["reviews"]:
                 writer.writerow([
                     r["city"],
                     r["location_id"],
                     r["address"],
                     r["name"],
                     r["website"],
                     r["tripadvisor_url"],
                     r["rating"],
                     r["num_reviews"],
                     review.get("title"),
                     review.get("text"),
                     review.get("rating")
                 ])
         else:
             writer.writerow([
                 r["city"],
                 r["location_id"],
                 r["address"],
                 r["name"],
                 r["website"],
                 r["tripadvisor_url"],
                 r["rating"],
                 r["num_reviews"],
                 None,
                 None,
                 None
             ])

print("Saved to restaurants_with_reviews.csv")