import os
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor

TXT_FILE = "scrapers/guru/links/all_links.txt"
PROGRESS_FILE = "scrapers/guru/processed_links.txt"
OUTPUT_FILE = "data/guru.csv"
BATCH_SIZE = 100

def create_driver():
    options = Options()
    
    options.page_load_strategy = 'eager'
    # options.add_argument("--headless=new")
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    
    # tryingn to avoid verify human
    options.add_argument("--disable-blink-features=AutomationControlled")
    uas = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ]
    options.add_argument(f"user-agent={random.choice(uas)}")
    # prefs = {"profile.managed_default_content_settings.images": 2}
    # options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options)


def load_all_reviews(driver):
    try:
        show_all_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "show_all"))
        )
        driver.execute_script("arguments[0].click();", show_all_btn)
        time.sleep(2)
    except:
        pass


def extract_reviews_joined(soup):
    reviews_list = []
    container = soup.find("div", id="comments_container")

    if container:
        review_cards = container.find_all("div", class_="o_review")

        for card in review_cards:
            text_tag = card.find("span", class_="text_full")
            r_text = text_tag.get_text(" ", strip=True) if text_tag else ""

            date_tag = card.find("span", class_="grey")
            r_date = date_tag.get_text(strip=True) if date_tag else "Unknown"

            score = "N/A"
            star_span = card.find("span", class_="stars")
            if star_span:
                inner_span = star_span.find("span", style=True)
                if inner_span:
                    try:
                        perc = float(inner_span['style'].split(":")[1].replace("%", "").replace(";", "").strip())
                        score = round(perc / 20, 1)
                    except:
                        pass

            reviews_list.append(f"[{score}] [{r_date}]: {r_text}")

    return " || ".join(reviews_list) if reviews_list else "N/A"


def extract_restaurant_data(url):
    driver = create_driver()

    name = "N/A"
    rating = "N/A"
    cuisines = "N/A"
    price_range = "N/A"
    location = "N/A"
    features = "N/A"
    contact_info = "N/A"
    joined_reviews = "N/A"  

    try:
        driver.get(url)
        time.sleep(random.uniform(8, 15))

        soup = BeautifulSoup(driver.page_source, "lxml")
        
        if "captcha" in driver.page_source.lower() or "verify you are human" in driver.page_source.lower():
            print(f"verification on {url}! You have 15 seconds to click the box...")
            time.sleep(15) 
            
            if "captcha" in driver.page_source.lower() or "verify you are human" in driver.page_source.lower():
                print(f"🚩 Still blocked. Skipping {url}")
                return None
            else:
                print("CAPTCHA bypassed! Continuing extraction...")

        # Name
        title_container = soup.find("div", class_="title_container")
        if title_container and title_container.find("h1"):
            name = title_container.find("h1").get_text(strip=True)

        # Rating
        stars_fill = soup.find("div", class_="rating-stars__fill")
        if stars_fill and "style" in stars_fill.attrs:
            try:
                perc = float(stars_fill['style'].replace("width:", "").replace("%", "").replace(";", "").strip())
                rating = round(perc / 20, 1)
            except:
                pass

        # Cuisines
        wrapper = soup.find("div", class_="cuisine_wrapper")
        if wrapper:
            all_cuisines = wrapper.find("div", class_="cuisine_hidden") or wrapper.find("div", class_="cuisine_shown")
            if all_cuisines:
                btn = all_cuisines.find("span", class_="more_cuisines__btn")
                if btn:
                    btn.decompose()
                cuisines = all_cuisines.get_text(separator=", ", strip=True).strip(", ")

        # Price
        cost_span = soup.find("span", class_="cost")
        if cost_span:
            price_range = len(cost_span.find_all("i"))

        # Location
        address_container = soup.find("div", id="info_location")
        if address_container:
            full_text = address_container.get_text(strip=True)
            location = full_text.replace("Address", "", 1).strip()

        # Features
        features_block = soup.find("div", class_="features_block")
        if features_block:
            overflow_div = features_block.find("div", class_="overflow")
            if overflow_div:
                features = overflow_div.get_text(separator=", ", strip=True)

        # Contact
        call_element = soup.find("a", class_="call")
        if call_element:
            contact_info = call_element.get("href", "").replace("tel:", "").strip()

        # Reviews
        load_all_reviews(driver)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "lxml")
        joined_reviews = extract_reviews_joined(soup)

        return {
            "name": name,
            "rating": rating,
            "cuisines": cuisines,
            "price_range": price_range,
            "location": location,
            "features": features,
            "contact_info": contact_info,
            "all_reviews": joined_reviews,
            "url": url
        }

    except Exception as e:
        print(f" Error on {url}: {e}")
        return None

    finally:
        driver.quit()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    with open(TXT_FILE, "r", encoding="utf-8") as f:
        all_links = [line.strip() for line in f if line.strip()]
        
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            processed = {line.strip() for line in f}

    links_to_do = [l for l in all_links if l not in processed][:BATCH_SIZE]

    print(f"Scraping {len(links_to_do)} links...")

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        batch_results = list(executor.map(extract_restaurant_data, links_to_do))
        results = [r for r in batch_results if r is not None]

    if results:
        df = pd.DataFrame(results)
        file_exists = os.path.isfile(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8')
        
        #update the progress file
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            for r in results:
                f.write(r['url'] + "\n")
        
        print(f" Batch complete. Saved {len(results)} records.")