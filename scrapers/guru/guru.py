import csv
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os

TXT_FILE = "scrapers/guru/links/all_links.txt"
PROGRESS_FILE = "scrapers/guru/processed_links.txt"
OUTPUT_FILE = "data/guru.csv"
BATCH = 5

def load_all_reviews(driver):
    try:
        show_all_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CLASS_NAME, "show_all")))
        driver.execute_script("arguments[0].click();", show_all_btn)
        time.sleep(10)
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
            r_date = date_tag.get_text(strip=True) if date_tag else "Unknown date"
            
            score = "N/A"
            star_span = card.find("span", class_="stars")
            if star_span:
                inner_span = star_span.find("span", style=True)
                if inner_span:
                    try:
                        style_val = inner_span['style'].split(":")[1].replace("%", "").replace(";", "").strip()
                        score = round(float(style_val) / 20, 1)
                    except: pass
            
            reviews_list.append(f"[{score}] [{r_date}]: {r_text}")
    return " || ".join(reviews_list) if reviews_list else "N/A"

def extract_restaurant_data(url):
    # Initializing all variables to N/A
    name, rating, cuisines, price_range = "N/A", "N/A", "N/A", "N/A"
    location, features, contact_info, joined_reviews = "N/A", "N/A", "N/A", "N/A"

    options = Options()
    options.page_load_strategy = 'normal'
    options.add_argument("--disable-blink-features=AutomationControlled")
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "info_location"))
            )
            time.sleep(5) 
        except:
            print(f"⚠️ Page loaded but info blocks are missing for: {url}")

        soup = BeautifulSoup(driver.page_source, "lxml")

        # Name
        title_container = soup.find("div", class_="title_container")
        if title_container and title_container.find("h1"):
            name = title_container.find("h1").get_text(strip=True)

        # Rating
        stars_fill = soup.find("div", class_="rating-stars__fill")
        if stars_fill and "style" in stars_fill.attrs:
            try:
                style_string = stars_fill['style']
                percentage_str = style_string.replace("width:", "").replace("%", "").replace(";", "").strip()
                rating = round(float(percentage_str)/20, 1)
            except: pass

        # Cuisines
        c_wrapper = soup.find("div", class_="cuisine_wrapper")
        if c_wrapper:
            c_tag = c_wrapper.find("div", class_="cuisine_hidden") or c_wrapper.find("div", class_="cuisine_shown")
            if c_tag:
                cuisines = c_tag.get_text(separator=", ", strip=True)

        # Price Range
        cost_span = soup.find("span", class_="cost")
        if cost_span:
            price_range = len(cost_span.find_all("i"))

        # Location
        loc_div = soup.find("div", id="info_location")
        if loc_div:
            location = loc_div.get_text(strip=True).replace("Address", "", 1).strip()

        # Features
        feat_div = soup.find("div", class_="features_block")
        if feat_div:
            features = feat_div.get_text(separator=", ", strip=True)

        # Contact
        call_link = soup.find("a", class_="call")
        if call_link:
            contact_info = call_link.get("href", "").replace("tel:", "").strip()
            
        # REVIEWS
        load_all_reviews(driver)
        soup_after_reviews = BeautifulSoup(driver.page_source, "lxml")    
        joined_reviews = extract_reviews_joined(soup_after_reviews)
            
        return {
            "name": name, "rating": rating, "cuisines": cuisines,
            "price_range": price_range, "location": location,
            "features": features, "contact_info": contact_info,
            "all_reviews": joined_reviews, "url": url
        }
        
    except Exception as e:
        print(f"Error on {url}: {e}")
        return None
    finally:
        driver.quit()

if __name__ == "__main__":
    if not os.path.exists("data"): 
        os.makedirs("data")
    
    # Load links from file
    if not os.path.exists(TXT_FILE):
        print(f"Error: {TXT_FILE} not found.")
        exit()

    with open(TXT_FILE, "r", encoding="utf-8") as f:
        all_links = [line.strip() for line in f if line.strip()]

    # Load progress
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            processed = {line.strip() for line in f}

    links_to_scrape = [l for l in all_links if l not in processed][:BATCH]

    print(f"🚀 Starting sequential scrape for {len(links_to_scrape)} links...")

    for link in links_to_scrape:
        data = extract_restaurant_data(link)
        
        if data:
            # Save data to CSV
            df_new = pd.DataFrame([data])
            file_exists = os.path.isfile(OUTPUT_FILE)
            df_new.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8')
            
            # Update progress file
            with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
                f.write(link + "\n")
            
            print(f"✅ Saved: {data['name']}")
            delay = random.uniform(8, 13)
            print(f" Sleeping for {delay:.1f}s to avoid detection...")
            time.sleep(delay)
        else:
            print(f"⏭️ Skipping {link} due to block or error.")

    print(f" Batch complete! Check {OUTPUT_FILE}")