from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

from config import FIELDNAMES
from scraper import scrape_restaurant

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

try:
    # First: 
    # Collect links
    driver.get("https://wanderlog.com/list/geoCategory/30885/best-restaurants-to-have-dinner-in-beirut")
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h2 a.color-gray-900")))

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    links = driver.find_elements(By.CSS_SELECTOR, "h2 a.color-gray-900")
    hrefs = list(dict.fromkeys([l.get_attribute("href") for l in links]))
    print(f"Found {len(hrefs)} restaurants")

    # Second: 
    # Scrape & write to CSV
    with open("beirut_restaurants.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, href in enumerate(hrefs):
            print(f"Scraping [{i+1}/{len(hrefs)}]: {href}")
            row = scrape_restaurant(driver, href, wait)
            writer.writerow(row)
            print(f"  Done: {row['name']}")

finally:
    driver.quit()
    print("Done! Saved to beirut_restaurants.csv")