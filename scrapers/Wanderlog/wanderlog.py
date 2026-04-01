from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

from config import FIELDNAMES
from driver import create_driver, scrape_with_own_driver

MAX_WORKERS = 3

# First: 
# Collect links
driver = create_driver()
wait = WebDriverWait(driver, 10)

try:
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

finally:
    driver.quit()


# Second: 
# Scrape in parallel by multithreading
# then write to CSV

with open("beirut_restaurants.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_with_own_driver, href): href for href in hrefs}

        for i, future in enumerate(as_completed(futures)):
            href = futures[future]
            try:
                row = future.result()
                writer.writerow(row)
                print(f"  [{i+1}/{len(hrefs)}] Done: {row['name']}")
            except Exception as e:
                print(f"  Failed: {href} → {e}")

print("Done! Saved to beirut_restaurants.csv")