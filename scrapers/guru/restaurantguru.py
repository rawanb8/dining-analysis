import winsound
import ctypes
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from concurrent.futures import ThreadPoolExecutor

cities_to_scrape={
    "Beirut": "https://t.restaurantguru.com/restaurant-Beirut-t1",
    "Sidon": "https://t.restaurantguru.com/restaurant-Sidon-t1",
    "Bourj Hammoud": "https://t.restaurantguru.com/restaurant-Bourj-Hammoud-t1",
    "Zahle": "https://t.restaurantguru.com/restaurant-Zahle-t1",
    "Bar Elias": "https://t.restaurantguru.com/restaurant-Bar-Elias-t1",
    "Hazmiyeh": "https://t.restaurantguru.com/restaurant-Hazmiyeh-t1",
    "Tyre": "https://t.restaurantguru.com/restaurant-Tyre-t1",
    "Forn El Chebbak": "https://t.restaurantguru.com/restaurant-Tyre-t1",
    "Byblos": "https://t.restaurantguru.com/restaurant-Byblos-t1",
    "Khalde": "https://t.restaurantguru.com/restaurant-Khalde-t1",
    "Tripoli": "https://t.restaurantguru.com/restaurant-Tripoli-North-t1"
}
#bypass captcha (it wont work if headless mode is on)
def alert_user_for_captcha():
    """Plays a sound and shows a popup if a CAPTCHA is detected."""
    for _ in range(3):
        winsound.Beep(1000, 500) 
        time.sleep(0.1)
    ctypes.windll.user32.MessageBoxW(0, "🚨 CAPTCHA DETECTED! Solve it in the browser, then click OK here.", "Scraper Alert", 0x1000)

def check_for_captcha(driver):
    #Checks for 'unusual traffic' or recaptcha and pauses for manual solution.
    captcha_keywords = ["verify that you are human", "unusual traffic", "g-recaptcha"]
    try:
        page_text = driver.page_source.lower()
        if any(word in page_text for word in captcha_keywords):
            print("🚨 Captcha detected! Alerting user...")
            alert_user_for_captcha()
            # Wait loop until keywords are gone
            while any(word in driver.page_source.lower() for word in captcha_keywords):
                print("Waiting for you to solve the captcha...")
                time.sleep(10)
            print("✅ Captcha cleared. Resuming...")
            return True
    except:
        pass 
    return False

def handle_vignette(driver):
    #close google ad if present
    if "#google_vignette" in driver.current_url:
        try:
            print("Ad detected. Attempting to close...")
            close_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "dismiss-button"))
            )
            close_btn.click()
            print("Ad closed.")
            time.sleep(2)
        except:
            print("Could not find dismiss button, moving on...")

def scrape_city_links(city_name, city_url):
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    #doesnt load images not to exhaust ram
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    city_links = set()
    
    try:
        print(f"[Thread-{city_name}]: {city_url}")
        driver.get(city_url)
        check_for_captcha(driver)
        handle_vignette(driver)

        last_height = driver.execute_script("return document.body.scrollHeight")
        retries = 0
        #give the page 3 chances to load(ktir am yekhod wa2t to load w its missing data)
        while retries < 3: 
            #scroll by chunks
            check_for_captcha(driver)
            try:
                
                for _ in range(3):
                    driver.execute_script("window.scrollBy(0, 1500);")
                    time.sleep(random.uniform(1.5, 2.5))
                
                soup = BeautifulSoup(driver.page_source, "lxml")
                containers = soup.find_all("div", class_="rest-card__title")
                
                new_found = 0
                for div in containers:
                    link_tag = div.find("a", href=True)
                    if link_tag:
                        href = link_tag['href']
                        if "restaurantguru.com" in href and href not in city_links:
                            city_links.add(href)
                            new_found += 1
                print(f"[{city_name}] Found {new_found} new. Total unique: {len(city_links)}")
                
            except TimeoutException:
                print(f" [{city_name}] interaction timed out. waiting 10 seconds")
                time.sleep(10)
                continue
            
            #check if we actually reached bottom scroll
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                retries += 1
                print(f" [{city_name}] No new height. Retry {retries}/3...")
                time.sleep(10) 
            else:
                retries = 0 
                last_height = new_height

        #save to files
        with open(f"scrapers/guru/links/{city_name}_links.txt", "w", encoding="utf-8") as f:
            for link in sorted(city_links):
                f.write(link + "\n")
                
        print(f"[{city_name}] Final Count: {len(city_links)}")

    except Exception as e:
        print(f"❌ [{city_name}] Error: {e}")
    finally:
        driver.quit()
def run_multi_city_part_1():
    print("--- STARTING Part 1: LINK COLLECTION ---")
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(lambda p: scrape_city_links(*p), cities_to_scrape.items())
    print("\n--- ALL CITIES FINISHED ---")

if __name__ == "__main__":
    run_multi_city_part_1()