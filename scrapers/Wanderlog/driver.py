from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

from scraper import scrape_restaurant

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

def scrape_with_own_driver(href):
    driver = create_driver()
    wait = WebDriverWait(driver, 10)
    try:
        return scrape_restaurant(driver, href, wait)
    finally:
        driver.quit()