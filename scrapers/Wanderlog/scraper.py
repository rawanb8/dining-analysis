from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
import json
import re

from config import FIELDNAMES
from helpers import extract_cuisine

def scrape_restaurant(driver, url, wait):
    data = {field: "N/A" for field in FIELDNAMES}
    description = ""
    reviews_summary = ""

    try:
        driver.get(url)

        # Name
        try:
            data["name"] = wait.until(EC.presence_of_element_located((
                By.XPATH, "//h1[contains(@class,'PlacePageHeader__title')]"
            ))).text.strip()
        except Exception:
            pass

        # Website
        try:
            data["website"] = wait.until(EC.presence_of_element_located((
                By.XPATH, "//h6[contains(text(),'Website')]/following-sibling::a[1]"
            ))).get_attribute("href")
        except Exception:
            pass

        # Address
        try:
            address_element = wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "a[href*='google.com/maps/search']"
            )))
            data["address"] = address_element.text.strip()
            data["address_link"] = address_element.get_attribute("href")
        except Exception:
            pass

        # Ranking
        try:
            rank_element = wait.until(EC.presence_of_element_located((
                By.XPATH,
                "//a[contains(@class,'text-muted') and .//span[contains(@class,'font-weight-bold')]]"
                "/span[contains(@class,'font-weight-bold')]"
            )))
            ranking_text = rank_element.text.strip()
            rank_match = re.search(r'#(\d+)', ranking_text)
            data["rank_number"] = rank_match.group(1) if rank_match else "N/A"
        except Exception:
            pass

        # Phone
        try:
            data["phone"] = wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "a[href*='tel:']"
            ))).text.strip()
        except Exception:
            pass

        # Description
        try:
            description = wait.until(EC.presence_of_element_located((
                By.XPATH, "//div[contains(@class,'mt-5')]//div[not(@class)]"
            ))).text.strip()
            data["description"] = description
        except Exception:
            pass

        # Why to go
        try:
            features_element = wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "div.mt-5 > div > div"
            )))
            data["why_to_go"] = features_element.text.strip()
        except Exception:
            pass

        # Reviews Summary
        try:
            summary_element = wait.until(EC.presence_of_element_located((
                By.XPATH, "//h2[contains(text(),'Reviews')]/following-sibling::text()[1]/.."
            )))
            reviews_summary = driver.execute_script(
                "return arguments[0].childNodes[1].textContent;", summary_element
            ).strip()
            data["reviews_summary"] = reviews_summary
        except Exception:
            pass

        # Cuisine
        cuisine = extract_cuisine(description)
        if not cuisine:
            cuisine = extract_cuisine(reviews_summary)
        data["cuisine"] = cuisine if cuisine else "N/A"

        # Menu
        try:
            page_source = driver.page_source
            match = re.search(r'window\.__MOBX_STATE__\s*=\s*({.*?});', page_source, re.DOTALL)
            if match:
                state = json.loads(match.group(1))
                menu_items_raw = state["placePage"]["data"]["menuItems"]
                menu_items = [
                    item["name"] for item in menu_items_raw
                    if item["name"].strip() and item["name"].lower() != "menu"
                ]
                menu_link = next(
                    (item["captionURL"] for item in menu_items_raw if item.get("captionURL")),
                    None
                )
                data["menu_items"] = ", ".join(menu_items) if menu_items else "N/A"
                data["menu_link"] = menu_link if menu_link else "N/A"
        except Exception:
            pass

        # Working Hours
        try:
            hour_elements = driver.find_elements(
                By.XPATH,
                "//div[contains(@class,'d-flex') and contains(@class,'mb-1') and contains(@class,'align-items-center')]"
                "//span[contains(@class,'color-gray-600')]"
            )
            hours = [el.text.strip() for el in hour_elements if el.text.strip() and ":" in el.text]
            data["working_hours"] = " | ".join(hours) if hours else "N/A"
        except Exception:
            pass

        # Tips
        try:
            tips_elements = driver.find_elements(
                By.XPATH,
                "//ul[contains(@class,'fa-ul') and contains(@class,'list-group')]//li"
            )
            tips = [el.text.strip() for el in tips_elements if el.text.strip()]
            data["tips"] = " | ".join(tips) if tips else "N/A"
        except Exception:
            pass

        # Ratings
        try:
            rating_containers = driver.find_elements(
                By.XPATH,
                "//div[contains(@class,'ComponentBreakpoints__col-sm-6')]//div[.//div[contains(@class,'font-size-48')]]"
            )
            platforms = [
                ("google_rating", "google_label", "google_reviews"),
                ("tripadvisor_rating", "tripadvisor_label", "tripadvisor_reviews")
            ]
            for i, container in enumerate(rating_containers[:2]):
                try:
                    rating = container.find_element(
                        By.XPATH, ".//div[contains(@class,'font-size-48')]"
                    ).text.strip().replace("Review score", "").replace("out of 5", "").strip().split("\n")[0]
                    label = container.find_element(By.XPATH, ".//strong").text.strip()
                    num_reviews = container.find_element(
                        By.XPATH, ".//div[contains(@class,'text-muted')]"
                    ).text.strip().replace(" reviews", "")
                    data[platforms[i][0]] = rating
                    data[platforms[i][1]] = label
                    data[platforms[i][2]] = num_reviews
                except NoSuchElementException:
                    pass
        except Exception:
            pass

        # Star Distribution
        try:
            star_map = {
                "5 stars": "star_5", "4 stars": "star_4", "3 stars": "star_3",
                "2 stars": "star_2", "1 star": "star_1"
            }
            star_rows = driver.find_elements(By.XPATH, "//div[contains(@class,'d-table-row')]")
            for row in star_rows:
                try:
                    label = row.find_element(
                        By.XPATH, ".//div[contains(@class,'font-weight-bold') and contains(@class,'text-nowrap')]"
                    ).text.strip()
                    count = row.find_element(
                        By.XPATH, ".//div[contains(@class,'col-1') and not(contains(@class,'font-weight-bold'))]"
                    ).text.strip()
                    if label in star_map and count:
                        data[star_map[label]] = int(count)
                except NoSuchElementException:
                    pass
        except Exception:
            pass

        # Individual Reviews
        try:
            review_blocks = driver.find_elements(
                By.XPATH, "//div[contains(@id,'PlaceReviewsSection__review-')]"
            )
            reviews = []
            for block in review_blocks:
                try:
                    r_rating = block.find_element(By.XPATH, ".//strong[1]").text.strip()
                    r_date = block.find_element(By.XPATH, ".//a[contains(@href,'maps/reviews')]").text.strip()
                    r_text = block.find_element(By.XPATH, ".//div[contains(@class,'ExpandableText__text')]").text.strip()
                    reviews.append({
                        "rating": r_rating,
                        "date": r_date,
                        "text": r_text
                    })
                except Exception:
                    continue
            data["reviews"] = " || ".join(
                [f"[{r['rating']}] [{r['date']}]: {r['text']}" for r in reviews]
            ) if reviews else "N/A"
        except Exception:
            pass

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return data