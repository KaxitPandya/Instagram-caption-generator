import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

APP_URL = os.environ.get("STREAMLIT_APP_URL")
if not APP_URL:
    raise SystemExit("Missing STREAMLIT_APP_URL env var")

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )

    try:
        driver.get(APP_URL)
        time.sleep(5)

        # Try to click the wake button if present
        # Streamlit's wake screen often contains wording like:
        # "Yes, get this app back up!"
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for b in buttons:
                txt = (b.text or "").strip().lower()
                if "get this app back up" in txt or "wake" in txt:
                    b.click()
                    time.sleep(5)
                    break
        except NoSuchElementException:
            pass

        # Optional: wait a bit so the app actually boots
        time.sleep(10)
        print("Done. App should be awake/warmed.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
