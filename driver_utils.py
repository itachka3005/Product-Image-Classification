

import time
import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import config

logger = logging.getLogger("scraper")


def make_driver(headless: bool = config.HEADLESS) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(config.PAGE_LOAD_TIMEOUT)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def wait_for_ready(driver, extra_sleep: float = 2.5, timeout: int = config.PAGE_LOAD_TIMEOUT):
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
    except Exception:
        pass

    try:
        driver.execute_script("""
            const modals = document.querySelectorAll('.modal-backdrop, .modal-content, [class*="modal"]');
            modals.forEach(el => el.remove());
            document.body.style.overflow = 'unset';
        """)
    except Exception:
        pass

    time.sleep(extra_sleep)


def dump_debug(driver, name: str):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = config.OUTPUT_DIR / f"debug_{name}.png"
    html_path = config.OUTPUT_DIR / f"debug_{name}.html"
    try:
        driver.save_screenshot(str(png_path))
        html_path.write_text(driver.page_source, encoding="utf-8")
        logger.warning("Debug-дамп сохранён: %s , %s (current_url=%s)", png_path, html_path, driver.current_url)
    except Exception as exc:
        logger.error("Не удалось сохранить debug-дамп для %s: %s", name, exc)


def scroll_to_load_all(driver, pause=config.SCROLL_PAUSE_SEC, max_scrolls=config.MAX_SCROLLS_PER_PAGE):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def click_if_present(driver, selectors):
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException

    for sel in selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
            return True
        except (NoSuchElementException, ElementClickInterceptedException):
            continue
    return False
