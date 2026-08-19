
import json
import logging
from urllib.parse import urljoin, urlparse

from selenium.webdriver.common.by import By

import config
from driver_utils import make_driver, wait_for_ready, dump_debug

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scraper")

GENERIC_EXCLUDE = [
    "login", "signin", "sign-in", "register", "cart", "checkout", "account",
    "wishlist", "favorite", "compare", "search", "javascript:", "tel:",
    "mailto:", "#", "facebook.com", "instagram.com", "telegram.me", "t.me",
    "youtube.com", "twitter.com", "whatsapp.com", "/blog/", "/news/",
    "/contacts", "/about", "/help", "/faq", "/delivery", "/payment",
    "/privacy", "/terms", "/policy",
]


def _is_plausible_path(path: str) -> bool:
    path = path.strip("/")
    if not path or path.count("/") > 3:
        return False
    last_segment = path.split("/")[-1]
    if last_segment.isdigit():
        return False
    if len(last_segment) < 2:
        return False
    return True


def discover_site(driver, site_key: str, site_cfg: dict) -> list[dict]:
    logger.info("Открываю %s", site_cfg["base_url"])
    driver.get(site_cfg["base_url"])
    wait_for_ready(driver)

    domain = urlparse(site_cfg["base_url"]).netloc
    found = {}  


    for selector in site_cfg["link_container_selectors"]:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        for el in elements:
            href = el.get_attribute("href")
            if not href:
                continue
            low = href.lower()
            if any(bad in low for bad in site_cfg["href_exclude"]):
                continue
            if not any(good in low for good in site_cfg["href_include"]):
                continue
            _register(found, site_cfg["base_url"], href, (el.text or "").strip())

    if not found:
        logger.info("%s: сконфигурированные селекторы ничего не дали, пробую общий скан всех ссылок", site_key)
        all_links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
        for el in all_links:
            href = el.get_attribute("href")
            if not href:
                continue
            low = href.lower()
            if any(bad in low for bad in GENERIC_EXCLUDE):
                continue
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc != domain:
                continue
            if not _is_plausible_path(parsed.path):
                continue
            _register(found, site_cfg["base_url"], href, (el.text or "").strip())

    results = list(found.values())
    logger.info("%s: найдено %d кандидатов в категории", site_key, len(results))

    if not results:
        logger.warning("%s: 0 результатов даже общим сканом — сохраняю debug-дамп для разбора", site_key)
        dump_debug(driver, site_key)

    return results


def _register(found: dict, base_url: str, href: str, text: str):
    full_url = urljoin(base_url, href)
    parsed = urlparse(full_url)
    norm_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    entry = found.setdefault(norm_url, {"text": text, "count": 0, "url": norm_url})
    entry["count"] += 1
    if text and not entry["text"]:
        entry["text"] = text


def _rank_categories(entries: list[dict], top_n: int) -> list[dict]:
    ranked = sorted(entries, key=lambda e: e["count"], reverse=True)
    return ranked[:top_n]


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    driver = make_driver()
    try:
        for site_key, site_cfg in config.SITES.items():
            try:
                entries = discover_site(driver, site_key, site_cfg)
            except Exception as exc:
                logger.error("Не удалось обработать %s: %s", site_key, exc)
                entries = []
            all_results[site_key] = entries
    finally:
        driver.quit()

    with open(config.CATEGORIES_DISCOVERED_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info("Сохранено: %s", config.CATEGORIES_DISCOVERED_FILE)

    print("\n=== Топ кандидаты по сайтам (эвристика, проверь глазами!) ===")
    for site_key, entries in all_results.items():
        print(f"\n--- {site_key} ---")
        for e in _rank_categories(entries, config.N_CATEGORIES * 2):
            print(f"  [{e['count']:>2}] {e['text'][:40]:40s} {e['url']}")

    print(
        "\nДалее: открой output/discovered_categories.json, выбери 5 категорий, "
        "которые есть (желательно) на всех трёх сайтах, и оформи их в "
        "output/categories.json по образцу из categories.example.json."
    )


if __name__ == "__main__":
    main()
