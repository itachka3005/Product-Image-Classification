"""
Если понадобится подправить что-то ещё (например для новых категорий,
которых нет в готовом categories.json) — правь именно этот файл, весь
остальной код от конкретной верстки не зависит.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CATEGORIES_DISCOVERED_FILE = OUTPUT_DIR / "discovered_categories.json"
CATEGORIES_CURATED_FILE = OUTPUT_DIR / "categories.json"
DATASET_DIR = BASE_DIR / "dataset"
LOG_DIR = BASE_DIR / "logs"

N_CATEGORIES = 5
IMAGES_PER_CATEGORY_TARGET = 220  
PAGE_LOAD_TIMEOUT = 20
SCROLL_PAUSE_SEC = 1.2
MAX_SCROLLS_PER_PAGE = 20
MAX_PAGES_PER_CATEGORY = 15  
HEADLESS = True
REQUEST_TIMEOUT = 15
MIN_IMAGE_SIDE_PX = 200  

SITES = {
    "asaxiy": {
        "base_url": "https://asaxiy.uz",
        "link_container_selectors": ["a"],
        "href_include": ["/product/"],
        "href_exclude": ["login", "cart", "checkout", "account", "wishlist", "#", "/uslugi/"],
        "pagination_pattern": "{url}/page={n}",
        "product_card_selectors": [
            "[class*='product-card'] img",
            "[class*='card'] img",
            "a[href*='/product/'] img",
        ],
        "image_attrs": ["data-src", "data-original", "data-lazy-src", "lazy-src", "src", "srcset"],
        "image_url_must_contain": ["assets.asaxiy.uz"],
        "load_more_selectors": [
            "button[class*='load-more']",
            "button[class*='show-more']",
        ],
    },
    "texnomart": {
        "base_url": "https://texnomart.uz",
        "link_container_selectors": ["a"],
        "href_include": ["/katalog/"],
        "href_exclude": ["login", "cart", "checkout", "account", "wishlist", "#"],
        "pagination_pattern": "{url}?page={n}",
        "product_card_selectors": [
            "[class*='product-card'] img",
            "[class*='card'] img",
            "a[href*='/product/detail/'] img",
        ],
        "image_attrs": ["data-src", "data-original", "data-lazy-src", "lazy-src", "src", "srcset"],
        "image_url_must_contain": ["mini-io-api.texnomart.uz"],
        "load_more_selectors": [
            "button[class*='load-more']",
            "button[class*='show-more']",
        ],
    },
    "olcha": {
        "base_url": "https://olcha.uz",
        "link_container_selectors": ["a"],
        "href_include": ["/category/"],
        "href_exclude": ["login", "cart", "checkout", "account", "wishlist", "#"],
        "pagination_pattern": "{url}?page={n}",
        "product_card_selectors": [
            "[class*='product-card'] img",
            "[class*='card'] img",
            "a[href*='/product/'] img",
        ],
        "image_attrs": ["data-src", "data-original", "data-lazy-src", "lazy-src", "src", "srcset"],
        "image_url_must_contain": [],
        "load_more_selectors": [
            "button[class*='load-more']",
            "button[class*='show-more']",
        ],
    },
}