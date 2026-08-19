
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from selenium.webdriver.common.by import By

import config
from driver_utils import make_driver, wait_for_ready, dump_debug

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scraper")


def _best_src(img_el, image_attrs) -> str | None:

    for attr in image_attrs:
        val = img_el.get_attribute(attr)
        if val:
            # srcset вида "url1 1x, url2 2x" -> берём первый url
            return val.split(",")[0].strip().split(" ")[0]
    return None


def _page_url(category_url: str, site_cfg: dict, page_num: int) -> str:
    if page_num <= 1:
        return category_url
    return site_cfg["pagination_pattern"].format(url=category_url.rstrip("/"), n=page_num)


def _is_real_product_image(url: str, must_contain: list) -> bool:
    if not must_contain:
        return True
    return any(part in url for part in must_contain)


def collect_image_urls_on_page(driver, site_cfg: dict, page_url: str) -> list[str]:
    driver.get(page_url)
    wait_for_ready(driver, extra_sleep=1.5)

    must_contain = site_cfg.get("image_url_must_contain", [])
    urls = []
    for selector in site_cfg["product_card_selectors"]:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if not elements:
            continue
        found_this_selector = []
        for el in elements:
            src = _best_src(el, site_cfg["image_attrs"])
            if src and _is_real_product_image(src, must_contain):
                found_this_selector.append(src)
        if found_this_selector:
            urls = found_this_selector
            break  

    if not urls:
        try:
            script_texts = driver.execute_script("""
                return Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                            .map(s => s.textContent);
            """)
            for raw in script_texts:
                data = json.loads(raw)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") == "ItemList":
                        for element in item.get("itemListElement", []):
                            img = element.get("item", {}).get("image")
                            if img and _is_real_product_image(img, must_contain):
                                urls.append(img)
        except Exception:
            pass

    return urls

def download_image(url: str, dest_dir: Path, prefix: str, index: int, seen_hashes: set,
                    referer: str = None) -> tuple[bool, str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer

        resp = requests.get(url, timeout=config.REQUEST_TIMEOUT, headers=headers)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        content = resp.content
        if not content:
            return False, "пустой ответ"

        digest = hashlib.md5(content).hexdigest()
        if digest in seen_hashes:
            return False, "дубликат"

        img = Image.open(BytesIO(content))
        img.verify() 
        img = Image.open(BytesIO(content))  
        if img.width < config.MIN_IMAGE_SIDE_PX or img.height < config.MIN_IMAGE_SIDE_PX:
            return False, f"слишком маленькая ({img.width}x{img.height})"

        img = img.convert("RGB")
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{prefix}_{index:04d}.jpg"
        img.save(out_path, "JPEG", quality=92)

        seen_hashes.add(digest)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def scrape_site_category(driver, site_key: str, category_url: str, dest_dir: Path,
                          seen_hashes: set, remaining: int) -> int:
    site_cfg = config.SITES[site_key]
    saved = 0
    idx = 0

    for page_num in range(1, config.MAX_PAGES_PER_CATEGORY + 1):
        if saved >= remaining:
            break
        page_url = _page_url(category_url, site_cfg, page_num)
        logger.info("  [%s] страница %d: %s", site_key, page_num, page_url)
        try:
            image_urls = collect_image_urls_on_page(driver, site_cfg, page_url)
        except Exception as exc:
            logger.error("  Ошибка на странице %s: %s", page_url, exc)
            break

        if not image_urls:
            if page_num == 1:
                logger.warning("  [%s] 0 картинок уже на первой странице — сохраняю debug-дамп", site_key)
                dump_debug(driver, f"{site_key}_no_images")
            else:
                logger.info("  [%s] страница %d пустая — категория закончилась", site_key, page_num)
            break

        logger.info("  [%s] страница %d: найдено %d кандидатов, пример: %s",
                    site_key, page_num, len(image_urls), image_urls[0])

        referer = site_cfg["base_url"]
        page_saved = 0
        fail_reasons = {}
        for url in image_urls:
            if saved >= remaining:
                break
            idx += 1
            ok, reason = download_image(url, dest_dir, site_key, idx, seen_hashes, referer=referer)
            if ok:
                saved += 1
                page_saved += 1
            else:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        logger.info("  [%s] страница %d: +%d новых (всего по сайту: %d)", site_key, page_num, page_saved, saved)
        if page_saved == 0 and fail_reasons:
            logger.warning("  [%s] страница %d: все скачивания провалились: %s",
                            site_key, page_num, fail_reasons)
            if page_num == 1:
                dump_debug(driver, f"{site_key}_download_failed")
        if page_saved == 0 and page_num > 1:
            break

    return saved


def scrape_category(driver, category_name: str, site_urls: dict) -> int:
    dest_dir = config.DATASET_DIR / category_name
    seen_hashes = set()
    total_saved = 0

    for site_key, category_url in site_urls.items():
        if total_saved >= config.IMAGES_PER_CATEGORY_TARGET:
            break
        logger.info("[%s / %s] старт: %s", category_name, site_key, category_url)
        remaining = config.IMAGES_PER_CATEGORY_TARGET - total_saved
        saved = scrape_site_category(driver, site_key, category_url, dest_dir, seen_hashes, remaining)
        total_saved += saved
        logger.info("  итого по категории '%s' пока: %d", category_name, total_saved)

    return total_saved


def main():
    if not config.CATEGORIES_CURATED_FILE.exists():
        logger.error(
            "Нет файла %s. Заполни его списком категорий (см. README).",
            config.CATEGORIES_CURATED_FILE,
        )
        return

    with open(config.CATEGORIES_CURATED_FILE, "r", encoding="utf-8") as f:
        categories = json.load(f)

    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    driver = make_driver()
    summary = {}
    try:
        for category_name, site_urls in categories.items():
            saved = scrape_category(driver, category_name, site_urls)
            summary[category_name] = saved
    finally:
        driver.quit()

    print("\n=== Итог ===")
    for cat, count in summary.items():
        status = "OK" if count >= 200 else "МАЛО, нужно >=200"
        print(f"  {cat}: {count} изображений ({status})")


if __name__ == "__main__":
    main()