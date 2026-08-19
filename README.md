# Product Image Classification

Классификатор изображений товаров (смартфоны, ноутбуки, телевизоры, часы,
наушники), собранных с asaxiy.uz, texnomart.uz, olcha.uz.

Проект состоит из трёх этапов ML Task: **Parsing** (сбор датасета) →
**Data Preparation** (очистка и разбиение) → **Modeling** (fine-tuning ViT
+ инференс через Gradio).

## Итоговое качество модели

Модель — дообученный `google/vit-base-patch16-224-in21k` (Vision
Transformer), fine-tuned на собственном датасете.

```
              precision    recall  f1-score   support
  headphones     0.9091    0.9091    0.9091        33
      laptop     0.9655    0.9032    0.9333        31
  smartphone     0.8684    1.0000    0.9296        33
          tv     1.0000    0.9697    0.9846        33
       watch     1.0000    0.9394    0.9688        33
    accuracy                         0.9448       163
```

Confusion matrix сохраняется в `output/confusion_matrix.png` после запуска
`evaluate.py`.

## Установка (локально)

1. Установи Google Chrome (нужен для парсинга, драйвер подтянется
   автоматически через `webdriver-manager`).
2. Создай окружение и поставь зависимости:
   ```bash
   python -m venv venv
   venv\Scripts\activate        
   pip install -r requirements.txt
   ```

## Структура проекта и порядок запуска

### 1. Parsing — сбор датасета

`output/categories.json` уже заполнен проверенными ссылками на 5 категорий
по всем трём сайтам.

```bash
python scrape_images.py
```

Складывает картинки в `dataset/<category_name>/<site>_<n>.jpg`. Подробности
о том, как это устроено и что делать, если сайт поменял вёрстку — см. в
конце README, раздел "Известные ограничения парсинга".

`discover_categories.py` — опциональный инструмент для поиска новых
категорий (не работает для texnomart.uz — Nuxt SPA, категории туда внесены
вручную).

### 2. Data Preparation — очистка и разбиение

```bash
python prepare_data.py
```

Отсеивает битые файлы и дубли (по md5-хешу), делит датасет на
`train/val/test` (70/15/15) и складывает в `dataset_prepared/`.

### 3. Modeling — fine-tuning ViT

```bash
python train.py
```

Дообучает `google/vit-base-patch16-224-in21k` на `dataset_prepared/` через
`transformers.Trainer`. Модель сохраняется как папка (формат Hugging Face)
в `models/vit-product-classifier/` — а не единым `.pth`-файлом, как было
в первой версии на ResNet18.

**Важно про версии:** зафиксируйте `transformers==4.46.3` и
`accelerate==0.34.2` (см. `requirements.txt`) — более свежие версии могут
переименовывать внутренние слои ViT и ломать загрузку предобученных весов
(проверено на практике — сыпется предупреждениями `UNEXPECTED`/`MISSING`
почти по всем слоям энкодера, и веса фактически не подгружаются).

**Обучение на GPU (рекомендуется):** ViT заметно тяжелее ResNet18, на CPU
обучение медленное. Проще всего прогнать `train.py` на бесплатном GPU в
Kaggle Notebooks (Settings → Accelerator → GPU T4 x2 или P100; для этого
нужна подтверждённая по телефону учётка Kaggle). Датасет `dataset_prepared`
заливается туда как Kaggle Dataset, пути `PREPARED_DIR`/`MODEL_SAVE_DIR` в
`train.py` меняются на `/kaggle/input/...` и `/kaggle/working/...`
соответственно. Готовую модель после обучения архивируем
(`shutil.make_archive`) и скачиваем через панель Output.

### Оценка модели

```bash
python evaluate.py
```

Строит `classification_report` и confusion matrix на `dataset_prepared/test`.

### Инференс из командной строки

```bash
python predict.py "путь_к_картинке.jpg"
```

### Веб-интерфейс (Gradio)

```bash
python app.py
```

Поднимает локальный сервер на `http://127.0.0.1:7860` и (если задан
`NGROK_AUTH_TOKEN`) публичную ссылку через ngrok.

Токен ngrok передаётся через переменную окружения, не хардкодится в коде:
```powershell
$env:NGROK_AUTH_TOKEN="ваш_токен"
python app.py
```
Получить токен: https://dashboard.ngrok.com/get-started/your-authtoken

## Известные ограничения парсинга

- Сайты могут поменять вёрстку/добавить антибот-защиту — тогда
  `product_card_selectors`/`image_attrs` в `config.py` придётся обновить.
  Если для какой-то категории/сайта картинок собралось 0 или мало — в
  `output/` сохранится debug-дамп (`.png` + `.html`) для разбора.
- `IMAGES_PER_CATEGORY_TARGET = 220` в `config.py` — с запасом сверх 200,
  т.к. на этапе Data Preparation часть неизбежно уходит (дубли, брак).
- При более агрессивной антибот-защите может понадобиться
  `undetected-chromedriver` вместо текущей базовой маскировки в
  `driver_utils.py`.

## Структура репозитория

```
├── config.py                  # конфиг сайтов-источников для парсинга
├── driver_utils.py             # утилиты Selenium (маскировка, debug-дампы)
├── discover_categories.py      # (опционально) поиск новых категорий
├── scrape_images.py            # сбор изображений
├── prepare_data.py             # очистка + train/val/test split
├── train.py                    # fine-tuning ViT
├── evaluate.py                 # оценка на test + confusion matrix
├── predict.py                  # инференс из CLI
├── app.py                      # веб-интерфейс Gradio + ngrok
├── requirements.txt
├── output/                     # categories.json, confusion_matrix.png, debug-дампы
├── dataset/                    # сырые изображения (не в git)
├── dataset_prepared/           # train/val/test (не в git)
└── models/
    └── vit-product-classifier/ # дообученная модель (см. .gitignore)
```