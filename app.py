import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from pyngrok import ngrok
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vit-product-classifier"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()


def predict_image(image: Image.Image):
    if image is None:
        return {}

    image_rgb = image.convert("RGB")
    inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    confidences = {
        model.config.id2label[i]: float(probabilities[i])
        for i in range(len(model.config.id2label))
    }
    return confidences


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Загрузите фото товара"),
    outputs=gr.Label(num_top_classes=5, label="Результат классификации"),
    title="Классификатор товаров (ViT)",
    description="Загрузите изображение смартфона, ноутбука, телевизора, часов или наушников для распознавания.",
    theme="soft",
)

ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)

if __name__ == "__main__":
    port = 7860
    public_url = ngrok.connect(port)
    print("Публичный URL:", public_url)
    interface.launch(server_name="0.0.0.0", server_port=port, share=True)