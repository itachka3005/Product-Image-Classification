import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import gradio as gr
from pyngrok import ngrok

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    if image is None:
        return {}
    

    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Загрузите фото товара"),
    outputs=gr.Label(num_top_classes=5, label="Результат классификации"),
    title="Классификатор товаров",
    description="Загрузите изображение смартфона, ноутбука, телевизора, часов или наушников для распознавания.",
    theme="soft"
)


ngrok.set_auth_token("3I1tDsJmZhPfrQNpdkyqrepmaTl_5ZmqxhaQ5trTAk74ahrW2")

if __name__ == "__main__":
    port = 7860
    public_url = ngrok.connect(port)
    print("Публичный URL:", public_url)
    interface.launch(server_name="0.0.0.0", server_port=port, share=True)