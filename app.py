import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

# Load the full model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("deepfake_detector.pth", map_location=device)
model.eval()

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return "Fake" if predicted.item() == 1 else "Real"

# Gradio UI
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="pil"),
                         outputs="text",
                         title="Deepfake Detector",
                         description="Upload an image to check if it's Real or Fake")

interface.launch()

