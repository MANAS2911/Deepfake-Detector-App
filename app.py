
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Define the model class
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1)
        ])

        self.top_down_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fpn_layers[0](x)
        x = self.classifier(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
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
