import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Define the model class exactly as in training
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        ])
        
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        c2 = self.resnet[:5](x)
        c3 = self.resnet[5](c2)
        c4 = self.resnet[6](c3)
        c5 = self.resnet[7](c4)
        
        p5 = self.fpn_layers[0](c5)
        p4 = self.fpn_layers[1](c4) + nn.functional.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.fpn_layers[2](c3) + nn.functional.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.fpn_layers[3](c2) + nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
        
        d5 = self.top_down_layers[0](p5)
        d4 = self.top_down_layers[1](p4)
        d3 = self.top_down_layers[2](p3)
        d2 = self.top_down_layers[3](p2)
        
        d5 = nn.functional.interpolate(d5, size=d2.shape[2:], mode='nearest')
        d4 = nn.functional.interpolate(d4, size=d2.shape[2:], mode='nearest')
        d3 = nn.functional.interpolate(d3, size=d2.shape[2:], mode='nearest')
        
        combined = d5 + d4 + d3 + d2
        out = self.final_layer(combined)
        return out

# Load the model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
model.eval()

# Define image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "Fake" if predicted.item() == 1 else "Real"

# Gradio interface
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="pil"),
                         outputs="text",
                         title="Deepfake Detector",
                         description="Upload an image to check if it's Real or Fake.")

interface.launch(share = True)

