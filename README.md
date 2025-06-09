# 🎭 Advanced Deepfake Detection System

A deep learning-powered image forensics system built using ResNet50 and Feature Pyramid Networks (FPN) for multi-scale, hierarchical detection of deepfake images. The system efficiently detects tampered media by combining fine-grained and global image features, achieving superior detection rates across multiple benchmark datasets.

---

## 🚀 Features

- 🧠 Dual-architecture model combining ResNet50 and FPN.
- 📊 Multi-scale image feature extraction for robust deepfake detection.
- 📈 Achieved 93.09% Accuracy and 98.56% ROC-AUC on the OpenForensics dataset.
- 🎨 Integrated Grad-CAM heatmaps for model explainability.
- 📄 Automated PDF report generation for prediction results.
- ⚙️ Fully customizable PyTorch-based training pipeline.
- 🏆 Participated in Microsoft AI Agents Hackathon 2025.

---

## 📁 Project Structure

Deepfake-Detection-System
- models (Trained model weights)
- data - Dataset folders (train/validation/test)
- app.py                       (Code for making webapp)
- results/                       (Prediction results, confusion matrices, and plots)
- requirements.txt               (Dependencies)
- README.md                      (Project documentation)

## 🛠️ Installation

1. **Clone the repository**
git clone https://github.com/your-username/Deepfake-Detection-System.git
cd Deepfake-Detection-System

2. **Create a virtual environment (optional but recommended)**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

## 📊 Model Details

A dual-architecture deep learning model:
- ResNet50 for hierarchical feature extraction.
- Feature Pyramid Networks (FPN) for multi-scale feature fusion.

Evaluation Metrics:
- Accuracy: 93.09%
- Recall: 90.13%
- ROC-AUC: 98.56%

## 🧠 Model Explainability

- Integrated Grad-CAM heatmaps for visualizing manipulated image regions.
- Generate visual insights alongside predictions to enhance model transparency.
- Automated PDF report creation with detection results and Grad-CAM visualizations.

## 📌 Requirements

All dependencies are listed in requirements.txt. Key libraries include:
- pytorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- opencv-python
- reportlab

## 🙋‍♂️ Author

Manas Choudhary,
Final Year Computer Engineering Student,
Project: Advanced Deepfake Detection System

Feel free to connect on [LinkedIn](www.linkedin.com/in/contactmanaschoudhary) or raise an issue or PR.

## ⭐ Star This Repository
If you like this project, give it a ⭐ to help others find it!

## App Link
[Deepfake Detection System](https://huggingface.co/spaces/Maddy2911/deepfake-detector)

