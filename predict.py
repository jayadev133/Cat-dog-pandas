import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ==============================
# 1. Device
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# 2. Model
# ==============================

class AnimalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================
# 3. Load trained model
# ==============================

checkpoint = torch.load(
    "models/classifier.pth",
    map_location=device
)

model = AnimalCNN().to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

classes = checkpoint["classes"]
image_size = checkpoint["image_size"]


# ==============================
# 4. Image preprocessing
# ==============================

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# ==============================
# 5. Get image path
# ==============================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python predict.py path_to_image.jpg")
    sys.exit()


image_path = sys.argv[1]


# ==============================
# 6. Load image
# ==============================

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# ==============================
# 7. Prediction
# ==============================

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)


predicted_class = classes[prediction.item()]

confidence_percent = confidence.item() * 100


# ==============================
# 8. Result
# ==============================

print("\n" + "=" * 50)

print("PREDICTION RESULT")

print("=" * 50)

print("Image:", image_path)

print("Predicted class:", predicted_class)

print(f"Confidence: {confidence_percent:.2f}%")

print("=" * 50)