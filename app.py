import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms


# ==============================
# Page configuration
# ==============================

st.set_page_config(
    page_title="Cat Dog Panda Classifier",
    page_icon="🐾",
    layout="centered"
)


# ==============================
# Device
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# CNN Model
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
# Load model
# ==============================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "models/classifier.pth",
        map_location=device
    )

    model = AnimalCNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    return model, classes, image_size


model, classes, image_size = load_model()


# ==============================
# Image preprocessing
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
# UI
# ==============================

st.title("🐾 Cat, Dog & Panda Classifier")

st.write(
    "Upload an image and the trained PyTorch CNN will "
    "predict whether it is a cat, dog, or panda."
)

st.info(
    f"Running on: "
    f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==============================
# Prediction
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        width="stretch"
    )

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[
        prediction.item()
    ]

    confidence_percent = (
        confidence.item() * 100
    )

    st.subheader("Prediction")

    st.success(
        f"🐾 {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence_percent:.2f}%"
    )

    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        classes,
        probabilities[0]
    ):

        percentage = probability.item() * 100

        st.write(
            f"**{class_name.capitalize()}**: "
            f"{percentage:.2f}%"
        )

        st.progress(
            float(probability.item())
        )