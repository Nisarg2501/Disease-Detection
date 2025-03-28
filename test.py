import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import base64

# 🎨 Function to set background image
def set_bg_image(image_file):
    with open(image_file, "rb") as file:
        img_data = base64.b64encode(file.read()).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Set background (Ensure 'background.jpg' is in the same folder)
set_bg_image("background.jpeg")

# Define class labels for diseases (Modify this based on your dataset)
CLASS_LABELS = [
    "Healthy", "Powdery Mildew", "Leaf Spot", "Rust", "Blight",
    "Bacterial Wilt", "Yellow Leaf Curl", "Downy Mildew", "Anthracnose", "Mosaic Virus"
]

# Load trained model
@st.cache_resource
def load_model():
    NUM_CLASSES = len(CLASS_LABELS)
    model = models.vgg16(pretrained=False)  # Adjust based on the model used in training
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)  # Adjust classifier
    model.load_state_dict(torch.load("plant-disease-model.pth", map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

model = load_model()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit app interface
st.markdown(
    "<h1 style='text-align: center; color: green;'>🌿Plant Disease Image Recognition🌿</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='color: blue;'>📤 Upload an Image of a Plant Leaf</h4>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image)
      
        _, predicted = torch.max(outputs, 1)
        disease_name = CLASS_LABELS[predicted.item()]
    
    st.markdown(
    f"""
    <p style='color: #39ff14; font-size: 24px; font-weight: bold; text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14;'>
        🔍 Predicted Disease: {disease_name}
    </p>
    """,
    unsafe_allow_html=True
)







