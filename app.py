import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import pipeline
import random

# Load the object detection model from Hugging Face with GPU support
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    return pipeline("object-detection", model="hustvl/yolos-tiny", device=device)

model = load_model()

# Function to generate random color
def random_color():
    return tuple(random.choices(range(256), k=3)) + (128,)  # RGBA with transparency

# Streamlit application interface
st.title("Object Detection with Labeling and Colorful Masking")
st.write("Upload an image, and the application will detect all objects, display their names, and apply colorful masks.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    # Perform object detection
    detections = model(image)

    # Create a drawing context
    draw = ImageDraw.Draw(image, "RGBA")

    # Use a larger font size for better visibility
    font = ImageFont.truetype("arial.ttf", 20)  # Change "arial.ttf" to any path of a .ttf file if not available

    # Annotate the image with bounding boxes, labels, and masks
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']

        # Generate a random color for the mask
        mask_color = random_color()

        # Draw a semi-transparent mask over the detected object
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], fill=mask_color)

        # Draw the bounding box
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="red", width=3)
        
        # Display the label and confidence score with a larger font size
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((box['xmin'], box['ymin']), text, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
        draw.text((box['xmin'], box['ymin']), text, fill="white", font=font)

    # Display the annotated and masked image
    st.image(image, caption="Image with Detected Objects Labeled and Masked", use_column_width=True)
    st.write("Detection, labeling, and masking complete!")
