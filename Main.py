import os
import time
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mtcnn.mtcnn import MTCNN
from PIL import Image
import subprocess

# Set up the Streamlit page configuration
st.set_page_config(page_title="Spitting Prevention System", page_icon="🛡️")

logo = Image.open("Logo.png")  # Replace with your image path
st.image(logo, use_column_width=True)

# Directory to save detected faces
SAVE_DIR = "Detected Faces"

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Custom DepthwiseConv2D class to ignore 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove the unsupported argument
        super().__init__(*args, **kwargs)

# Load the model with custom objects
model = load_model("Spitting.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the labels
with open("labels.txt", "r") as file:
    class_names = file.readlines()

# Streamlit interface
st.title("Spitting Prevention System By Tech Social Shield")
st.markdown("### Detect and prevent spitting in images with advanced facial recognition technology.")
st.markdown("Upload an image to analyze whether any detected faces are exhibiting spitting behavior.")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Input for GitHub credentials
username = "Jaydev007-ui"
token = "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"
user_email = "jaydevzala07@gmail.com"
user_name = username

def set_git_config(email, name):
    try:
        subprocess.run(["git", "config", "--global", "user.email", email], check=True)
        subprocess.run(["git", "config", "--global", "user.name", name], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to set Git config: {e}")

def push_to_github(filename, username, token):
    try:
        subprocess.run(["git", "add", filename], check=True)
        subprocess.run(["git", "commit", "-m", f"Add detected spitting face: {filename}"], check=True)
        subprocess.run(["git", "push", f"https://{username}:{token}@github.com/{username}/Spitting-Prevention-System.git"], check=True)  # Update with your repo URL
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to save to GitHub: {e}")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image = np.array(Image.open(uploaded_image).convert('RGB'))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.write("Detecting faces in the uploaded image...")

    detector = MTCNN()
    results = detector.detect_faces(image_rgb)

    if results:
        spitting_detected = False
        detection_results = []
        confidence_threshold = 0.5  # Set a confidence threshold

        # Set Git user configuration
        if user_email and user_name:
            set_git_config(user_email, user_name)

        for result in results:
            x, y, width, height = result['box']
            face = image_rgb[y:y + height, x:x + width]
            face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            face_array = face_array / 255.0  # Normalize the image

            prediction = model.predict(face_array)
            index = np.argmax(prediction)
            class_name = class_names[index].strip().split(' ', 1)[1]
            confidence_score = prediction[0][index]
            detection_results.append((class_name, confidence_score, (x, y, width, height)))

            if class_name.lower() == "spitting" and confidence_score > confidence_threshold:
                spitting_detected = True
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

        image_rgb_cropped = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        if spitting_detected:
            st.image(image_rgb_cropped, caption="Detected Faces", use_column_width=True)
            highest_confidence_result = max(detection_results, key=lambda x: x[1])
            class_name, confidence_score, (x, y, width, height) = highest_confidence_result

            st.markdown("### Detection Result:")
            st.write(f"- **Class**: {class_name}, **Confidence**: {np.round(confidence_score * 100, 2)}%")

            for result in detection_results:
                detected_class_name, detected_confidence_score, (x, y, width, height) = result
                
                if detected_class_name.lower() == "spitting" and detected_confidence_score > confidence_threshold:
                    spitting_face = image_rgb[y:y + height, x:x + width]
                    face_filename = f"{SAVE_DIR}/spitting_face_{int(time.time())}.jpg"
                    cv2.imwrite(face_filename, spitting_face)

                    # Push the saved image to GitHub
                    if username and token:  # Ensure credentials are provided
                        push_to_github(face_filename, username, token)

            st.markdown("<h3 style='color: red;'>Alert!</h3>", unsafe_allow_html=True)
            st.success("Spitting detected in the image! Detected faces have been saved.")
        else:
            st.success("No spitting detected. Faces detected, but none exhibiting spitting behavior.")
    else:
        st.warning("No faces detected in the uploaded image.")

# Footer information
st.markdown("---")
st.markdown("### Development Phase")
st.markdown("This application is still in development. Your feedback is appreciated!")
st.markdown("**Contact Developer:** [Jaydev Zala](mailto:jaydevzala07@gmail.com)  \n**GitHub:** [Jaydev007-ui](https://github.com/Jaydev007-ui)")
