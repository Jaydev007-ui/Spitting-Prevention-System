import os
import time
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mtcnn.mtcnn import MTCNN
from PIL import Image
from github import Github
from streamlit_oauth import OAuth2Component
from pathlib import Path
import json

# Load configuration
def load_config():
    config_path = Path.home() / '.spitting_prevention_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

config = load_config()

# GitHub OAuth Configuration
client_id = os.environ.get("GITHUB_CLIENT_ID")
client_secret = os.environ.get("GITHUB_CLIENT_SECRET")
redirect_uri = "https://techsocialshield.streamlit.app/"  # Update this with your Streamlit app URL

# Initialize the OAuth2Component
oauth2 = OAuth2Component(
    "github",
    client_id=client_id,
    client_secret=client_secret,
    authorize_endpoint="https://github.com/login/oauth/authorize",
    token_endpoint="https://github.com/login/oauth/access_token",
    refresh_token_endpoint="https://github.com/login/oauth/access_token",
    revoke_token_endpoint=f"https://github.com/settings/connections/applications/{client_id}",
)

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
# model = load_model("Spitting.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
try:
    model = load_model("Spitting.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    raise

# Load the labels
with open("labels.txt", "r") as file:
    class_names = file.readlines()

def push_to_github(filename, access_token, repo_name):
    try:
        g = Github(access_token)
        repo = g.get_user().get_repo(repo_name)
        
        with open(filename, 'rb') as file:
            content = file.read()
        
        repo.create_file(
            f"detected_faces/{os.path.basename(filename)}",
            f"Add detected spitting face: {filename}",
            content
        )
        
        st.success("Successfully pushed to GitHub")
    except Exception as e:
        st.error(f"Failed to save to GitHub: {str(e)}")

def main():
    st.title("Spitting Prevention System By Tech Social Shield")
    st.markdown("### Detect and prevent spitting in images with advanced facial recognition technology.")
    st.markdown("Upload an image to analyze whether any detected faces are exhibiting spitting behavior.")

    # OAuth Flow
    if 'access_token' not in st.session_state:
        result = oauth2.authorize_button("Login with GitHub")
        if result and 'access_token' in result:
            st.session_state.access_token = result['access_token']
            st.experimental_rerun()
    else:
        st.success("Logged in to GitHub")
        
        # Upload an image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

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
                            push_to_github(face_filename, st.session_state.access_token, "Your-Repo-Name")

                    st.markdown("<h3 style='color: red;'>Alert!</h3>", unsafe_allow_html=True)
                    st.success("Spitting detected in the image! Detected faces have been saved and pushed to GitHub.")
                else:
                    st.success("No spitting detected. Faces detected, but none exhibiting spitting behavior.")
            else:
                st.warning("No faces detected in the uploaded image.")

    # Footer information
    st.markdown("---")
    st.markdown("### Development Phase")
    st.markdown("This application is still in development. Your feedback is appreciated!")
    st.markdown("**Contact Developer:** [Jaydev Zala](mailto:jaydevzala07@gmail.com)  \n**GitHub:** [Jaydev007-ui](https://github.com/Jaydev007-ui)")

if __name__ == "__main__":
    main()
