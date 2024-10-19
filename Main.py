import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mtcnn.mtcnn import MTCNN
from PIL import Image

st.set_page_config(page_title="Spitting Prevention System By Tech Social Shield", page_icon="🛡️")

# Display custom logo at the top
logo = Image.open("Logo.png")  # Replace with your image path
st.image(logo, use_column_width=True)

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
st.title("Spitting Prevention System")
st.markdown("### Detect and prevent spitting in images with advanced facial recognition technology.")
st.markdown("Upload an image to analyze whether any detected faces are exhibiting spitting behavior.")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the uploaded file to an OpenCV image
    image = np.array(Image.open(uploaded_image).convert('RGB'))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize the face detector
    detector = MTCNN()
    results = detector.detect_faces(image_rgb)

    # Check if any faces are detected
    if results:
        spitting_detected = False
        detection_results = []

        for result in results:
            x, y, width, height = result['box']
            # Crop the face from the image
            face = image_rgb[y:y + height, x:x + width]
            
            # Resize the face to (224, 224) pixels as expected by the model
            face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

            # Convert the face to a numpy array and reshape it
            face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            face_array = face_array / 255.0  # Normalize the image

            # Make predictions using the model
            prediction = model.predict(face_array)

            # Get the index of the highest predicted class
            index = np.argmax(prediction)

            # Get the corresponding class label
            class_name = class_names[index].strip().split(' ', 1)[1]
            confidence_score = prediction[0][index]

            # Collect results for display
            detection_results.append((class_name, confidence_score, (x, y, width, height)))

            # Check if the class is "spitting"
            if class_name.lower() == "spitting":
                spitting_detected = True
                # Draw bounding box on the original image
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Convert the image back to RGB for display in Streamlit
        image_rgb_cropped = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Display the image with the face bounding boxes
        st.image(image_rgb_cropped, caption="Detected Faces", use_column_width=True)

        # Show detection results
        st.markdown("### Detection Results:")
        for class_name, confidence_score, _ in detection_results:
            st.write(f"- **Face**: {class_name}, **Confidence**: {np.round(confidence_score * 100, 2)}%")

        if spitting_detected:
            st.markdown("<h3 style='color: red;'>Alert!</h3><p>Spitting detected in the image.</p>", unsafe_allow_html=True)
        else:
            st.success("No spitting detected.")
    else:
        st.warning("No faces detected in the uploaded image.")
