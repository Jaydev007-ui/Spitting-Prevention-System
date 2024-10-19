import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mtcnn.mtcnn import MTCNN
from PIL import Image

st.set_page_config(page_title="Spitting Prevention System", page_icon="🛡️")

# Custom DepthwiseConv2D class to ignore 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove the unsupported argument
        super().__init__(*args, **kwargs)

# Load the model with custom objects
model = load_model("Spitting.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Streamlit interface
st.title("Spitting Prevention System By Tech Social Shield")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the uploaded file to an OpenCV image
    image = np.array(Image.open(uploaded_image).convert('RGB'))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure correct format for OpenCV processing

    # Resize the image to (224, 224) pixels as expected by the model
    image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the image to a numpy array and reshape it to the model's input shape (1, 224, 224, 3)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image to [0, 1] as expected by Teachable Machine
    image_array = image_array / 255.0

    # Make predictions using the model
    prediction = model.predict(image_array)

    # Get the index of the highest predicted class
    index = np.argmax(prediction)

    # Get the corresponding class label from the labels file
    class_name = class_names[index].strip().split(' ', 1)[1]  # Extract the label only

    # Get the confidence score of the prediction
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    st.write(f"*Prediction:* {class_name}")
    st.write(f"*Confidence Score:* {str(np.round(confidence_score * 100, 2))}%")

    # If the predicted class is 'spitting', detect faces
    if class_name.lower() == "spitting":
        st.write("Spitting detected, running face detection...")
        detector = MTCNN()
        
        # Detect faces in the original image (before resizing)
        results = detector.detect_faces(image_rgb)

        # If faces are detected, crop the image around the first detected face
        if results:
            for result in results:
                x, y, width, height = result['box']
                # Draw bounding box on the image
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Convert the image back to RGB for display in Streamlit
            image_rgb_cropped = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            # Display the image with the face bounding box
            st.image(image_rgb_cropped, caption="Detected Face", use_column_width=True)
        else:
            st.write("No faces detected.")
    else:
        st.write("No spitting detected.")
