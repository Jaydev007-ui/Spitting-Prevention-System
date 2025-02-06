import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import os
import pickle


# Load SpitNet model for spitting detection
def load_spitnet_model():
    try:
        model = tf.keras.models.load_model("spitnet_model.h5")
        return model
    except:
        st.error("SpitNet model not found!")
        return None


# Load embedding model for facial recognition
def load_embedding_model():
    try:
        model = tf.keras.models.load_model("embedding_model.h5")
        return model
    except:
        st.error("Embedding model not found!")
        return None


# Preprocess image for embedding model
def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 127.5 - 1
    return img_array


# Video stream transformer class for facial recognition and spitting detection
class VideoTransformer:
    def __init__(self, spitnet_model, embedding_model):
        self.spitnet_model = spitnet_model
        self.embedding_model = embedding_model
        self.embeddings = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_preprocessed = preprocess_image(face)
            embedding = self.embedding_model.predict(face_preprocessed).flatten()

            # Compare with stored embeddings to identify the employee
            matched_employee = None
            min_dist = 100

            for emp_id, emp in st.session_state.employees.items():
                dist = np.linalg.norm(embedding - emp['embedding'])
                if dist < min_dist:
                    min_dist = dist
                    matched_employee = emp

            if matched_employee:
                # Spitting detection
                face_expanded = np.expand_dims(face_preprocessed, axis=0)
                spit_pred = self.spitnet_model.predict(face_expanded)
                spit_status = "No Spitting"

                if spit_pred > 0.5:
                    spit_status = "Spitting Detected"
                    # Save alert and image
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert_data = {
                        'timestamp': timestamp,
                        'image': frame,
                        'max_sim': min_dist,
                        'matched_emp': matched_employee['name']
                    }
                    st.session_state.alerts.append(alert_data)

                # Draw rectangle around face and display status
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{matched_employee['name']}: {spit_status}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame


def handle_employee_management(embedding_model):
    st.subheader("🔑 Manage Employee Records")
    st.markdown("Upload images of employees for facial recognition.")

    uploaded_files = st.file_uploader("Upload Employee Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # Preprocess image for embedding model
            img_resized = cv2.resize(img_array, (224, 224))
            face_array = np.expand_dims(img_resized, axis=0).astype('float32') / 127.5 - 1
            embedding = embedding_model.predict(face_array).flatten()

            # Store employee data in session state
            employee_id = uploaded_file.name.split('.')[0]
            st.session_state.employees[employee_id] = {
                'name': uploaded_file.name,
                'embedding': embedding
            }
        
        st.success("Employee images uploaded successfully!")

    # Display employee list
    if st.session_state.employees:
        st.subheader("🔍 Employee List")
        for emp_id, emp in st.session_state.employees.items():
            st.write(f"Employee Name: {emp['name']}")
            st.write(f"Embedding Length: {len(emp['embedding'])} (Embeddings stored)")
            st.write("---")


def handle_camera_stream(spitnet_model, embedding_model):
    st.subheader("🎥 Camera Stream")

    webrtc_streamer(
        key="spitting-prevention-system",
        video_processor_factory=lambda: VideoTransformer(spitnet_model, embedding_model),
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}),
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"style": {"transform": "rotateY(180deg)"}}  # Rotate video
    )


def handle_alert_history():
    st.subheader("🚨 Alert History")
    
    if not st.session_state.alerts:
        st.write("No alerts generated yet.")
    else:
        for alert in st.session_state.alerts:
            st.write(f"Alert at {alert['timestamp']}")
            st.image(alert['image'], caption=f"Spitting Detected: {alert['max_sim']:.2f} | Matched Employee: {alert['matched_emp']}")
            st.write("---")


def main():
    # The main function is where everything ties together.
    spitnet_model = load_spitnet_model()
    embedding_model = load_embedding_model()

    if not spitnet_model:
        return

    st.markdown("## 🛡️ SPITTING PREVENTION SYSTEM")

    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'employees' not in st.session_state:
        st.session_state.employees = {}
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    # Sidebar Authentication
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/681/681494.png", width=100)
        st.markdown("### 🔐 System Control Panel")
        
        if not st.session_state.logged_in:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("🚪 Login"):
                if username == "JAYDEV" and password == "ZALA":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            return
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.employees = {}
            st.session_state.alerts = []
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🧑‍💼 Employee Management")
        menu = st.radio("Navigation", ["📁 Employee Database", "📷 Camera Stream", "🚨 Alert History"])

    # Main Content
    if menu == "📁 Employee Database":
        handle_employee_management(embedding_model)
    elif menu == "📷 Camera Stream":
        handle_camera_stream(spitnet_model, embedding_model)
    elif menu == "🚨 Alert History":
        handle_alert_history()


# Entry point for running the application
if __name__ == "__main__":
    main()
