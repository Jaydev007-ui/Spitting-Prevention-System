import os
import io
import queue
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from PIL import Image
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading

# =====================================
# APP CONFIGURATION
# =====================================
st.set_page_config(
    page_title="Spitting Prevention System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =====================================
# MODEL LOADING
# =====================================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_spitnet_model():
    if not os.path.exists("keras_model.h5"):
        st.error("Model file 'keras_model.h5' not found!")
        return None
    try:
        model = load_model("keras_model.h5", 
                          compile=False,
                          custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        if model.input_shape != (None, 224, 224, 3):
            st.error("Model input shape mismatch! Expected (224, 224, 3)")
            return None
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# =====================================
# MAIN APP
# =====================================
class VideoTransformer(VideoProcessorBase):
    def __init__(self, spitnet_model, embedding_model):
        self.spitnet_model = spitnet_model
        self.embedding_model = embedding_model
        self.alerts = []
        self.frame_count = 0  # Frame counter

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image to reduce processing load
        img_resized = cv2.resize(img, (320, 240))  # Resize to 320x240 for faster processing
        img_gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(img_gray_resized, scaleFactor=1.1, minNeighbors=5)

        # Process every nth frame to reduce load
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # Process every 10th frame
            for (x, y, w, h) in faces:
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
                face_roi = img_resized[y:y + h, x:x + w]
                img_face_resized = cv2.resize(face_roi, (224, 224))

                # Spit detection
                face_array = np.expand_dims(img_face_resized, axis=0).astype('float32') / 127.5 - 1
                prediction = self.spitnet_model.predict(face_array)
                class_index = np.argmax(prediction)
                confidence = prediction[0][class_index]

                # Debugging output
                st.write(f"Class Index: {class_index}, Confidence: {confidence}")  # Debugging output

                spitting_detected = class_index == 0 and confidence > 0.5  # Lowered threshold for testing

                if spitting_detected:
                    self.handle_spitting_alert(face_array, img_resized)

        return av.VideoFrame.from_ndarray(img_resized, format="bgr24")

    def handle_spitting_alert(self, face_array, img_array):
        current_embedding = self.embedding_model.predict(face_array).flatten()
        max_sim = 0
        matched_emp = None
        
        for emp_id, emp in st.session_state.employees.items():
            similarity = cosine_similarity([current_embedding], [emp['embedding']])[0][0]
            if similarity > max_sim:
                max_sim = similarity
                matched_emp = emp_id
        
        alert_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": img_array,
            "max_sim": max_sim,
            "matched_emp": matched_emp
        }
        self.alerts.append(alert_data)

def main():
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

def handle_employee_management(embedding_model):
    st.markdown("## 👥 Employee Management")
    
    with st.form("employee_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📝 Employee Details")
            name = st.text_input("Full Name", placeholder="John Doe")
            phone = st.text_input("Phone Number", placeholder="+91 9876543210")
            email = st.text_input("Email Address", placeholder="john@company.com")
            address = st.text_area("Residential Address", placeholder="123 Main St, City")
        with col2:
            st.subheader("📸 Photo Upload")
            photo = st.file_uploader("Upload employee photo", type=["jpg", "jpeg", "png"])
            if photo:
                image = Image.open(photo)
                st.image(image, caption="Employee Photo", use_column_width=True)
        
        if st.form_submit_button("➕ Add Employee"):
            if not all([name, phone, email, address, photo]):
                st.error("All fields are required!")
            else:
                try:
                    img = Image.open(photo).convert('RGB')
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized)
                    
                    face_array = np.expand_dims(img_array, axis=0).astype('float32') / 127.5 - 1
                    embedding = embedding_model.predict(face_array).flatten()
                    
                    emp_id = f"EMP{len(st.session_state.employees)+1:03d}"
                    st.session_state.employees[emp_id] = {
                        "name": name,
                        "phone": phone,
                        "email": email,
                        "address": address,
                        "photo": photo.getvalue(),
                        "embedding": embedding
                    }
                    st.success(f"Employee {emp_id} added successfully!")
                except Exception as e:
                    st.error(f"Error processing photo: {e}")

    st.markdown("---")
    st.subheader("📋 Registered Employees")
    if not st.session_state.employees:
        st.info("No employees registered")
    else:
        for emp_id, details in st.session_state.employees.items():
            with st.expander(f"{emp_id} - {details['name']}"):
                col1, col2 = st.columns([1,3])
                with col1:
                    st.image(Image.open(io.BytesIO(details['photo'])), width=150)
                with col2:
                    st.write(f"**Phone:** {details['phone']}")
                    st.write(f"**Email:** {details['email']}")
                    st.write(f"**Address:** {details['address']}")
                    if st.button(f"❌ Remove {emp_id}", key=f"remove_{emp_id}"):
                        del st.session_state.employees[emp_id]
                        st.success(f"Employee {emp_id} removed successfully")
                        st.rerun()

def handle_camera_stream(spitnet_model, embedding_model):
    st.markdown("## 📷 Live Camera Stream")
    
    webrtc_streamer(
        key="spit-prevention-stream",
        video_processor_factory=lambda: VideoTransformer(spitnet_model, embedding_model),
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

def handle_alert_history():
    st.markdown("## 🚨 Alert History")
    if not st.session_state.alerts:
        st.info("No alerts recorded yet.")
        return
    
    for i, alert in enumerate(st.session_state.alerts):
        with st.expander(f"Alert {i+1} - {alert['timestamp']}"):
            col1, col2 = st.columns([1,2])
            with col1:
                st.image(alert['image'], width=200)
            with col2:
                st.write(f"Matched Employee: {alert['matched_emp']}")
                st.write(f"Similarity: {alert['max_sim']:.2f}")

if __name__ == "__main__":
    main()
