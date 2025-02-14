import streamlit as st
import requests
import numpy as np
import cv2
import os
import io
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from PIL import Image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model

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
                    st.markdown(f"""
                    **📞 Phone:** {details['phone']}  
                    **📧 Email:** {details['email']}  
                    **🏠 Address:** {details['address']}
                    """)

def handle_camera_stream(spitnet_model, embedding_model):
    st.markdown("## 📡 Live Monitoring")
    
    # RTSP URL for Raspberry Pi Camera
    rtsp_url = "http://192.168.94.30:5000/video_feed"
    
    if st.button("Start Stream"):
        if not rtsp_url:
            st.error("Please enter a valid RTSP camera address.")
            return

        st.write("### Video Feed")
        video_placeholder = st.empty()

        # Start capturing the video stream via RTSP
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            st.error("Unable to open video stream.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from stream.")
                break

            # Process frame for face and spitting detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangle around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the frame in Streamlit
            video_placeholder.image(frame, channels="BGR")

        cap.release()

    # Add image upload option for spitting detection
    st.markdown("---")
    st.markdown("## 📸 Upload Image for Spitting Detection")
    
    uploaded_image = st.file_uploader("Upload an image for spitting detection", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("🔍 Analyzing..."):
                try:
                    image = Image.open(uploaded_image).convert('RGB')
                    img_array = np.array(image)
                    
                    img_resized = Image.fromarray(img_array).resize((224, 224))
                    img_array = np.array(img_resized)
                    
                    face_array = np.expand_dims(img_array, axis=0).astype('float32') / 127.5 - 1
                    prediction = spitnet_model.predict(face_array)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index]
                    
                    spitting_detected = class_index == 0 and confidence > 0.5  # Lowered threshold for testing
                    
                    st.image(img_resized, caption="Processed Image", use_column_width=True)
                    
                    if spitting_detected:
                        handle_spitting_alert(face_array, embedding_model, img_array)
                    else:
                        st.success("## ✅ All Clear: No Spitting Detected")

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

def handle_spitting_alert(face_array, embedding_model, img_array):
    st.balloons()
    st.error("## 🚨 RED ALERT: Spitting Detected!")
    
    current_embedding = embedding_model.predict(face_array).flatten()
    max_sim = 0
    matched_emp = None
    
    for emp_id, emp in st.session_state.employees.items():
        similarity = cosine_similarity([current_embedding], [emp['embedding']])[0][0]
        if similarity > max_sim:
            max_sim = similarity
            matched_emp = emp_id
    
    if max_sim > 0.6 and matched_emp:
        emp = st.session_state.employees[matched_emp]
        alert = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "emp_id": matched_emp,
            "details": emp,
            "similarity": max_sim,
            "image": img_array
        }
        st.session_state.alerts.append(alert)
        st.markdown(f"""
        **Identified Employee:** {emp['name']} ({matched_emp})  
        **Confidence:** {max_sim*100:.2f}%
        """)
    else:
        st.warning("No matching employee found")

def handle_alert_history():
    st.markdown("## 🚨 Incident History")
    
    if not st.session_state.alerts:
        st.info("No alerts recorded")
    else:
        for alert in reversed(st.session_state.alerts):
            with st.expander(f"Alert - {alert['timestamp']}", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(alert['image'], caption="Incident Capture", width=300)
                with col2:
                    emp = alert['details']
                    st.markdown(f"""
                    **🆔 Employee ID:** {alert['emp_id']}  
                    **👤 Name:** {emp['name']}  
                    **📞 Phone:** {emp['phone']}  
                    **📧 Email:** {emp['email']}  
                    **🔍 Match Confidence:** {alert['similarity']*100:.2f}%
                    """)
                st.markdown("---")

if __name__ == "__main__":
    main()
