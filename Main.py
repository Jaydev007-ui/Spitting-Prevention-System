import os
import io
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

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (224, 224))
        
        # Spit detection
        face_array = np.expand_dims(img_resized, axis=0).astype('float32') / 127.5 - 1
        prediction = self.spitnet_model.predict(face_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        
        spitting_detected = class_index == 0 and confidence > 0.7
        
        if spitting_detected:
            # Handle spitting alert
            self.handle_spitting_alert(face_array, img)
        
        # Log the frame processing
        st.write("Frame processed")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def handle_spitting_alert(self, face_array, img_array):
        st.balloons()
        st.error("## 🚨 RED ALERT: Spitting Detected!")
        
        # Face recognition
        current_embedding = self.embedding_model.predict(face_array).flatten()
        max_sim = 0
        matched_emp = None
        
        for emp_id, emp in st.session_state.employees.items():
            similarity = cosine_similarity([current_embedding], [emp['embedding']])[0][0]
            if similarity > max_sim:
                max_sim = similarity
                matched_emp = emp_id
        
        if max_sim > 0.6:
            emp = st.session_state.employees[matched_emp]
            alert = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "emp_id": matched_emp,
                "details": emp,
                "similarity": max_sim,
                "image": img_array
            }
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            st.session_state.alerts.append(alert)
            
            st.markdown(f"""
            **Identified Employee:** {emp['name']} ({matched_emp})  
            **Confidence:** {max_sim*100:.2f}%
            """)
        else:
            st.warning("No matching employee found")

def main():
    spitnet_model = load_spitnet_model()
    embedding_model = load_embedding_model()

    if not spitnet_model:
        return

    st.markdown("## 🛡️ SPITTING PREVENTION SYSTEM")

    # Sidebar Authentication
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/681/681494.png", width=100)
        st.markdown("### 🔐 System Control Panel")
        
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
            
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
    
    if 'employees' not in st.session_state:
        st.session_state.employees = {}
    
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
                    
                    # Generate embedding
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
    
    # Option to use webcam
    use_webcam = st.radio("Select Input Source", ["Upload CCTV Snapshot", "Use Webcam"])
    
    if use_webcam == "Use Webcam":
        st.write("### Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_processor_factory=lambda: VideoTransformer(spitnet_model, embedding_model),
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if webrtc_ctx.video_processor:
            st.write("Webcam is running. Look at the video stream.")
            # Allow the user to stop the webcam manually
            if st.button("Stop Webcam"):
                webrtc_ctx.stop()
                st.write("Webcam has been stopped.")
    else:
        uploaded_image = st.file_uploader("Upload CCTV Snapshot", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            col1, col2 = st.columns(2)
            with col1:
                with st.spinner("🔍 Analyzing..."):
                    try:
                        image = Image.open(uploaded_image).convert('RGB')
                        img_array = np.array(image)
                        
                        # Resize image to 224x224
                        img_resized = Image.fromarray(img_array).resize((224, 224))
                        img_array = np.array(img_resized)
                        
                        # Spit detection
                        face_array = np.expand_dims(img_array, axis=0).astype('float32') / 127.5 - 1
                        prediction = spitnet_model.predict(face_array)
                        class_index = np.argmax(prediction)
                        confidence = prediction[0][class_index]
                        
                        spitting_detected = class_index == 0 and confidence > 0.7
                        
                        # Display results
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
    
    # Face recognition
    current_embedding = embedding_model.predict(face_array).flatten()
    max_sim = 0
    matched_emp = None
    
    for emp_id, emp in st.session_state.employees.items():
        similarity = cosine_similarity([current_embedding], [emp['embedding']])[0][0]
        if similarity > max_sim:
            max_sim = similarity
            matched_emp = emp_id
    
    if max_sim > 0.6:
        emp = st.session_state.employees[matched_emp]
        alert = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "emp_id": matched_emp,
            "details": emp,
            "similarity": max_sim,
            "image": img_array
        }
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        st.session_state.alerts.append(alert)
        
        st.markdown(f"""
        **Identified Employee:** {emp['name']} ({matched_emp})  
        **Confidence:** {max_sim*100:.2f}%
        """)
    else:
        st.warning("No matching employee found")

def handle_alert_history():
    st.markdown("## 🚨 Incident History")
    
    if 'alerts' not in st.session_state or not st.session_state.alerts:
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
