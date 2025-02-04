import os
import io
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from PIL import Image
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
import base64
import cv2

# =====================================
# APP CONFIGURATION
# =====================================
st.set_page_config(
    page_title="Spitting prevention system",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM STYLES
# =====================================
st.markdown("""
<style>
/* Your custom styles here */
.red-flash {
    background: rgba(255, 0, 0, 0.4);
    animation: red-flash 0.5s ease-out;
}
@keyframes red-flash {
    0% { background: rgba(255, 0, 0, 0.7); }
    100% { background: rgba(255, 0, 0, 0); }
}
</style>
""", unsafe_allow_html=True)

# =====================================
# CUSTOM COMPONENTS
# =====================================
def gradient_text(text):
    return f"""
    <h1 style="
        background: linear-gradient(45deg, #FF4B4B, #FF0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Arial Black';
        text-align: center;
    ">
        {text}
    </h1>
    """

def status_badge(status):
    color = "#00FF00" if status else "#FF0000"
    return f"""
    <div style="
        display: inline-block;
        padding: 5px 15px;
        background: {color};
        color: black;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 0 10px {color};
    ">
        {'🟢 ACTIVE' if status else '🔴 OFFLINE'}
    </div>
    """

# =====================================
# MODEL LOADING
# =====================================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_spitnet_model():
    try:
        model = load_model("keras_model.h5", 
                          compile=False,
                          custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
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

def detect_faces(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

# =====================================
# MAIN APP
# =====================================
def main():
    spitnet_model = load_spitnet_model()
    embedding_model = load_embedding_model()

    if not spitnet_model:
        return

    st.markdown(gradient_text("🛡️ SPITTING PREVENTION SYSTEM"), unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        {status_badge(True)}
        <div style="margin-top: 10px; color: #888;">v1.0 | AI-Powered Spit Detection</div>
    </div>
    """, unsafe_allow_html=True)

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
                    **📞 Phone:** `{details['phone']}`  
                    **📧 Email:** `{details['email']}`  
                    **🏠 Address:** `{details['address']}`
                    """)

def handle_camera_stream(spitnet_model, embedding_model):
    st.markdown("## 📡 Live Monitoring")
    uploaded_image = st.file_uploader("Upload CCTV Snapshot", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("🔍 Analyzing..."):
                try:
                    image = Image.open(uploaded_image).convert('RGB')
                    img_array = np.array(image)
                    
                    # Face detection check
                    if not detect_faces(img_array):
                        st.warning("## 👤 No Human Detected")
                        return

                    # Resize image
                    img_resized = Image.fromarray(img_array).resize((224, 224))
                    img_array = np.array(img_resized)
                    
                    # Spit detection
                    face_array = np.expand_dims(img_array, axis=0).astype('float32') / 127.5 - 1
                    prediction = spitnet_model.predict(face_array)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index]

                    if class_index == 0 and confidence > 0.85:
                        st.error("🚨 Spitting Detected!", icon="🚨")
                        st.markdown("<div class='red-flash'></div>", unsafe_allow_html=True)
                        log_alert(embedding_model, img_array)
                    else:
                        st.success("✅ No spitting detected")
                        
                    st.image(image, caption=f"Processed Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

def handle_alert_history():
    st.markdown("## 🗂️ Alert History")
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if not st.session_state.alerts:
        st.info("No alerts triggered")
    else:
        for alert in st.session_state.alerts:
            with st.expander(f"Alert ID: {alert['id']}"):
                st.image(alert['photo'], width=150)
                st.markdown(f"### Detected Person: {alert['person']}")
                st.markdown(f"**Confidence:** `{alert['confidence']:.2f}`")

def log_alert(embedding_model, img_array):
    try:
        img_resized = Image.fromarray(img_array).resize((224, 224))
        face_array = np.expand_dims(img_array, axis=0).astype('float32') / 127.5 - 1
        embedding = embedding_model.predict(face_array).flatten()

        # Find matching employee
        best_match = None
        highest_similarity = 0

        for emp_id, emp_data in st.session_state.employees.items():
            similarity = cosine_similarity([embedding], [emp_data['embedding']])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = emp_id

        if best_match and highest_similarity > 0.7:
            st.session_state.alerts.append({
                "id": len(st.session_state.alerts) + 1,
                "person": st.session_state.employees[best_match]['name'],
                "confidence": highest_similarity,
                "photo": Image.fromarray(img_array)
            })
            st.success(f"🚨 Alert: {st.session_state.employees[best_match]['name']} ({highest_similarity*100:.2f}% confidence)")
        else:
            st.warning("Unknown person detected")
    except Exception as e:
        st.error(f"Alert logging failed: {e}")

# =====================================
# RUN THE APP
# =====================================
if __name__ == "__main__":
    main()
