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
                    **📞 Phone:** {details['phone']}  
                    **📧 Email:** {details['email']}  
                    **🏠 Address:** {details['address']}
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
                    
                    spitting_detected = class_index == 0 and confidence > 0.7
                    
                    st.image(img_resized, caption="Processed Image", use_column_width=True)
                    
                    if spitting_detected:
                        handle_spitting_alert(face_array, embedding_model, img_array)
                    else:
                        st.success("## ✅ All Clear: No Spitting Detected")

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

def handle_spitting_alert(face_array, embedding_model, img_array):
    # Red flash effect
    st.markdown("""
    <div class="red-flash" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;">
    </div>
    """, unsafe_allow_html=True)
    
    st.error("## 🚨 RED ALERT: Spitting Detected!")

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
                # Add this function after the handle_alert_history function
import base64
import io
from PIL import Image
import streamlit as st

# Function to generate the fine letter HTML
def generate_fine_letter(alert):
    emp = alert['details']
    
    # Convert employee photo to base64
    employee_image = Image.open(io.BytesIO(emp['photo']))
    buffered_employee = io.BytesIO()
    employee_image.save(buffered_employee, format="PNG")
    employee_img_str = base64.b64encode(buffered_employee.getvalue()).decode()

    # Convert spitting incident photo to base64
    incident_image = Image.fromarray(alert['image'])
    buffered_incident = io.BytesIO()
    incident_image.save(buffered_incident, format="PNG")
    incident_img_str = base64.b64encode(buffered_incident.getvalue()).decode()
    
    # Get current date and time
    from datetime import datetime
    dt = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
    
    # HTML template with improved styling
    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: 'Arial', sans-serif; }}
        .letter-container {{
            border: 3px solid #e74c3c;
            border-radius: 15px;
            padding: 30px;
            max-width: 800px;
            margin: 20px auto;
            background: #f9f9f9;
        }}
        .header {{
            text-align: center;
            color: #e74c3c;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .logo {{
            width: 180px;
            margin-bottom: 15px;
        }}
        .section {{
            margin: 25px 0;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .signature-box {{
            margin-top: 40px;
            text-align: right;
            padding: 20px;
            border-top: 2px dashed #e74c3c;
        }}
        .fine-amount {{
            color: #e74c3c;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin: 25px 0;
        }}
        .employee-photo {{
            border: 2px solid #e74c3c;
            border-radius: 8px;
            margin: 15px 0;
        }}
    </style>
    </head>
    <body>
        <div class="letter-container">
            <div class="header">
                <h1>🛡️ SPITSHIELD PRO</h1>
                <h3>Public Health Violation Notice</h3>
            </div>
            
            <div class="section">
                <h2>📅 Violation Details</h2>
                <p><strong>Date:</strong> {dt.strftime('%d %B %Y')}</p>
                <p><strong>Time:</strong> {dt.strftime('%I:%M %p')}</p>
                <p><strong>Location:</strong> Main Office Premises</p>
            </div>

            <div class="section">
                <h2>👤 Offender Information</h2>
                <img src="data:image/png;base64,{employee_img_str}" class="employee-photo" width="150">
                <p><strong>Name:</strong> {emp['name']}</p>
                <p><strong>Employee ID:</strong> {alert['emp_id']}</p>
                <p><strong>Contact:</strong> {emp['phone']}</p>
            </div>

            <div class="fine-amount">
                ₹500 FINE IMPOSED
            </div>

            <div class="section">
                <h2>⚖️ Violation Particulars</h2>
                <p>Violation Code: SS-102</p>
                <p>Article 15 of Public Health & Safety Act, 2018</p>
                <p>Match Confidence: {alert['similarity']*100:.2f}%</p>
                
                <!-- Added incident proof section -->
                <div style="margin-top: 20px;">
                    <img src="data:image/png;base64,{incident_img_str}" 
                         class="incident-proof"
                         alt="Spitting Incident Proof">
                    <p class="proof-caption">Spitting Incident Visual Proof</p>
                </div>
            </div>

            <div class="signature-box">
                <p>Authorized Signatory:</p>
                <img src="https://cdn-icons-png.flaticon.com/512/1496/1496034.png" width="120">
                <p>SpitShield Pro Enforcement Unit</p>
                <p>Date: {datetime.now().strftime('%d %B %Y')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# Update the handle_alert_history function
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
                    
                    # Add download button for fine letter
                    st.download_button(
                        label="📄 Download Fine Notice",
                        data=generate_fine_letter(alert).encode('utf-8'),
                        file_name=f"fine_notice_{alert['emp_id']}_{alert['timestamp']}.html",
                        mime="text/html",
                        help="Download official fine notice in HTML format"
                    )
                st.markdown("---")


if __name__ == "__main__":
    main()

