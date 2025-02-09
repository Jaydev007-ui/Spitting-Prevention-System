import streamlit as st
import requests
import numpy as np
import cv2
import os
import io
import time
import base64
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from PIL import Image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from datetime import datetime
import torch  # Import PyTorch for YOLOv5

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
def load_embedding_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # Load your custom YOLOv5 model
    return model

# =====================================
# MAIN APP
# =====================================
def main():
    embedding_model = load_embedding_model()
    yolo_model = load_yolo_model()

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
        menu = st.radio("Navigation", ["📁 Employee Database", "📸 Upload Image for Detection", "🚨 Alert History"])

    # Main Content
    if menu == "📁 Employee Database":
        handle_employee_management(embedding_model)
    elif menu == "📸 Upload Image for Detection":
        handle_image_upload(yolo_model, embedding_model)
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

def handle_image_upload(yolo_model, embedding_model):
    st.markdown("## 📸 Upload Image for Spitting Detection")
    
    uploaded_image = st.file_uploader("Upload an image for spitting detection", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("🔍 Analyzing..."):
                try:
                    # Load and convert image
                    image = Image.open(uploaded_image).convert('RGB')
                    img_array = np.array(image)
                    
                    # YOLOv5 Detection
                    results = yolo_model(img_array)
                    detections = results.pred[0]
                    
                    # Initialize detection flags
                    spitting_detected = False
                    annotated_image = image.copy()
                    
                    # Process detections
                    if detections is not None and len(detections) > 0:
                        for *box, conf, cls in detections:
                            if conf > 0.5 and int(cls) == 0:  # Class 0 = spitting
                                spitting_detected = True
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = map(int, box)
                                draw = ImageDraw.Draw(annotated_image)
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                                
                                # Add confidence text
                                text = f"Spit: {conf:.2f}"
                                draw.text((x1, y1-20), text, fill="red")

                    # Display results
                    if spitting_detected:
                        st.image(annotated_image, 
                               caption="Spitting Detected - Visual Evidence", 
                               use_column_width=True)
                        handle_spitting_alert(img_array, embedding_model, img_array)
                    else:
                        st.image(image, 
                               caption="No Spitting Detected - All Clear", 
                               use_column_width=True)
                        st.success("""
                        ## ✅ System Verification Complete
                        **Status:** No spitting behavior detected  
                        **Recommendation:** Maintain good public hygiene practices
                        """)

                except Exception as e:
                    st.error(f"🚨 Processing Error: {str(e)}")
                    st.error("Please ensure the uploaded file is a valid image")

        with col2:
            st.markdown("### 🔬 Detection Analysis")
            if spitting_detected:
                st.error("""
                ## 🚨 Behavioral Alert
                **Violation Detected:** Public spitting incident  
                **Action Required:**
                - Immediate sanitation required
                - Employee identification in progress
                - Automated fine processing initiated
                """)
            else:
                st.success("""
                ## 🟢 Hygiene Compliance Verified
                **System Confirmed:** No public health violation  
                **Recommended Actions:**
                - Continue regular sanitation protocols
                - Maintain COVID-safe practices
                - Report any hygiene concerns immediately
                """)

    # Live stream section remains unchanged
    st.markdown("---")
    if st.button("🔴 Visit Live Stream"):
        st.markdown("[Click here to view the live stream](http://192.168.94.30:5000)")

    # Button to redirect to live stream
    st.markdown("---")
    if st.button("🔴 Visit Live Stream"):
        st.markdown("[Click here to view the live stream](http://192.168.94.30:5000)")

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
                <h1>🛡 SPITSHIELD PRO</h1>
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
                <h2>⚖ Violation Particulars</h2>
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
                    
                    # Add button to generate fine letter for each alert
                    if st.button(f"📄 Generate Fine Letter for {emp['name']}", key=alert['emp_id']):
                        html_content = generate_fine_letter(alert)
                        st.download_button(
                            label="Download Fine Letter",
                            data=html_content,
                            file_name=f"fine_letter_{alert['emp_id']}.html",
                            mime="text/html"
                        )
                st.markdown("---")

if __name__ == "__main__":
    main()
