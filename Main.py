import os
import time
import tempfile
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mtcnn.mtcnn import MTCNN
from PIL import Image
import sqlite3

# Set up the Streamlit page configuration
st.set_page_config(page_title="Spitting Prevention System", page_icon="🛡️")

# Directory to save detected faces
SAVE_DIR = "Detected_Faces"

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Custom DepthwiseConv2D class to ignore 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the model with custom objects
try:
    model = load_model("keras_model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Load the labels
try:
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
except FileNotFoundError:
    st.error("labels.txt file not found. Please make sure it's in the same directory as this script.")
    st.stop()

# Database setup
conn = sqlite3.connect('employees.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS employees
             (id INTEGER PRIMARY KEY, name TEXT, mobile TEXT, email TEXT, address TEXT, photo BLOB)''')
conn.commit()

# Authentication
def authenticate(username, password):
    return username == "JAYDEV" and password == "ZALA"

# Streamlit interface
st.title("Spitting Prevention System")

# Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>Spitting Prevention System</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
else:
    # Admin Panel
    menu = ["Employee Management", "Video Stream", "Spitting History"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Employee Management":
        st.subheader("Employee Management")
        
        # Add new employee
        with st.form("new_employee"):
            name = st.text_input("Name")
            mobile = st.text_input("Mobile")
            email = st.text_input("Email")
            address = st.text_area("Address")
            photo = st.file_uploader("Photo", type=['jpg', 'png', 'jpeg'])
            submit = st.form_submit_button("Add Employee")

            if submit:
                if photo is not None:
                    photo_bytes = photo.getvalue()
                    c.execute("INSERT INTO employees (name, mobile, email, address, photo) VALUES (?, ?, ?, ?, ?)",
                              (name, mobile, email, address, photo_bytes))
                    conn.commit()
                    st.success("Employee added successfully")
                else:
                    st.error("Please upload a photo")

        # View employees
        st.subheader("Employees")
        c.execute("SELECT * FROM employees")
        employees = c.fetchall()
        for employee in employees:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(employee[5], width=100)
            with col2:
                st.write(f"Name: {employee[1]}")
                st.write(f"Mobile: {employee[2]}")
                st.write(f"Email: {employee[3]}")
                st.write(f"Address: {employee[4]}")

    elif choice == "Video Stream":
        st.subheader("Video Stream")
        video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
        
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            
            vf = cv2.VideoCapture(tfile.name)
            
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Detect faces
                detector = MTCNN()
                results = detector.detect_faces(frame)
                
                for result in results:
                    x, y, width, height = result['box']
                    face = frame[y:y + height, x:x + width]
                    face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                    face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_array = face_array / 255.0
                    
                    # Model prediction
                    prediction = model.predict(face_array)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip().split(' ', 1)[1]
                    confidence_score = prediction[0][index]
                    
                    if class_name.lower() == "spitting" and confidence_score > 0.5:
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        cv2.putText(frame, f"Spitting: {confidence_score:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Save detected face
                        face_filename = f"{SAVE_DIR}/spitting_face_{int(time.time())}.jpg"
                        cv2.imwrite(face_filename, face)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame)
            
            vf.release()
            os.unlink(tfile.name)

    elif choice == "Spitting History":
        st.subheader("Spitting History")
        
        # Display detected spitting faces
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith(".jpg"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(os.path.join(SAVE_DIR, filename), width=200)
                with col2:
                    st.write(f"Detected at: {filename.split('_')[-1].split('.')[0]}")
                    
                    # Face matching
                    detected_face = cv2.imread(os.path.join(SAVE_DIR, filename))
                    c.execute("SELECT * FROM employees")
                    employees = c.fetchall()
                    
                    for employee in employees:
                        employee_face = cv2.imdecode(np.frombuffer(employee[5], np.uint8), cv2.IMREAD_COLOR)
                        
                        # Simple face matching
                        if np.mean(cv2.absdiff(detected_face, cv2.resize(employee_face, detected_face.shape[:2]))) < 50:
                            st.write(f"Matched Employee: {employee[1]}")
                            st.write(f"Mobile: {employee[2]}, Email: {employee[3]}")
                            st.image(employee[5], width=100)
                            break

# Logout button
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# Close the database connection
conn.close()
