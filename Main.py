import tempfile
import os
import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import math
import requests
from matplotlib import pyplot as plt
import subprocess

# Set up the Streamlit page configuration
st.set_page_config(page_title="Face Pose Detection System", page_icon="🛡️")

logo = Image.open(os.path.join(os.getcwd(), "Logo.png"))  # Ensure the correct path
st.image(logo, use_column_width=True)

# Directory to save detected faces
SAVE_DIR = "Detected Faces"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Streamlit interface
st.title("Face Pose Detection System By Tech Social Shield")
st.markdown("### Detect face poses in live streams with advanced facial recognition technology.")
st.markdown("Enter the IP address of the live stream to analyze detected faces' poses.")

# Input for live stream IP address
ip_address = st.text_input("Enter Live Stream IP Address (e.g., rtsp://192.168.1.100:8080)")

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

# Helper functions
def npAngle(a, b, c):
    ba = a - b
    bc = c - b 
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def visualizeCV2(frame, landmarks_, angle_R_, angle_L_, pred_):
    lineColor = (255, 255, 0)
    fontScale = 2
    fontThickness = 3
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        color = (0, 0, 0) if pred == 'Frontal' else (255, 0, 0) if pred == 'Right Profile' else (0, 0, 255)
        for land in landmarks:
            cv2.circle(frame, (int(land[0]), int(land[1])), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        cv2.putText(frame, pred, (int(landmarks[0][0]), int(landmarks[0][1])), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)

def predFacePose(frame):
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True)
    angle_R_List, angle_L_List, predLabelList = [], [], []
    if bbox_ is not None:
        for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
            if prob > 0.9:
                angR = npAngle(landmarks[0], landmarks[1], landmarks[2])
                angL = npAngle(landmarks[1], landmarks[0], landmarks[2])
                angle_R_List.append(angR)
                angle_L_List.append(angL)
                predLabel = 'Frontal' if (35 <= int(angR) <= 57 and 35 <= int(angL) <= 58) else ('Left Profile' if angR < angL else 'Right Profile')
                predLabelList.append(predLabel)
    return bbox_, landmarks_, angle_R_List, angle_L_List, predLabelList

def save_face(frame, bbox, filename):
    x1, y1, x2, y2 = [int(i) for i in bbox]
    face = frame[y1:y2, x1:x2]
    cv2.imwrite(filename, face)

def push_to_github(filename):
    username = st.secrets["github"]["username"]
    token = st.secrets["github"]["token"]
    try:
        subprocess.run(["git", "add", filename], check=True)
        subprocess.run(["git", "commit", "-m", f"Add detected face: {filename}"], check=True)
        subprocess.run(["git", "push", f"https://{username}:{token}@github.com/{username}/Face-Pose-Detection-System.git"], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to save to GitHub: {e}")

# Run the video capture and processing
if ip_address:
    video = cv2.VideoCapture(ip_address)
    stframe = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            st.warning("Failed to retrieve frame from the IP stream. Check the IP address.")
            break
        
        # Detect face pose
        bbox_, landmarks_, angle_R_List, angle_L_List, predLabelList = predFacePose(frame)
        
        # If faces are detected, save them and display
        if bbox_ is not None:
            for i, (bbox, landmarks, angle_R, angle_L, predLabel) in enumerate(zip(bbox_, landmarks_, angle_R_List, angle_L_List, predLabelList)):
                # Save each detected face
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(SAVE_DIR, f"detected_face_{timestamp}_{i}.jpg")
                save_face(frame, bbox, filename)
                
                # Push to GitHub
                push_to_github(filename)

            # Draw and display landmarks and predictions
            visualizeCV2(frame, landmarks_, angle_R_List, angle_L_List, predLabelList)

        # Display frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    video.release()

           



