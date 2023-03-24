import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

@st.cache_data
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./model/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./model/haarcascade_smile.xml')

def detect_faces(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw Rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, faces

def detect_eyes(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return img

def detect_smiles(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def cartonize_image(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Edges
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=3, C=10)
    # Color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    # Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cannize_image(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def main():
    """ Face Detection App """
    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Detection":
        st.subheader("Face Detection")

        img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        
        if img_file is not None:
            our_img = load_image(img_file)
            st.text("Original Image")
            st.image(our_img)
        
        enchance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enchance_type == "Gray-Scale":
            new_img = np.array(our_img.convert('RGB'))
            gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
            st.image(gray)
        
        if enchance_type == "Contrast":
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enchancer = ImageEnhance.Contrast(our_img)
            img_output = enchancer.enhance(c_rate)
            st.image(img_output)
        
        if enchance_type == "Brightness":
            b_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enchancer = ImageEnhance.Brightness(our_img)
            img_output = enchancer.enhance(b_rate)
            st.image(img_output)

        if enchance_type == "Blurring":
            new_img = np.array(our_img.convert("RGB"))
            blur_rate = st.sidebar.slider("Blurring", 0, 10)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (15, 15), blur_rate)
            st.image(blur_img)

        # Face Detection
        task = ["Faces", "Smiles", "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Feature", task)
        if st.button("Pocess"):
            if feature_choice == "Faces":
                result_img, result_faces = detect_faces(our_img)
                st.image(result_img)
                st.success(f"Found {result_faces} faces")
            elif feature_choice == "Smiles":
                result_img = detect_smiles(our_img)
                st.image(result_img)
            elif feature_choice == "Eyes":
                result_img = detect_eyes(our_img)
                st.image(result_img)
            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_img)
                st.image(result_canny)
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_img)
                st.image(result_img)

    elif choice == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()