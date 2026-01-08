import streamlit as st
import numpy as np
import cv2
from img_txt import solve_sudoku_from_image

st.set_page_config(page_title="AI Sudoku Solver", layout="wide")

st.title("🧠 AI Sudoku Solver (OCR + Backtracking)")
st.write("Upload a Sudoku image → Get solved grid instantly")

uploaded = st.file_uploader("Upload Sudoku Image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Input Image")
        st.image(image, channels="BGR")

    with st.spinner("Solving Sudoku..."):
        solved_img = solve_sudoku_from_image(image, "model/model_1.h5")

    with col2:
        st.subheader("✅ Solved Output")
        st.image(solved_img)
