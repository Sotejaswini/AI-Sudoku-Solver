import streamlit as st
import os
from img_txt import process_image
from solve_sudoku import solve_sudoku_with_backtrack
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(page_title="AI Sudoku Solver", layout="centered")

st.title("🧠 AI Sudoku Solver using OCR")
st.write("Upload a Sudoku image and get the solved grid instantly")

uploaded = st.file_uploader("Upload Sudoku Image", ["jpg", "png", "jpeg"])
MODEL_PATH = "model/model_1.h5"

def read_board(path):
    board = []
    with open(path) as f:
        for line in f:
            board.append(list(map(int, line.split())))
    return board

if uploaded:
    with open("input.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(uploaded, caption="Uploaded Sudoku")

    with st.spinner("Extracting digits..."):
        process_image("input.jpg", MODEL_PATH)

    board = read_board("inp.txt")
    solved = solve_sudoku_with_backtrack(board)

    st.success("Sudoku Solved!")

    st.subheader("Solved Grid")
    for row in solved:
        st.write(row)

    if os.path.exists("color_puzzle.jpg"):
        st.subheader("Solved Image")
        st.image(Image.open("color_puzzle.jpg"))

    for f in ["inp.txt", "input.jpg", "color_puzzle.jpg"]:
        if os.path.exists(f):
            os.remove(f)
