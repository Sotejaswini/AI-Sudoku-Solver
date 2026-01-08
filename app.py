# app.py
import streamlit as st
import cv2
import numpy as np
from img_txt import process_image, display_solutions_on_image
from solve_sudoku import solve_sudoku_with_backtrack
from keras.models import load_model

st.title("AI Sudoku Solver")

# Load the model once (assuming it's in the model/ folder)
@st.cache_resource
def load_ocr_model():
    return load_model('model/model_1.h5')

model = load_ocr_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Sudoku Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        # Process the image to extract unsolved board, cell positions, and color puzzle
        unsolved_board, cell_positions, color_puzzle = process_image(original_image, model)

        # Solve the Sudoku
        solved_board = solve_sudoku_with_backtrack(copy.deepcopy(unsolved_board))  # Use copy to preserve unsolved

        # Check if it was solved (i.e., changes were made)
        count = sum(1 for i in range(9) for j in range(9) if unsolved_board[i][j] == solved_board[i][j])
        if count == 81:
            st.warning("The input Sudoku appears to be already solved or invalid. Displaying as-is.")
            solved_image_rgb = cv2.cvtColor(color_puzzle, cv2.COLOR_BGR2RGB)
        else:
            # Generate solved image
            solved_image_rgb = display_solutions_on_image(cell_positions, color_puzzle.copy(), unsolved_board, solved_board)
            st.success("Sudoku Solved!")

        # Display images side by side
        with col1:
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        with col2:
            st.image(solved_image_rgb, caption="Solved Sudoku", use_column_width=True)

        # Display solved grid as table
        st.subheader("Solved Grid")
        st.table(solved_board)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}. Ensure the image contains a clear Sudoku grid.")
