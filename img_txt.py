import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import imutils
import copy

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from solve_sudoku import solve_sudoku_with_backtrack


# ------------------ PUZZLE DETECTION ------------------ #

def detect_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzle = approx
            break

    warped_color = four_point_transform(image, puzzle.reshape(4, 2))
    warped_gray = four_point_transform(gray, puzzle.reshape(4, 2))
    return warped_color, warped_gray

# ------------------ DIGIT EXTRACTION ------------------ #

def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    if cv2.countNonZero(mask) / float(mask.size) < 0.03:
        return None

    return cv2.bitwise_and(thresh, thresh, mask=mask)

# ------------------ MAIN PIPELINE ------------------ #

def solve_sudoku_from_image(image, model_path):
    model = load_model(model_path)
    image = imutils.resize(image, width=600)

    color, gray = detect_puzzle(image)

    step_x = gray.shape[1] // 9
    step_y = gray.shape[0] // 9

    board = np.zeros((9, 9), dtype="int")
    cells = []

    for y in range(9):
        row = []
        for x in range(9):
            sx, sy = x * step_x, y * step_y
            ex, ey = (x + 1) * step_x, (y + 1) * step_y
            cell = gray[sy:ey, sx:ex]
            digit = extract_digit(cell)

            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                board[y, x] = model.predict(roi, verbose=0).argmax(axis=1)[0]

            row.append((sx, sy, ex, ey))
        cells.append(row)

    solved = solve_sudoku_with_backtrack(copy.deepcopy(board.tolist()))

    for y in range(9):
        for x in range(9):
            if board[y, x] == 0:
                sx, sy, ex, ey = cells[y][x]
                cv2.putText(
                    color, str(solved[y][x]),
                    (sx + 15, ey - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2
                )

    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
