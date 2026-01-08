import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import cv2
import imutils
import copy
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


def detect_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzle_contour = approx
            break

    warped_color = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped_gray = four_point_transform(gray, puzzle_contour.reshape(4, 2))
    return warped_color, warped_gray


def identify_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    if cv2.countNonZero(mask) / float(cell.shape[0] * cell.shape[1]) < 0.03:
        return None

    return cv2.bitwise_and(thresh, thresh, mask=mask)


def process_image(image_path, model_path):
    model = load_model(model_path)

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    color_puzzle, gray_puzzle = detect_puzzle(image)

    board = np.zeros((9, 9), dtype="int")

    step_x = gray_puzzle.shape[1] // 9
    step_y = gray_puzzle.shape[0] // 9

    for y in range(9):
        for x in range(9):
            start_x, start_y = x * step_x, y * step_y
            end_x, end_y = (x + 1) * step_x, (y + 1) * step_y

            cell = gray_puzzle[start_y:end_y, start_x:end_x]
            digit = identify_digit(cell)

            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi, verbose=0).argmax(axis=1)[0]
                board[y][x] = prediction

    with open("inp.txt", "w") as f:
        for row in board:
            f.write(" ".join(map(str, row)) + "\n")

    cv2.imwrite("color_puzzle.jpg", color_puzzle)
