import numpy as np
import cv2
import gradio as gr

def binarize(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

demo = gr.Interface(binarize, gr.Image(), "image")
demo.launch()