import streamlit as st
import numpy as np
from PIL import Image
import altair as alt
import pandas as pd
import cv2

st.title("Facial Expression Recognition App")
st.write("")
st.write("")
option = st.selectbox(
     'Choose the model you want to use?',
     ('resnet50', 'resnet101', 'densenet121','shufflenet_v2_x0_5','mobilenet_v2'))
""
option2 = st.selectbox(
     'you can select some image',
     ('image_dog', 'image_snake'))

file_up = st.file_uploader("Upload an image", type="jpg")
if file_up is None:
    if option2 =="image_dog":
        image=Image.open("image/dog.jpg")
        file_up="image/dog.jpg"
    else:
        image=Image.open("image/snake.jpg")
        file_up="image/snake.jpg"
