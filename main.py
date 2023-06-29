import streamlit as st
import numpy as np
from PIL import Image
import altair as alt
import pandas as pd
import mmcv
import matplotlib.pyplot as plt
from mmcls.apis.inference import init_model, inference_model

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
st.image(image, caption='Uploaded Image.', use_column_width=True)
st.write("")
st.write("Just a second...")
img = mmcv.imread(file_up)
model = init_model(
    config="/mmcls/configs/apvit/RAF.py",
    checkpoint="/mmcls/APViT_RAF-3eeecf7d.pth"
)
result = inference_model(model, img)
st.success('successful prediction')
st.write(result)
