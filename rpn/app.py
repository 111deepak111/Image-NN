from tensorflow.keras.applications import ResNet50,imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import streamlit as st



def selective_search(image,method='fast'):
  ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
  ss.setBaseImage(image)
  if method=='fast':
    ss.switchToSelectiveSearchFast()
  else:
    ss.switchToSelectiveSearchQuality()

  rects=ss.process()
  return rects

print("[INFO] loading ResNet...")
model = ResNet50(weights="imagenet")

path='/content/download (1).jpg'

st.set_page_config(layout="wide",page_title="Flowers CNN model")
h1="Object detection"

img_file_buffer = st.file_uploader('Upload a jpg image', type='jpg')

method=st.radio("Method",["Fast","Qualtiy"])

if img_file_buffer is not None:
    img = tf.keras.utils.load_img(
    img_file_buffer, target_size=(img_height, img_width)
    )
    pred(img)

# load the input image from disk and grab its dimensions
