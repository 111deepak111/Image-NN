import tensorflow as tf 
import streamlit as st
import pandas as pd
import numpy as np
import sys
import io as StringIO
from PIL import Image

st.set_page_config(layout="wide",page_title="Flowers CNN model")

flowers=pd.DataFrame(['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'])
flowers.columns=['Label']


img_height = 180
img_width = 180

model=tf.keras.models.load_model('Flowers2.keras')

orig=sys.stdout
outputbuf=StringIO.StringIO()
sys.stdout=outputbuf
model.summary()
sys.stdout=orig
modelDesc=outputbuf.getvalue()

h1="CNN flowers classification"

def pred(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    tabulate(score)

def tabulate(score):
    lst=[]
    s=''
    left,right=st.columns(2)
    for i in range(len(score)):
        s=f"<tr><td style='min-width:100px;text-align:center;'>{flowers['Label'][i]}</td><td style='min-width:200px;text-align:center;'>{score[i]}</td></tr>"
        lst.append(s)
    s="{} with a {:.2f}% confidence.".format(flowers['Label'][np.argmax(score)], 100 * np.max(score))
    with left:
        st.markdown(f"<table style='margin-left:auto;margin-right:auto;margin-bottom:50px;margin-top:25px;color: white;'>{''.join(lst)}</table>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<h3 style='text-align: center;margin:75px auto;'>{s}</h3>",unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: center; color: white;'>{h1}</h1>", unsafe_allow_html=True)

lst=[]
for i in range(len(flowers)):
    s=f"<tr><td style='min-width:100px;text-align:center;'>{i}</td><td style='min-width:200px;text-align:center;'>{flowers['Label'][i]}</td></tr>"
    lst.append(s)
st.markdown(f"<table style='margin-left:auto;margin-right:auto;margin-bottom:50px;margin-top:25px;color: white;'>{''.join(lst)}</table>", unsafe_allow_html=True)

img_file_buffer = st.file_uploader('Upload a jpg image', type='jpg')

if img_file_buffer is not None:
    img = tf.keras.utils.load_img(
    img_file_buffer, target_size=(img_height, img_width)
    )
    pred(img)

st.markdown(f"<h3 style='margin:10px auto;'>Model Description</h3>",unsafe_allow_html=True)

st.text(modelDesc)

# sunflower_path = ["./d1.jpg","./d2.jpg","./d3.jpg","./d4.jpg","./t1.jpg","./t2.jpg","./t3.jpg","./s1.jpg"]
