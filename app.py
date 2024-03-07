import tensorflow as tf 
import streamlit as st
import pandas as pd
import numpy as np
model=tf.keras.models.load_model('Flowers2.keras')

h1="CNN flowers classification"
st.write(h1)
model.summary(print_fn=lambda x: st.text(x))

flowers=pd.DataFrame(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
flowers.columns=['Label']
st.table(flowers)

# sunflower_path = ["./d1.jpg","./d2.jpg","./d3.jpg","./d4.jpg","./t1.jpg","./t2.jpg","./t3.jpg","./s1.jpg"]
img_height = 180
img_width = 180

def pred(img):

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    print("\n")
    score=flowers['Label'][np.argmax(score)]
    print(score)
    print("\n\n\n")
