import sys
input=sys.argv[1]
print(input)

import tensorflow as tf

model = tf.keras.models.load_model('my_model4.hdf5',compile=False)


import streamlit as st
import nest_asyncio
import cv2
from PIL import Image, ImageOps
import numpy as np


st.write("""
         # Skin disease detection
         """
         )
st.write("This is a simple image classification web app to predict skin disease")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(229, 229),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is CP!")
    elif np.argmax(prediction) == 1:
        st.write("It is HFMD!")
    elif np.argmax(prediction) == 2:
        st.write("It is HR!")
    else:
        st.write("It is HZ!")
    
    st.text("Probability (0: CP, 1: HFMD, 2: HR, 3: HZ")
    st.write(prediction)
