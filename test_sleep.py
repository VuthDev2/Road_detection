import cv2
import streamlit as st
import time
import numpy as np

st.title("Test Video")
placeholder = st.empty()
for i in range(100):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    placeholder.image(img, channels="RGB")
    # time.sleep(0.01)
st.success("Done")
