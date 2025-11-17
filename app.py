#!/usr/bin/env python
# coding: utf-8

# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # load model files
    prototxt = r"C:\Users\asus\OneDrive\Desktop\Machine learning\BLACK_WHITE_IMAGE-OPENCV\models\models_colorization_deploy_v2.prototxt"
    model = r"C:\Users\asus\OneDrive\Desktop\Machine learning\BLACK_WHITE_IMAGE-OPENCV\models\colorization_release_v2.caffemodel"
    points = r"C:\Users\asus\OneDrive\Desktop\Machine learning\BLACK_WHITE_IMAGE-OPENCV\models\pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # add cluster centers as 1x1 convolutions
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # scale image
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # resize & extract L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # run through network
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize ab to original image size
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # combine original L with predicted ab
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # LAB → RGB
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    # scale to uint8
    colorized = (255 * colorized).astype("uint8")
    return colorized


##########################################################################################################

st.write("# Colorizing Black & White image")
st.write("This is an app to turn Colorize your B&W images.")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.text("Your original image")
        st.image(image, use_container_width=True)  

    with col2:
        st.text("Your colorized image")
        color = colorizer(img)
        st.image(color, use_container_width=True)   

    print("done!")
