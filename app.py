import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Transport Classification")

#image uploader
file = st.file_uploader("Upload image", type = ['jpg','jpeg','png','svg'])
if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    #prediction
    pred, pred_id, probs= model.predict(img)
    st.success(f'Prediction : {pred}')
    st.info(f'Probality : {probs[pred_id]*100:.1f}%')

#plotting
figure = px.bar(x = probs * 100, y = model.dls.vocab)
st.plotly_chart(figure)
