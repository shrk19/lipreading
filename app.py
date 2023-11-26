# App Imports
import streamlit as st
import imageio

# Model Imports 
import tensorflow as tf 
from utils import load_data, num_to_char
from model import get_model
import os

st.set_page_config(layout='wide')

st.title('LipReadingAI')
with st.sidebar:
    st.image('Lip.png')
    st.header('Deciphering spoken language from lip movements', divider='rainbow')
    st.info('An iteration of the original Lipnet architecture!', icon='ℹ️')
    
options = os.listdir(os.path.join('data', 's1'))
model = get_model()
print(f'model loaded')
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('This is the converted video in mp4 format.')
        file_path = os.path.join('data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # render video 
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.info('This is the actual annotation', icon='✅')
        st.text(tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8'))
        
    with col2:
        st.info('This is what the model input', icon='📥')
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 
        

        st.info('This is the model output.', icon='📤') 
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('This is the decoded output.', icon='🔍')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

        