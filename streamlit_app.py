import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras

loaded_model = load_model('my_model.h5')

def process_image(image):
    # with keras
    img = keras.preprocessing.image.load_img(image, target_size=(128, 128))
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, [128, 128])

    
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    
    
    return img

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Face Mask Detection')


    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        with st.spinner('Model working....'):
            img_array = process_image(image)
            prediction = loaded_model.predict(img_array).argmax()
            if prediction == 1:
                st.success('The person in the image is wearing a mask')
            else:
                st.error('The person in the image is not wearing a mask')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    # notebook_path = 'object_recognition_model.ipynb'
    # with open(notebook_path, "rb") as file:
    #     btn = st.download_button(
    #         label="Download Jupyter Notebook",
    #         data=file,
    #         file_name="object_recognition_model.ipynb",
    #         mime="application/x-ipynb+json"
    #     )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Face-Mask-Detection-CNN)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is created to demonstrate the face mask detection model using Convolutional Neural Networks (CNN).')
    st.write('The model is trained on a dataset containing images of people with and without masks.')
    st.write('The model is built using TensorFlow and Keras libraries in Python.')
    
    
    
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Face-Mask-Detection-CNN)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
