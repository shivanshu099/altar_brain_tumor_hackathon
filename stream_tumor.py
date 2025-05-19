import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



def help_input_pre(img):
    """
    Preprocess the input image for prediction.
    """
    img = img.resize((150, 150))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 150, 150, 1)
    return img


# Load model and label mapping
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

@st.cache_data
def get_label_map():
    return {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

model = load_trained_model('best_model.h5')
label_map = get_label_map()

# App title
st.title("Brain Tumor Classification")
st.write("Upload an MRI image and the app will predict the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose a brain MRI image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded MRI', use_column_width=True)

    # Preprocess image
    img_np=help_input_pre(image)    
    # Prediction
    with st.spinner('Predicting...'):
        probs = model.predict(img_np)[0]
        pred_idx = np.argmax(probs)
        pred_label = label_map[pred_idx]
        confidence = probs[pred_idx]

    # Display results
    st.success(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # Show probability distribution
    st.subheader("Class Probabilities")
    prob_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    st.bar_chart(prob_dict)


#streamlit run stream_tumor.py

    
