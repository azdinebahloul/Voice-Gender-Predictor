import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Charger les données
data = pd.read_csv(r'./Data/voice.csv')
data = data[['meanfreq', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'meanfun', 'label']]

# Séparer les features et la variable cible
X = data.drop('label', axis=1)
y = data['label']

# Créer le modèle RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rfc.fit(X, y)

st.set_page_config(page_title="Gender Classification App", page_icon="https://www.svgrepo.com/show/37992/male-and-female-symbol.svg")

# Ajouter un titre personnalisé avec style
st.markdown("<h1 style='text-align: center; font-size: 36px; font-family: Arial, sans-serif;'>Gender Recognition by Voice</h1>", unsafe_allow_html=True)

# Ajouter du texte personnalisé avec style
st.markdown("<p style='text-align: justify; font-size: 18px; font-family: Arial, sans-serif;'>Predicting the gender of a voice has never been easier! Our machine learning model uses Random Forest algorithm, which boasts an impressive 99% accuracy rate. Simply adjust the sliders to input voice features, and voila! Get an instant result with just a few clicks. Try it out now! 🤖👨👩🔍🎉</p>", unsafe_allow_html=True)


import streamlit as st
from PIL import Image

image = Image.open(r'../Voice-Gender-Predictor/Picture needed/vecteezy_male-and-female-gender-icon-symbol-vector_7737986.jpg')

st.image(image)
# Créer des curseurs pour chaque variable
meanfreq = st.slider("Meanfreq", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
median = st.slider("Median", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
Q25 = st.slider("Q25", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
Q75 = st.slider("Q75", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
IQR = st.slider("IQR", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
skew = st.slider("Skew", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
kurt = st.slider("Kurt", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
meanfun = st.slider("Meanfun", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

input_data = {'meanfreq': meanfreq, 'median': median, 'Q25': Q25,
              'Q75': Q75, 'IQR': IQR, 'skew': skew, 'kurt': kurt, 'meanfun': meanfun}

input_df = pd.DataFrame([input_data])

prediction = rfc.predict(input_df)


if prediction[0] == 'female':
    st.markdown("<h2 style='font-size:18px; font-family: Arial,sans-serif;'>The gender of this voice is :</h2>", unsafe_allow_html=True)
    st.write(f"<h2 style='color: pink; font-size:18px; font-family: Arial'>{prediction[0]}</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='font-size:18px; font-family: Arial,sans-serif;'>The gender of this voice is :</h2>", unsafe_allow_html=True)
    st.write(f"<h2 style='color: #ADD8E6; font-size:18px; font-family: Arial'>{prediction[0]}</h2>", unsafe_allow_html=True)




st.markdown("<p style='text-align: center; font-size: 10px; font-family: Arial, sans-serif;'>Made by Azdine Bahloul</p>", unsafe_allow_html=True)
