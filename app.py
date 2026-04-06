import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
import tempfile
import google.generativeai as genai
from fertilizer_data import fertilizer_data
from tensorflow.keras import mixed_precision
import os

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Crop AI Assistant",
    page_icon="🌱",
    layout="centered"
)

# -------------------------
# UI STYLE
# -------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

body{
background-color:#0f172a;
}

.title{
text-align:center;
font-size:32px;
font-weight:bold;
color:white;
}

.card{
background:#1e293b;
padding:15px;
border-radius:12px;
margin-top:15px;
color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌱 Smart Crop AI Assistant</div>", unsafe_allow_html=True)

st.write("Detect crop diseases and get fertilizer recommendations")

# -------------------------
# LANGUAGE
# -------------------------

language = st.selectbox(
    "🌍 Select Language",
    ["English","Hindi","Telugu"]
)

# -------------------------
# GEMINI API
# -------------------------

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

ai_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_my_model():

    mixed_precision.set_global_policy("mixed_float16")

    model = tf.keras.models.load_model("best_model.keras")

    return model


model = load_my_model()

# -------------------------
# CLASS NAMES
# -------------------------

class_names = [
"Apple___Apple_Scab",
"Apple___Healthy",
"Bell_Pepper___Bacterial_Spot",
"Bell_Pepper___Healthy",
"Cherry___Healthy",
"Cherry___Powdery_Mildew",
"Corn_Maize___Common_Rust_",
"Corn_Maize___Healthy",
"Grape___Black_Rot",
"Grape___Healthy",
"Peach___Bacterial_Spot",
"Peach___Healthy",
"Potato___Healthy",
"Potato___Late_Blight",
"Strawberry___Healthy",
"Strawberry___Leaf_Scorch",
"Tomato___Healthy",
"Tomato___Late_Blight"
]

# -------------------------
# IMAGE PREPROCESS
# -------------------------

def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    return img

# -------------------------
# IMAGE INPUT
# -------------------------

st.subheader("📷 Capture or Upload Crop Leaf")

camera = st.camera_input("Take Photo")

upload = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

image=None

if camera:
    image = Image.open(camera).convert("RGB")

elif upload:
    image = Image.open(upload).convert("RGB")

# -------------------------
# PREDICTION
# -------------------------

if image:

    st.image(image,caption="Leaf Image",use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img,verbose=0)

    pred = np.argmax(prediction)

    result = class_names[pred]

    confidence = np.max(prediction)*100

    crop,disease = result.split("___")

    st.markdown(f"""
    <div class="card">
    <h3>🌿 Crop : {crop}</h3>
    <h3>🦠 Disease : {disease}</h3>
    <h4>🎯 Confidence : {confidence:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# FERTILIZER
# -------------------------

    fert = fertilizer_data.get(result)

    if fert:

        st.markdown(f"""
        <div class="card">
        <h3>🧪 Recommended Fertilizer</h3>
        <b>{fert['fertilizer_name']}</b>
        </div>
        """, unsafe_allow_html=True)

        st.image(fert["image"],use_column_width=True)

        dosage = fert["dosage"][language]

        st.markdown(f"""
        <div class="card">
        <h4>📋 Dosage</h4>
        {dosage}
        </div>
        """, unsafe_allow_html=True)

        precaution = fert["precautions"][language]

        st.markdown(f"""
        <div class="card">
        <h4>⚠ Precautions</h4>
        {precaution}
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# VOICE GUIDE
# -------------------------

        if st.button("🔊 Play Voice Guide"):

            lang_code={"English":"en","Hindi":"hi","Telugu":"te"}[language]

            tts=gTTS(text=dosage,lang=lang_code)

            tmp=tempfile.NamedTemporaryFile(delete=False)

            tts.save(tmp.name)

            st.audio(tmp.name)

# -------------------------
# AI CHATBOT
# -------------------------

st.markdown("---")

st.subheader("🧠 Ask Crop AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

prompt=st.chat_input("Ask about crops, fertilizers or farming...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response=ai_model.generate_content(
        f"You are an agricultural expert helping farmers. Answer simply: {prompt}"
    )

    answer=response.text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})

