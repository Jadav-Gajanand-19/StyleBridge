import streamlit as st
import numpy as np
import joblib

# Load the trained model and encoders
model = joblib.load("fashion_model.pkl")
enc_skin = joblib.load("encoder_SkinTone.pkl")
enc_sex = joblib.load("encoder_Sex.pkl")
enc_region = joblib.load("encoder_Region.pkl")
enc_season = joblib.load("encoder_Season.pkl")
enc_occasion = joblib.load("encoder_Occasion.pkl")
enc_style = joblib.load("encoder_RecommendedStyle.pkl")

st.set_page_config(page_title="AI Fashion Recommender", layout="centered")
st.markdown("## 🛍️ AI-Powered Fashion Recommender")
st.write("Find the perfect style based on your profile and shop across top platforms.")

# Input fields
age = st.slider("👤 Age", 15, 60, 25)
skin_tone = st.selectbox("🎨 Skin Tone", enc_skin.classes_)
sex = st.selectbox("⚧️ Sex", enc_sex.classes_)
region = st.selectbox("🌍 Region", enc_region.classes_)
season = st.selectbox("☀️ Season", enc_season.classes_)
occasion = st.selectbox("🎉 Occasion", enc_occasion.classes_)

# Predict button
if st.button("✨ Recommend My Style"):
    input_vector = np.array([[
        age,
        enc_skin.transform([skin_tone])[0],
        enc_sex.transform([sex])[0],
        enc_region.transform([region])[0],
        enc_season.transform([season])[0],
        enc_occasion.transform([occasion])[0],
    ]])
    pred = model.predict(input_vector)
    style = enc_style.inverse_transform(pred)[0]

    st.success(f"✅ Recommended Style: **{style.upper()}**")

    st.markdown("---")
    st.markdown("### 🛒 Shop this style online:")

    # Build search URLs
    search_term = style.replace(" ", "+")
    urls = {
        "Amazon": f"https://www.amazon.in/s?k={search_term}",
        "Flipkart": f"https://www.flipkart.com/search?q={search_term}",
        "Ajio": f"https://www.ajio.com/search/?text={search_term}",
        "Myntra": f"https://www.myntra.com/{search_term.replace('+', '-')}",
    }

    cols = st.columns(4)
    for idx, (site, url) in enumerate(urls.items()):
        with cols[idx]:
            st.markdown(f"🔗 [{site}]({url})", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by AI • Trained on synthetic fashion data • Inspired by Myntra")
