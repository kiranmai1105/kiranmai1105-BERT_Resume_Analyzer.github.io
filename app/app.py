import streamlit as st
import sys
import os

sys.path.append(os.path.abspath("src"))
from bert_model import predict_resume_bert, predict_resume_bert_top3
from ats import compute_ats_score
from pdf_utils import extract_text_from_pdf

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer (BERT)", layout="centered")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🚀 AI Resume Analyzer (BERT Powered)")
st.write("Upload or paste resume and analyze against job description")

# -------------------------------
# Input Selection
# -------------------------------
option = st.radio("Choose Input:", ["Paste Text", "Upload PDF"])

resume_text = ""

# -------------------------------
# Paste Resume
# -------------------------------
if option == "Paste Text":
    resume_text = st.text_area("Paste Resume Text", height=250)

# -------------------------------
# Upload PDF
# -------------------------------
elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF processed successfully!")

# -------------------------------
# Job Description Input
# -------------------------------
jd_text = st.text_area("Paste Job Description", height=200)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze Resume"):

    if resume_text.strip() == "" or jd_text.strip() == "":
        st.warning("⚠️ Please provide both Resume and Job Description")

    else:
        # -------------------------------
        # BERT Prediction
        # -------------------------------
        category, confidence = predict_resume_bert(resume_text)

        st.success(f"🎯 Predicted Role: {category}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        # -------------------------------
        # Top 3 Predictions
        # -------------------------------
        st.subheader("🔍 Top Predictions")
        top_preds = predict_resume_bert_top3(resume_text)

        for label, score in top_preds:
            st.write(f"• {label}: {score:.2f}%")

        # -------------------------------
        # ATS Score
        # -------------------------------
        score, matched_skills, missing_skills = compute_ats_score(resume_text, jd_text)

        st.subheader("📊 ATS Match Score")
        st.progress(int(score))
        st.write(f"Match Score: {score:.2f}%")

        # -------------------------------
        # Match Quality Indicator
        # -------------------------------
        if score > 70:
            st.success("🔥 Strong Match")
        elif score > 40:
            st.warning("⚠️ متوسط match (can improve)")
        else:
            st.error("❌ Low match — improve skills")

        # -------------------------------
        # Skills Section
        # -------------------------------
        st.subheader("✅ Matched Skills")
        if matched_skills:
            st.write(matched_skills)
        else:
            st.write("No matched skills found")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.write(missing_skills)
        else:
            st.write("No missing skills 🎉")

        # -------------------------------
        # Show Extracted Resume Text
        # -------------------------------
        with st.expander("📄 View Extracted Resume Text"):
            st.write(resume_text[:3000])