# app.py
import streamlit as st
import pandas as pd
import joblib

from lccde_model import LCCDE_Ensemble

# Load the ensemble model (the one you saved)
@st.cache_resource
def load_model():
    return joblib.load("ensemble_ids.pkl")

model = load_model()

st.set_page_config(page_title="LCCDE Intrusion Detection System", layout="wide")
st.title("🚨 LCCDE-Based Intrusion Detection System (Ensemble of 3 Models)")

st.markdown("""
Upload a CSV file containing network traffic features.  
The model will analyze each record and classify it as **Normal** or a type of **Attack**  
using the LCCDE ensemble logic.
""")

# Define class names
CLASS_NAMES = {
    0: "Benign",
    1: "DoS",
    2: "PortScan",
    3: "Web Attack",
    4: "Botnet",
    5: "Infiltration"
}

# File upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(uploaded_df.head())

    if st.button("🔍 Run Detection"):
        with st.spinner("Running Ensemble Detection..."):
            preds = model.predict(uploaded_df)
            uploaded_df["Prediction"] = preds

        st.success("✅ Detection Complete!")

        # --- Display readable results ---
        st.subheader("📋 Classification Results")
        for i, row in uploaded_df.iterrows():
            class_id = int(row["Prediction"])
            class_name = CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
            st.write(f"**Row {i+1}: Class {class_id} → {class_name}**")
            if i >= 9:  # only show first 10
                break

        st.subheader("🧾 Class Summary")
        for cls, count in uploaded_df["Prediction"].value_counts().items():
            st.write(f"Class {cls} ({CLASS_NAMES.get(cls, 'Unknown')}): {count} samples")

        # Download button
        csv = uploaded_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results",
            data=csv,
            file_name="LCCDE_Results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a CSV file to start detection.")
