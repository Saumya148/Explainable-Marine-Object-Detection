import streamlit as st
import subprocess
import os
import time
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Marine Object Detection",
    layout="wide"
)

# ---------------------------------
# Session State
# ---------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------------------------
# Title
# ---------------------------------

st.title("🌊 Marine Object Detection System")

st.write("Upload an underwater video and generate the detection report.")

# ---------------------------------
# Upload Video
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

# ---------------------------------
# Save Uploaded Video
# ---------------------------------

if uploaded_file is not None:

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded successfully.")

    if st.button("▶ Run Detection"):

        with st.spinner("Processing... Please wait..."):

            subprocess.run(
                [sys.executable, "video.py", uploaded_file.name],
                check=True
            )

        for _ in range(20):

            if (
                os.path.exists("output.mp4")
                and os.path.exists("report.txt")
                and os.path.exists("metrics.json")
                and os.path.exists("counts.csv")
            ):
                break

            time.sleep(1)

        st.session_state.processed = True

        st.success("Detection Completed Successfully!")

# ---------------------------------
# Show Results
# ---------------------------------

if st.session_state.processed:

    # ==========================================
    # Dashboard Metrics
    # ==========================================

    if os.path.exists("metrics.json"):

        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Frames",
            metrics["frames"]
        )

        c2.metric(
            "Objects",
            metrics["total_objects"]
        )

        c3.metric(
            "Time (sec)",
            metrics["processing_time"]
        )

    # ==========================================
    # Side by Side Videos
    # ==========================================

    st.subheader("🎥 Original vs Processed Video")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Original Video")

        with open(uploaded_file.name, "rb") as f:
            st.video(f.read())

    with col2:

        st.write("### Processed Video")

        with open("output.mp4", "rb") as f:

            processed_video = f.read()

        st.video(processed_video)

    st.download_button(
        label="⬇ Download Processed Video",
        data=processed_video,
        file_name="marine_detection.mp4",
        mime="video/mp4"
    )

    # ==========================================
    # Object Count Table
    # ==========================================

    if os.path.exists("counts.csv"):

        st.subheader("📊 Object Count Table")

        df = pd.read_csv("counts.csv")

        st.dataframe(df)

    # ==========================================
    # Pie Chart
    # ==========================================

    if os.path.exists("counts.csv"):

        st.subheader("🥧 Class Distribution")

        df = pd.read_csv("counts.csv")

        df = df[df["Count"] > 0]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            df["Count"],
            labels=df["Class"],
            autopct="%1.1f%%"
        )

        ax.set_title("Detected Objects")

        st.pyplot(fig)

    # ==========================================
    # Report
    # ==========================================

    if os.path.exists("report.txt"):

        with open("report.txt", "r") as f:

            report = f.read()

        st.subheader("📄 Detection Report")

        st.text(report)

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name="marine_report.txt",
            mime="text/plain"
        )
        # ---------------------------------
# Download PDF Report
# ---------------------------------

if os.path.exists("marine_report.pdf"):

    with open("marine_report.pdf", "rb") as pdf_file:

        pdf_bytes = pdf_file.read()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="marine_report.pdf",
        mime="application/pdf"
    )