import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import datetime
import time


st.set_page_config(page_title="Traffic Detector System", layout="wide", page_icon="🚦")

st.markdown("""
    <style>
    .title {
        background: linear-gradient(to right, #ff4b4b, #ff7e5f);
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
        font-size: 48px;
        font-weight: 800;
    }
    .subheader {
        text-align: center;
        color: #444;
        font-size: 20px;
        margin-top: -10px;
    }
    .dev-info {
        text-align: center;
        color: #555;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🚦 Traffic Detector System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Department of Computer Science & Engineering | PSIT Kanpur</p>", unsafe_allow_html=True)
st.markdown("<p class='dev-info'><b>Developed By:</b> Kartavya Agrahari, Khushi Kaushal, Khushi Yadav, Janvi Kamal, Gurucharan Singh<br><b>Guide:</b> Mrs. Suman Kuril (Assistant Professor)</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/684/684908.png", width=80)
st.sidebar.title("🧠 Project Information")
st.sidebar.markdown("""
**Title:** Traffic Detector System  
**Domain:** AI | Machine Learning | Computer Vision  
**Language:** Python (OpenCV + Streamlit)  

---
📍 **Team Members:**
- Kartavya Agrahari  
- Khushi Kaushal  
- Khushi Yadav  
- Janvi Kamal  
- Gurucharan Singh  

👩‍🏫 **Guide:** Mrs. Suman Kuril  
🏫 **Institute:** PSIT Kanpur  
""")

st.sidebar.info("Upload your traffic video below and see real-time detection results 👇")


uploaded_video = st.file_uploader("📤 Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("### ⏳ Processing Video... Please Wait")

    progress_bar = st.progress(0)
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    vehicle_count = 0
    detections = []
    line_y = 450
    offset = 6
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        roi = frame[200:600, 100:800]

        mask = fgbg.apply(roi)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w < 40 or h < 40:
                continue
            center = (x + w // 2, y + h // 2)
            detections.append(center)

        for (x, y) in detections:
            if (line_y - offset) < (y + 200) < (line_y + offset):
                vehicle_count += 1
                detections.remove((x, y))

        progress = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
        progress_bar.progress(min(progress, 100))

    cap.release()

    if vehicle_count < 10:
        congestion = "LOW"
        color = "green"
    elif vehicle_count < 25:
        congestion = "MEDIUM"
        color = "orange"
    else:
        congestion = "HIGH"
        color = "red"

    st.success("✅ Video Processed Successfully!")

    st.markdown(f"""
    <div style='text-align:center;'>
        <h3>🚗 Total Vehicles Detected: <span style='color:#00b300;'>{vehicle_count}</span></h3>
        <h3>🚦 Congestion Level: <span style='color:{color};'>{congestion}</span></h3>
    </div>
    """, unsafe_allow_html=True)

  
    df = pd.DataFrame({
        "Date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Total Vehicles": [vehicle_count],
        "Congestion Level": [congestion]
    })

    st.subheader("📊 Summary Report")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Summary as CSV",
        data=csv,
        file_name="traffic_summary.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Please upload a video file to start detection.")



uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("### Processing video... please wait ⏳")

    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    vehicle_count = 0
    detections = []
    line_y = 450
    offset = 6

    frames_processed = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        roi = frame[200:600, 100:800]

        mask = fgbg.apply(roi)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w < 40 or h < 40:
                continue
            center = (x + w // 2, y + h // 2)
            detections.append(center)

        for (x, y) in detections:
            if (line_y - offset) < (y + 200) < (line_y + offset):
                vehicle_count += 1
                detections.remove((x, y))

        frames_processed += 1

    cap.release()

    if vehicle_count < 10:
        congestion = "LOW"
        color = "green"
    elif vehicle_count < 25:
        congestion = "MEDIUM"
        color = "orange"
    else:
        congestion = "HIGH"
        color = "red"

    st.success("✅ Processing complete!")

    st.markdown(f"""
    ### Results:
    - **Total Vehicles Detected:** `{vehicle_count}`
    - **Traffic Congestion Level:** <span style='color:{color}; font-weight:bold;'>{congestion}</span>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({"Vehicles": [vehicle_count], "Congestion": [congestion]})
    st.dataframe(df)
else:
    st.info("Please upload a traffic video file to start detection.")
