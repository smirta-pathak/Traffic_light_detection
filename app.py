# app.py
import streamlit as st
import tempfile
import imageio
import cv2
import numpy as np
import time
import os

st.set_page_config(page_title="Traffic Light Detection", layout="centered")
st.title("🚦 Traffic Light Detection (Streamlit)")

# --- HSV ranges (your original values) ---
red_low1 = np.array([0, 160, 120]); red_up1 = np.array([3, 255, 255])
red_low2 = np.array([177, 160, 120]); red_up2 = np.array([180, 255, 255])
yellow_low = np.array([20, 100, 100]); yellow_up = np.array([45, 255, 255])
green_low = np.array([50, 100, 100]); green_up = np.array([90, 255, 255])

st.markdown("Upload a short video (mp4/avi). For faster results use a short clip or lower resolution.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if not uploaded_file:
    st.info("Upload a video to run detection.")
    st.stop()

# Save upload to a temporary file
t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
t_in.write(uploaded_file.read())
t_in.flush()
t_in.close()

# output temp path
out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
out_tmp.close()
out_path = out_tmp.name

# Optional: Resize frames to speed up processing (set to None to keep original)
MAX_WIDTH = 720  # change or set to None

progress_text = st.empty()
progress_bar = st.progress(0)

@st.cache_data
def _read_video_props(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, total

fps, total_frames = _read_video_props(t_in.name)
progress_text.text(f"Processing video — 0 / {total_frames} frames")

# Process and write using imageio (ffmpeg)
writer = imageio.get_writer(out_path, fps=float(fps))

cap = cv2.VideoCapture(t_in.name)
frame_count = 0
start_time = time.time()

# detection log (keeps counts similar to your script)
detec_log = {'Red': {'TP': 0, 'FP': 0}, 'Yellow': {'TP': 0, 'FP': 0}, 'Green': {'TP': 0, 'FP': 0}}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # optional resize
    if MAX_WIDTH is not None and frame.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # masks (with small morphological denoise)
    mask_red1 = cv2.inRange(hsv, red_low1, red_up1)
    mask_red2 = cv2.inRange(hsv, red_low2, red_up2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.erode(mask_red, np.ones((5,5), np.uint8), iterations=1)
    mask_red = cv2.dilate(mask_red, np.ones((5,5), np.uint8), iterations=1)

    mask_yellow = cv2.inRange(hsv, yellow_low, yellow_up)
    mask_yellow = cv2.erode(mask_yellow, np.ones((5,5), np.uint8), iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, np.ones((3,3), np.uint8), iterations=1)
    # make yellow exclude red areas (avoid double counting)
    mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_red))

    mask_green = cv2.inRange(hsv, green_low, green_up)
    mask_green = cv2.erode(mask_green, np.ones((5,5), np.uint8), iterations=1)
    mask_green = cv2.dilate(mask_green, np.ones((5,5), np.uint8), iterations=1)
    # make green exclude yellow areas
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_yellow))

    # combined regions to find shapes once
    combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_yellow, mask_green))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 10000:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < 0.5:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # count pixels inside bbox for each mask
        roi_red = cv2.countNonZero(mask_red[y:y+h, x:x+w])
        roi_yellow = cv2.countNonZero(mask_yellow[y:y+h, x:x+w])
        roi_green = cv2.countNonZero(mask_green[y:y+h, x:x+w])

        # choose dominant
        if roi_red > roi_yellow and roi_red > roi_green:
            label = "Red"
        elif roi_yellow > roi_green:
            label = "Yellow"
        else:
            label = "Green"

        # draw rectangle and label on original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        detec_log[label]['TP'] += 1

    # pixel-based state (backup)
    red_pix = cv2.countNonZero(mask_red)
    yellow_pix = cv2.countNonZero(mask_yellow)
    green_pix = cv2.countNonZero(mask_green)
    state = "None"
    if red_pix > 200:
        state = "Red"
    elif green_pix > 100:
        state = "Green"
    elif yellow_pix > 50:
        state = "Yellow"

    cv2.putText(frame, f"Traffic Light: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # write frame as RGB for imageio
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # update UI progress
    if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))
    progress_text.text(f"Processing video — {frame_count} / {total_frames} frames")

# cleanup
cap.release()
writer.close()
end_time = time.time()
elapsed = end_time - start_time

# small accuracy summary
summary = f"Frames processed: {frame_count} — Time: {elapsed:.2f}s — FPS (processing): {frame_count/elapsed:.2f}"
st.success("Processing complete!")
st.text(summary)

# show output video and provide download
st.video(out_path)
with open(out_path, "rb") as f:
    st.download_button("Download processed video", data=f.read(), file_name="processed_output.mp4", mime="video/mp4")
