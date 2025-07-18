import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd

st.title("CASA Web App – 精子密度與移動速度分析")

source = st.sidebar.radio("Video Source", ["Upload Video", "Use Webcam"])

SPERM_REAL_LENGTH_UM = 22  # 精子總長度，單位微米

# 統計資料
results = []

# 追蹤用
prev_gray = None
object_tracker = {}  # 追蹤精子編號
next_object_id = 0

# 背景減除器 (可選，用於降雜訊)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


def create_circular_mask(frame):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    radius = int(min(center) * 0.5)  # 更小的圓形遮罩，只包含中心亮區
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def draw_detection_area(frame, mask):
    overlay = frame.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
    return overlay

def detect_and_analyze(frame, prev_gray, fps, mask):
    global object_tracker, next_object_id

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        return gray, frame, 0, 0, 0, 0, 0

    flow = cv2.calcOpticalFlowFarneback(prev_gray, blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    motion_mask = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    motion_mask = cv2.bitwise_and(motion_mask, mask)

    _, thresh = cv2.threshold(motion_mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_sizes = []
    valid_speeds = []
    sperm_count = 0

    new_tracker = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 800:
            x, y, w, h = cv2.boundingRect(cnt)
            size = max(w, h)
            valid_sizes.append(size)

            avg_flow = np.mean(mag[y:y + h, x:x + w])
            valid_speeds.append(avg_flow)

            center = (x + w // 2, y + h // 2)
            matched_id = None
            for obj_id, data in object_tracker.items():
                prev_center, frames_count, persist_count, lost_frames = data
                if np.linalg.norm(np.array(center) - np.array(prev_center)) < 60:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = next_object_id
                next_object_id += 1

            if matched_id in object_tracker:
                _, prev_frames_count, prev_persist_count, _ = object_tracker[matched_id]
                frames_count = prev_frames_count + 1
                persist_count = prev_persist_count + 1
            else:
                frames_count = 1
                persist_count = 1

            new_tracker[matched_id] = (center, frames_count, persist_count, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            if frames_count >= 5:
                sperm_count += 1

    # 對失蹤物件保留一段時間
    for obj_id, data in object_tracker.items():
        if obj_id not in new_tracker:
            prev_center, frames_count, persist_count, lost_frames = data
            lost_frames += 1
            if lost_frames <= 5:
                new_tracker[obj_id] = (prev_center, frames_count, persist_count, lost_frames)

    object_tracker = new_tracker

    frame_height, frame_width = frame.shape[:2]
    avg_pixel_size = np.mean(valid_sizes) if valid_sizes else 0

    if avg_pixel_size > 0:
        um_per_pixel = SPERM_REAL_LENGTH_UM / avg_pixel_size
        frame_area_mm2 = (frame_width * um_per_pixel) * (frame_height * um_per_pixel) / 1_000_000
        sperm_density = sperm_count / frame_area_mm2
    else:
        sperm_density = 0
        frame_area_mm2 = 0
        um_per_pixel = 0

    avg_speed_um_per_sec = 0
    if valid_speeds:
        avg_speed_px_per_frame = np.mean(valid_speeds)
        optical_flow_correction_factor = 0.6  # 實驗值，可調整
        avg_speed_um_per_sec = avg_speed_px_per_frame * um_per_pixel * fps * optical_flow_correction_factor

    # 統整統計過程（排除全為零的例外）
    if sperm_count == 0 and avg_pixel_size == 0 and frame_area_mm2 == 0 and avg_speed_um_per_sec == 0:
        return gray, frame, 0, 0, 0, 0, 0

    if 'aggregate_densities' not in globals():
        global aggregate_densities, aggregate_speeds
        aggregate_densities = []
        aggregate_speeds = []

    aggregate_densities.append(sperm_density)
    aggregate_speeds.append(avg_speed_um_per_sec)

    return gray, frame, sperm_count, avg_pixel_size, frame_area_mm2, sperm_density, avg_speed_um_per_sec

def get_overall_report():
    if 'aggregate_densities' in globals() and aggregate_densities:
        overall_density = np.mean([d for d in aggregate_densities if d > 0])
    else:
        overall_density = 0

    if 'aggregate_speeds' in globals() and aggregate_speeds:
        overall_speed = np.mean([s for s in aggregate_speeds if s > 0])
    else:
        overall_speed = 0

    return overall_density, overall_speed

def display_overall_report():
    overall_density, overall_speed = get_overall_report()
    st.subheader("Overall Summary Report")
    st.write("**Average Sperm Density:** {:.2f} sperms/mm²".format(overall_density))
    st.write("**Average Sperm Speed:** {:.2f} µm/s".format(overall_speed))

if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload AVI, MP4, or MOV file", type=["avi", "mp4", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        stframe = st.empty()
        stats_placeholder = st.sidebar.empty()

        frame_counter = 0
        prev_gray = None
        mask = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            if mask is None:
                mask = create_circular_mask(frame)

            prev_gray, annotated_frame, sperm_count, avg_pixel_size, frame_area_mm2, sperm_density, avg_speed_um_per_sec = detect_and_analyze(
                frame, prev_gray, fps, mask)

            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            overlay_frame = draw_detection_area(annotated_frame, mask)

            results.append({
                'Frame': frame_counter,
                'Sperm Count': sperm_count,
                'Avg Sperm Size (px)': avg_pixel_size,
                'Frame Area (mm2)': frame_area_mm2,
                'Density (sperms/mm2)': sperm_density,
                'Avg Speed (um/s)': avg_speed_um_per_sec
            })

            stats_placeholder.markdown(f"""
            **Frame:** {frame_counter}  
            **Sperm Count:** {sperm_count}  
            **Average Sperm Size:** {avg_pixel_size:.2f} px  
            **Image Area:** {frame_area_mm2:.2f} mm²  
            **Sperm Density:** {sperm_density:.2f} sperms/mm²  
            **Average Speed:** {avg_speed_um_per_sec:.2f} µm/s  
            **(Reference: 35 µm/s)**
            """)

            display_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_frame, channels="RGB")

        cap.release()

        df = pd.DataFrame(results)

        st.subheader("Analysis Results")
       
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name='casa_analysis_results.csv',
            mime='text/csv',
        )
        display_overall_report()
else:
    st.warning("Webcam mode not implemented in this version.")

