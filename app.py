import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import tempfile
import os
import sys
import importlib.util
import datetime
from PIL import Image

# Set Streamlit Page Config
st.set_page_config(
    page_title="Multi-System CV Perception Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    
    .title-text {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    .sidebar-header {
        font-family: 'Orbitron', sans-serif;
        color: #00f2fe;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00f2fe;
        padding-bottom: 5px;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to dynamically import scripts with spaces in names
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Show spinner while importing modules and loading models
with st.spinner("Initializing CV Modules and checking files..."):
    try:
        air_combat = load_module("air_combat", "Air Combat Intelligence System.py")
        license_plate = load_module("license_plate", "License Plate Intelligence System.py")
        airport_runway = load_module("airport_runway", "Airport Runway.py")
    except Exception as e:
        st.error(f"Error loading system modules: {e}")

st.markdown("<h1 class='title-text'>📡 Multi-System CV Perception Hub</h1>", unsafe_allow_html=True)

# Initialize Session States
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'engine_running' not in st.session_state:
    st.session_state.engine_running = False

# System Selection in Sidebar
st.sidebar.markdown("<div class='sidebar-header'>System Selection</div>", unsafe_allow_html=True)
selected_system = st.sidebar.selectbox(
    "Select Perception Engine:",
    [
        "Air Combat Intelligence System",
        "License Plate Intelligence System",
        "Airport Runway Intelligence System"
    ]
)

# Video Source Selection
st.sidebar.markdown("<div class='sidebar-header'>Input Source</div>", unsafe_allow_html=True)
input_type = st.sidebar.radio("Select Source Type:", ["Upload Video File", "Live Camera"])

uploaded_file = None
camera_index = 0

if input_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Video (MP4, AVI, MOV):", type=["mp4", "avi", "mov", "mkv"])
else:
    camera_index = st.sidebar.number_input("Camera Index:", min_value=0, value=0, step=1)

# Advanced Configuration
st.sidebar.markdown("<div class='sidebar-header'>Parameters</div>", unsafe_allow_html=True)
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold:", 0.1, 1.0, 0.25, 0.05)
detect_every = st.sidebar.slider("Detection Interval (Frames):", 1, 10, 2, 1)

# Status & Launch control
st.sidebar.markdown("<div class='sidebar-header'>Control Panel</div>", unsafe_allow_html=True)
start_button = st.sidebar.button("🚀 Start Perception Engine", use_container_width=True)
stop_button = st.sidebar.button("⏹ Stop Engine", use_container_width=True)

if stop_button:
    st.session_state.stop_requested = True
    st.session_state.engine_running = False

# Layout Tabs (Always rendered)
tab1, tab2, tab3 = st.tabs(["📺 Live Feed", "📊 Real-Time Analytics", "📁 Reports & Exports"])

with tab1:
    st.subheader("Live Perception Output")
    # Live rendering placeholders
    video_placeholder = st.empty()
    status_text = st.empty()
    
    # Live KPI metrics
    m1, m2, m3, m4 = st.columns(4)
    met1 = m1.empty()
    met2 = m2.empty()
    met3 = m3.empty()
    met4 = m4.empty()

    if not st.session_state.engine_running:
        video_placeholder.markdown("""
        <div class="info-card">
            <h3>Ready to Start</h3>
            <p>👈 Select your configuration in the sidebar and click <b>Start Perception Engine</b> to process and stream the video feed here.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Perception Analytics")
    chart_placeholder = st.empty()
    if st.session_state.csv_data is not None:
        df = st.session_state.csv_data
        if not df.empty:
            chart_placeholder.line_chart(df.drop(columns=["timestamp"], errors="ignore"))
    else:
        chart_placeholder.markdown("""
        <div class="info-card">
            <h3>No Analytics Data Available</h3>
            <p>Data charts will populate automatically once the perception engine starts processing video frames.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("Generated Reports & Output Videos")
    export_placeholder = st.empty()
    
    if st.session_state.csv_data is not None or st.session_state.output_video_path is not None:
        with export_placeholder.container():
            if st.session_state.csv_data is not None:
                csv_buffer = st.session_state.csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Activity Report",
                    data=csv_buffer,
                    file_name="perception_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            if st.session_state.output_video_path is not None and os.path.exists(st.session_state.output_video_path):
                with open(st.session_state.output_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video File",
                        data=f,
                        file_name="processed_feed.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
    else:
        export_placeholder.markdown("""
        <div class="info-card">
            <h3>No Reports Generated</h3>
            <p>Once video processing is complete, you will be able to download the processed video file and a structured CSV log from this tab.</p>
        </div>
        """, unsafe_allow_html=True)

def get_temp_output_path(suffix=".mp4"):
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, f"cv_output_{int(time.time())}{suffix}")

# Run chosen system
if start_button:
    st.session_state.stop_requested = False
    st.session_state.engine_running = True
    
    # Establish source path
    source_path = None
    if input_type == "Upload Video File":
        if uploaded_file is None:
            st.warning("Please upload a video file first.")
            st.stop()
        # Save uploaded file to temp file
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        temp_in.write(uploaded_file.read())
        source_path = temp_in.name
        temp_in.close()
    else:
        source_path = str(camera_index)

    # Initialize capturing
    cap = cv2.VideoCapture(int(source_path) if source_path.isdigit() else source_path)
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        st.session_state.engine_running = False
        st.stop()
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # Setup video writer
    out_video_path = get_temp_output_path()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (w, h))

    # Reset placeholder for video
    video_placeholder.empty()
    chart_placeholder.empty()
    export_placeholder.empty()
        
    frame_num = 0
    t_prev = time.perf_counter()
    csv_rows = []
    
    # Instantiate models / engines
    try:
        from ultralytics import YOLO
        if selected_system == "License Plate Intelligence System":
            plate_model_path = license_plate.ensure_plate_model()
            model = YOLO("yolov8n.pt")  # vehicle model
            plate_model = YOLO(plate_model_path)
            ocr = license_plate.OCREngine()
            tracker = license_plate.VehicleTracker()
            plate_records = []
            plate_counts = license_plate.defaultdict(int)
        
        elif selected_system == "Air Combat Intelligence System":
            model = YOLO("yolov8s.pt")
            tracker = air_combat.Tracker()
            heatmap_pts = []
            peak_threat = 0.0
            
        else:
            model = YOLO("yolov8s.pt")
            tracker = airport_runway.Tracker()
            twin = airport_runway.DigitalTwin()
            heatmap_bank = airport_runway.HeatmapBank(w, h)
            restricted_boxes = [
                ((int(w*0.1), int(h*0.3)), (int(w*0.4), int(h*0.6)), "RUNWAY 09L-27R"),
                ((int(w*0.6), int(h*0.4)), (int(w*0.95), int(h*0.85)), "TAXIWAY ALPHA"),
            ]
            zones_minimap = {
                "RUNWAY 09L": (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.6), (0,0,150)),
                "TAXIWAY ALPHA": (int(w*0.6), int(h*0.4), int(w*0.95), int(h*0.85), (0,100,100)),
            }

    except Exception as model_err:
        st.error(f"Error initializing models/modules: {model_err}")
        st.session_state.engine_running = False
        st.stop()

    chart_data = []
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    last_dets = []
    
    while cap.isOpened():
        if st.session_state.stop_requested:
            st.warning("Perception Engine stopped by user.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        t_now = time.perf_counter()
        fps_disp = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        
        # 1. License Plate System
        if selected_system == "License Plate Intelligence System":
            if frame_num % detect_every == 0 or frame_num == 1:
                res = model(frame, verbose=False, conf=conf_threshold)[0]
                dets = []
                for box in res.boxes:
                    c = int(box.cls[0])
                    if c in [2, 3, 5, 7]:
                        xy = box.xyxy[0].cpu().numpy()
                        dets.append(( (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])), "vehicle", float(box.conf[0])))
                last_dets = dets
                
            tracks = tracker.update(last_dets, frame_num)
            ocr_budget = 3
            for vid, track in tracks.items():
                vx1, vy1, vx2, vy2 = track.bbox
                license_plate.draw_neon_box(frame, vx1, vy1, vx2, vy2, license_plate.NEON_GREEN if track.confirmed_plate else license_plate.NEON_CYAN)
                license_plate.draw_label(frame, f"VEH#{track.vehicle_id}", vx1, vy1-4, license_plate.NEON_CYAN)
                
                vehicle_crop = frame[max(0, vy1):min(h, vy2), max(0, vx1):min(w, vx2)]
                if vehicle_crop.size > 0:
                    p_res = plate_model(vehicle_crop, verbose=False, conf=conf_threshold)[0]
                    for p_box in p_res.boxes:
                        pxy = p_box.xyxy[0].cpu().numpy()
                        px1 = vx1 + int(pxy[0])
                        py1 = vy1 + int(pxy[1])
                        px2 = vx1 + int(pxy[2])
                        py2 = vy1 + int(pxy[3])
                        pconf = float(p_box.conf[0])
                        
                        license_plate.draw_neon_box(frame, px1, py1, px2, py2, license_plate.NEON_YELLOW, thickness=1, glow=False)
                        
                        if frame_num - track.last_ocr_frame >= 45 and ocr_budget > 0:
                            plate_roi = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                            if plate_roi.size > 0:
                                text, conf = ocr.read_plate(plate_roi)
                                track.last_ocr_frame = frame_num
                                ocr_budget -= 1
                                
                                if text and len(text) >= 4:
                                    if conf > track.confirmed_conf:
                                        track.confirmed_plate = text
                                        track.confirmed_conf = conf
                                    plate_records.append(license_plate.PlateRecord(
                                        plate_text=text, confidence=conf,
                                        vehicle_id=track.vehicle_id,
                                        timestamp=datetime.datetime.now().isoformat(),
                                        frame_number=frame_num
                                    ))
                                    plate_counts[text] += 1
                                    track.plates.append(text)
                
                if track.confirmed_plate:
                    license_plate.draw_plate_panel(frame, track.confirmed_plate, track.confirmed_conf, vx1, vy2+5, track.vehicle_id, confirmed=True)
                elif track.plates:
                    license_plate.draw_plate_panel(frame, track.plates[-1], 0.0, vx1, vy2+5, track.vehicle_id, confirmed=False)

            stats_dict = {
                "vehicles": len(tracks),
                "plates": len(plate_records),
                "unique": len(plate_counts),
                "top_plate": max(plate_counts, key=plate_counts.get) if plate_counts else "N/A"
            }
            license_plate.draw_dashboard(frame, stats_dict, fps_disp, frame_num)
            
            met1.metric("Active Vehicles", stats_dict["vehicles"])
            met2.metric("Total Plates OCR'd", stats_dict["plates"])
            met3.metric("Unique Plates", stats_dict["unique"])
            met4.metric("Top Detected Plate", stats_dict["top_plate"])
            
            csv_rows.append({
                "frame": frame_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "active_vehicles": stats_dict["vehicles"],
                "total_plates_read": stats_dict["plates"],
                "unique_plates_count": stats_dict["unique"]
            })
            chart_data.append({"Frame": frame_num, "Vehicles": stats_dict["vehicles"], "Unique Plates": stats_dict["unique"]})

        # 2. Air Combat System
        elif selected_system == "Air Combat Intelligence System":
            if frame_num % detect_every == 0 or frame_num == 1:
                res = model(frame, verbose=False, conf=conf_threshold)[0]
                dets = []
                for box in res.boxes:
                    c = int(box.cls[0])
                    if c in air_combat.ALL_CLS_IDS:
                        xy = box.xyxy[0].cpu().numpy()
                        cls_name = air_combat.AIRCRAFT_CLS.get(c) or air_combat.GROUND_CLS.get(c)
                        dets.append(( (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])), cls_name, float(box.conf[0])))
                last_dets = dets
                
            tracks = tracker.update(last_dets)
            aircraft_tracks = [t for t in tracks if t.base_cls == "aircraft"]
            ground_tracks = [t for t in tracks if t.base_cls != "aircraft"]
            
            speeds = []
            scores = []
            drones = 0
            for tr in aircraft_tracks:
                label = air_combat.classify_aircraft(tr, w, h)
                tr._disp_label = label
                score = air_combat.threat_score(tr, aircraft_tracks, w, h)
                scores.append(score)
                vx, vy = air_combat.velocity_px(tr)
                speeds.append(np.hypot(vx, vy)*18)
                if label == "drone": drones += 1
                color = air_combat.CLASS_COLORS.get(label, air_combat.COL_MILAIR)
                air_combat.draw_target(frame, tr, label, color, frame_num, w, h, score=score)
                cx, cy = tr.centroid()
                heatmap_pts.append((cx, cy))
                
            for tr in ground_tracks:
                label = tr.base_cls
                tr._disp_label = label
                color = air_combat.CLASS_COLORS.get(label, air_combat.COL_VEHICLE)
                air_combat.draw_target(frame, tr, label, color, frame_num, w, h, score=None)
                
            max_threat = max(scores) if scores else 0.0
            peak_threat = max(peak_threat, max_threat)
            top_target = aircraft_tracks[scores.index(max_threat)].tid if scores else -1
            collisions = sum(1 for s in scores if s >= 66)
            density = min(100.0, len(tracks) / 12.0 * 100)
            
            stats_dict = {
                "aircraft": len(aircraft_tracks),
                "drones": drones,
                "avg_speed": sum(speeds)/len(speeds) if speeds else 0.0,
                "top_target": top_target,
                "max_threat": max_threat,
                "density": density,
                "collisions": collisions,
            }
            
            air_combat.draw_scanline(frame, frame_num)
            air_combat.draw_minimap(frame, tracks, w, h, heatmap_pts)
            air_combat.draw_dashboard(frame, stats_dict, fps_disp, frame_num)
            air_combat.draw_corners(frame)
            
            met1.metric("Active Aircraft", stats_dict["aircraft"])
            met2.metric("Threat Index", f"{stats_dict['max_threat']:.0f}%")
            met3.metric("Airspace Density", f"{stats_dict['density']:.0f}%")
            met4.metric("Collision Risks", stats_dict["collisions"])
            
            csv_rows.append({
                "frame": frame_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "active_aircraft": stats_dict["aircraft"],
                "active_drones": stats_dict["drones"],
                "max_threat": stats_dict["max_threat"],
                "airspace_density": stats_dict["density"]
            })
            chart_data.append({"Frame": frame_num, "Airspace Density": stats_dict["density"], "Max Threat %": stats_dict["max_threat"]})

        # 3. Airport Runway System
        else:
            if frame_num % detect_every == 0 or frame_num == 1:
                res = model(frame, verbose=False, conf=conf_threshold)[0]
                dets = []
                for box in res.boxes:
                    c = int(box.cls[0])
                    if c in airport_runway.ALL_CLS_IDS:
                        xy = box.xyxy[0].cpu().numpy()
                        dets.append(( (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])), float(box.conf[0]), c))
                last_dets = dets
                
            tracks = tracker.update(last_dets)
            stats_dict = airport_runway.defaultdict(int)
            
            for tr in tracks:
                col = airport_runway.NEON.get(tr.category, (180,180,180))
                airport_runway.glow_edge(frame, tr.bbox, col, passes=3)
                airport_runway.corner_ticks(frame, tr.bbox, col)
                airport_runway.draw_trajectory(frame, tr.history, col)
                
                is_aircraft = tr.category in (
                    airport_runway.Category.COMMERCIAL_AIRCRAFT,
                    airport_runway.Category.CARGO_AIRCRAFT,
                    airport_runway.Category.PRIVATE_JET,
                    airport_runway.Category.HELICOPTER
                )
                if is_aircraft:
                    future = tracker.predict_path(tr, steps=12)
                    airport_runway.draw_predicted_path(frame, future, col)
                    airport_runway.target_lock_anim(frame, tr.bbox, col, frame_num)
                    status = airport_runway.estimate_status(tr)
                    risk_label, risk_col = airport_runway.estimate_risk(tr, restricted_boxes)
                    lines = [
                        f"Status: {status}",
                        f"Speed: {tr.speed_kmh:5.1f} km/h",
                        f"Heading: {tr.heading_deg:6.1f} deg",
                        f"Risk: {risk_label}",
                    ]
                    airport_runway.label_panel(frame, lines, tr.bbox[0], max(0, tr.bbox[1]-78), col, width=190, title=f"Aircraft #{tr.track_id}")
                    stats_dict["aircraft"] += 1
                    if risk_label == "HIGH RISK":
                        stats_dict["high_risk"] += 1
                elif tr.category in (airport_runway.Category.GROUND_CREW, airport_runway.Category.AIRPORT_STAFF):
                    lines = [
                        f"Movement: {airport_runway.estimate_status(tr)}",
                        f"Speed: {tr.speed_kmh:4.1f} km/h"
                    ]
                    airport_runway.label_panel(frame, lines, tr.bbox[0], max(0, tr.bbox[1]-46), col, width=140, title=f"Crew #{tr.track_id}")
                    stats_dict["crew"] += 1
                else:
                    lines = [
                        f"Type: {tr.category.value.replace('_',' ').title()}",
                        f"Speed: {tr.speed_kmh:4.1f} km/h"
                    ]
                    airport_runway.label_panel(frame, lines, tr.bbox[0], max(0, tr.bbox[1]-46), col, width=160, title=f"Vehicle #{tr.track_id}")
                    stats_dict["vehicles"] += 1
            
            zone_layer = frame.copy()
            for zname, (zx1, zy1, zx2, zy2, zcol) in zones_minimap.items():
                cv2.rectangle(zone_layer, (zx1, zy1), (zx2, zy2), zcol, -1)
            cv2.addWeighted(zone_layer, 0.07, frame, 0.93, 0, frame)
            for zname, (zx1, zy1, zx2, zy2, zcol) in zones_minimap.items():
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zcol, 1)
                cv2.putText(frame, zname, (zx1+4, zy1+14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, zcol, 1, cv2.LINE_AA)
                
            airport_runway.draw_scanline(frame, frame_num)
            airport_runway.draw_corner_frame(frame)
            airport_runway.draw_global_dashboard(frame, stats_dict, fps_disp, frame_num)
            
            twin_img = twin.render(tracks, zones_minimap)
            airport_runway.overlay_digital_twin(frame, twin_img)
            
            met1.metric("Active Aircraft", stats_dict["aircraft"])
            met2.metric("Ground Vehicles", stats_dict["vehicles"])
            met3.metric("Ground Crew", stats_dict["crew"])
            met4.metric("High Risk Targets", stats_dict["high_risk"])
            
            csv_rows.append({
                "frame": frame_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "aircraft_count": stats_dict["aircraft"],
                "vehicles_count": stats_dict["vehicles"],
                "crew_count": stats_dict["crew"],
                "high_risk_alerts": stats_dict["high_risk"]
            })
            chart_data.append({"Frame": frame_num, "Aircraft": stats_dict["aircraft"], "Vehicles": stats_dict["vehicles"]})

        writer.write(frame)
        
        # Display image dynamically
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        status_text.text(f"Processing Frame: {frame_num} / {total_frames} | Live FPS: {fps_disp:.1f}")
        
        p_val = min(1.0, float(frame_num) / max(total_frames, 1))
        progress_bar.progress(p_val)
        
        # Update chart
        if len(chart_data) > 0 and frame_num % 5 == 0:
            df = pd.DataFrame(chart_data)
            df.set_index("Frame", inplace=True)
            with chart_placeholder.container():
                st.line_chart(df)
                
    cap.release()
    writer.release()
    
    st.session_state.csv_data = pd.DataFrame(csv_rows)
    st.session_state.output_video_path = out_video_path
    st.session_state.engine_running = False
    
    status_text.text(f"Perception complete! Processed {frame_num} frames.")
    progress_bar.progress(1.0)
    st.success("Successfully generated report files. Go to 'Reports & Exports' tab to download.")
    st.rerun()
