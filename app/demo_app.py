import streamlit as st
import os
import cv2
import tempfile
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NitVision – Object Detection",
    page_icon="🔭",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #a0a0b0;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #333;
    }
    div[data-testid="stTab"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔭 NitVision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time object detection powered by YOLOv5 · Person · Car · Bicycle · Dog</p>', unsafe_allow_html=True)
st.divider()

# ── Model loading ─────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(current_dir, "..", "models", "yolov5su.pt")

@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except ImportError:
        st.error("ultralytics not installed. Run: `pip install ultralytics`")
        st.stop()

model = load_model(MODEL_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=0.30, step=0.05,
        help="Only show detections above this confidence score"
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Overlap threshold for non-maximum suppression"
    )
    st.divider()
    st.markdown("### 📦 Model")
    st.code("YOLOv5su (ultralytics)", language=None)
    st.divider()
    st.markdown("### 🏷️ Classes")
    st.markdown("🧍 Person · 🚗 Car · 🐕 Dog")

# ── Guard: model missing ──────────────────────────────────────────────────────
if model is None:
    st.error(
        f"❌ Model not found at `{os.path.abspath(MODEL_PATH)}`\n\n"
        "Make sure `yolov5su.pt` exists inside the `models/` folder."
    )
    st.stop()

# ── Helper: detection stats table ─────────────────────────────────────────────
def build_stats_table(result):
    boxes   = result.boxes
    classes = result.names
    if len(boxes) == 0:
        return None
    stats = {}
    for box in boxes:
        cls_id   = int(box.cls[0].item())
        conf     = float(box.conf[0].item())
        cls_name = classes[cls_id]
        if cls_name not in stats:
            stats[cls_name] = {"count": 0, "conf_sum": 0.0, "max_conf": 0.0}
        stats[cls_name]["count"]    += 1
        stats[cls_name]["conf_sum"] += conf
        stats[cls_name]["max_conf"]  = max(stats[cls_name]["max_conf"], conf)
    return [
        {
            "Class":           cls_name,
            "Count":           d["count"],
            "Avg Confidence":  f"{d['conf_sum'] / d['count']:.2%}",
            "Max Confidence":  f"{d['max_conf']:.2%}",
        }
        for cls_name, d in sorted(stats.items(), key=lambda x: -x[1]["count"])
    ]

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_image, tab_video = st.tabs(["🖼️  Image Detection", "🎬  Video Detection"])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 — IMAGE
# ──────────────────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown("#### Upload an image to detect objects")

    uploaded_image = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="img_uploader",
    )

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")

            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            with st.spinner("Running detection…"):
                results = model.predict(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result  = results[0]
                res_img = result.plot()   # numpy RGB array

            with col2:
                st.markdown("**Detected Objects**")
                st.image(res_img, use_container_width=True)

            # Metrics row
            boxes = result.boxes
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Detections", len(boxes))
            if len(boxes) > 0:
                avg_conf = float(boxes.conf.mean().item())
                max_conf = float(boxes.conf.max().item())
                m2.metric("Avg Confidence", f"{avg_conf:.2%}")
                m3.metric("Max Confidence", f"{max_conf:.2%}")

            # Stats table
            table = build_stats_table(result)
            if table:
                st.markdown("#### 📊 Detection Summary")
                st.table(table)
            else:
                st.info("No objects detected. Try lowering the **Confidence Threshold** in the sidebar.")

        except Exception as e:
            st.error(f"Error during detection: {e}")
    else:
        st.info("⬆️ Upload an image above to get started.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2 — VIDEO
# ──────────────────────────────────────────────────────────────────────────────
with tab_video:
    st.markdown("#### Upload a video to run frame-by-frame detection")

    uploaded_video = st.file_uploader(
        "Choose an MP4, AVI, or MOV file",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
    )

    if uploaded_video:
        # Write to temp input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        cap = cv2.VideoCapture(tmp_in_path)

        if not cap.isOpened():
            st.error("❌ Could not open video. Please try MP4 format.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps

            # Info row
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Total Frames", total_frames)
            i2.metric("FPS", f"{fps:.1f}")
            i3.metric("Resolution", f"{width}×{height}")
            i4.metric("Duration", f"{duration_sec:.1f}s")

            st.divider()

            # Processing options
            opt1, opt2 = st.columns(2)
            with opt1:
                skip_n = st.number_input(
                    "Process every N-th frame (1 = all frames, higher = faster)",
                    min_value=1, max_value=30, value=2, step=1,
                )
            with opt2:
                max_proc = st.number_input(
                    "Max frames to process (0 = all)",
                    min_value=0, max_value=total_frames, value=0, step=50,
                )

            process_btn = st.button("▶️  Start Processing", type="primary", use_container_width=True)

            if process_btn:
                # Output temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    tmp_out_path = tmp_out.name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0, text="Starting…")
                status_col1, status_col2 = st.columns([1, 1])
                frame_preview = st.empty()

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx   = 0
                processed   = 0
                limit       = int(max_proc) if max_proc > 0 else total_frames
                all_detections = {}   # class → total count

                while True:
                    ret, frame = cap.read()
                    if not ret or processed >= limit:
                        break

                    if frame_idx % int(skip_n) == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        results = model.predict(
                            rgb,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            verbose=False,
                        )
                        result = results[0]

                        # Collect aggregate stats
                        for box in result.boxes:
                            cls_name = result.names[int(box.cls[0].item())]
                            all_detections[cls_name] = all_detections.get(cls_name, 0) + 1

                        # Write annotated frame
                        annotated_rgb = result.plot()
                        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                        writer.write(annotated_bgr)

                        processed += 1

                        # Update UI every 5 processed frames
                        if processed % 5 == 0 or processed == 1:
                            pct = min(processed / limit, 1.0)
                            progress_bar.progress(
                                pct,
                                text=f"Processing… {processed}/{limit} frames ({pct:.0%})"
                            )
                            frame_preview.image(
                                annotated_rgb,
                                caption=f"Live preview — frame {frame_idx}",
                                use_container_width=True,
                            )

                    frame_idx += 1

                cap.release()
                writer.release()

                progress_bar.progress(1.0, text=f"✅ Done! Processed {processed} frames.")
                frame_preview.empty()

                # Read output and display
                with open(tmp_out_path, "rb") as f:
                    video_bytes = f.read()

                st.success(f"✅ Video processed — {processed} frames annotated.")
                st.video(video_bytes)

                st.download_button(
                    label="⬇️  Download Annotated Video",
                    data=video_bytes,
                    file_name="nitvision_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                # Aggregate stats
                if all_detections:
                    st.markdown("#### 📊 Overall Detection Summary")
                    agg_table = [
                        {"Class": cls, "Total Detections": count}
                        for cls, count in sorted(all_detections.items(), key=lambda x: -x[1])
                    ]
                    st.table(agg_table)
                else:
                    st.info("No objects detected across the video. Try lowering the confidence threshold.")

                # Cleanup
                for path in [tmp_in_path, tmp_out_path]:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    else:
        st.info("⬆️ Upload a video above to get started.")
