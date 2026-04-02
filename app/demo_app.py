import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Night Vision Object Detection",
    page_icon="🌡️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
CLASS_NAMES = ["Person", "Car", "Bicycle", "Dog"]

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Detectable Classes**")
for cls in CLASS_NAMES:
    st.sidebar.markdown(f"- {cls}")

# ── Main title ────────────────────────────────────────────────────────────────
st.title("🌡️ Night Vision Object Detection")
st.markdown(
    "Upload a **FLIR thermal infrared image** to detect objects using a "
    "YOLOv8s model fine-tuned on the FLIR ADAS thermal dataset."
)
st.markdown("---")

# ── Check model exists before anything else ───────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ **Model not found:** `models/best.pt`\n\n"
        f"Please run **02_train.ipynb** first to train the model. "
        f"The trained weights will be saved to `models/best.pt` automatically."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a thermal image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:

    # Read uploaded image as numpy array (RGB)
    pil_img  = Image.open(uploaded).convert("RGB")
    img_rgb  = np.array(pil_img)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Run inference
    with st.spinner("Running inference..."):
        results = model.predict(
            source  = img_bgr,
            conf    = conf_threshold,
            iou     = 0.45,
            imgsz   = 640,
            verbose = False,
        )

    result     = results[0]
    annotated  = result.plot()                          # BGR with boxes drawn
    annotated  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ── Side-by-side display ──────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Original Image")
        st.image(img_rgb, use_container_width=True)

    with col_right:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    # ── Detection table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Detection Summary")

    boxes       = result.boxes
    class_ids   = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    if len(class_ids) == 0:
        st.info("No objects detected at the current confidence threshold. Try lowering it.")
    else:
        # Aggregate count + avg confidence per class
        agg = defaultdict(lambda: {"count": 0, "conf_sum": 0.0})
        for cls_id, conf in zip(class_ids, confidences):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            agg[name]["count"]    += 1
            agg[name]["conf_sum"] += conf

        table_data = [
            {
                "Class":           name,
                "Count":           v["count"],
                "Avg Confidence":  f"{v['conf_sum'] / v['count']:.2f}",
            }
            for name, v in sorted(agg.items(), key=lambda x: -x[1]["count"])
        ]

        st.table(table_data)

        total = sum(v["count"] for v in agg.values())
        st.caption(f"Total detections: **{total}** &nbsp;|&nbsp; Confidence threshold: **{conf_threshold}**")

else:
    st.info("👆 Upload a thermal image to get started.")
