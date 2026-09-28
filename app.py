"""
Helmet Detection – Streamlit App
Model: YOLO11n fine-tuned on helmet dataset
Classes: 0 = helmet  |  1 = no_helmet
"""

import io
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection – YOLO11",
    page_icon="⛑️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⛑️ Helmet Detection using YOLO11")
st.markdown(
    "Detect **helmet** and **no-helmet** in images or videos "
    "using a fine-tuned YOLO11n model trained on the Kaggle Helmet Detection dataset."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_BEST = os.path.join(os.path.dirname(__file__), "best.pt")
HAS_BEST   = os.path.exists(LOCAL_BEST)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if HAS_BEST:
        st.success("✅ `best.pt` detected — using trained model.")
        model_choice = st.radio(
            "Model",
            ["Trained best.pt (recommended)", "Upload a different .pt"],
        )
    else:
        st.warning("No `best.pt` found. Place it next to app.py or upload below.")
        model_choice = st.radio(
            "Model",
            ["Base yolo11n.pt (auto-download)", "Upload trained best.pt"],
        )

    uploaded_model = None
    if "Upload" in model_choice:
        uploaded_model = st.file_uploader("Upload .pt weights", type=["pt"])

    st.divider()
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)




# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached so it loads only once per path)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> YOLO:
    return YOLO(path)


# Resolve which model file to use
if "Upload" in model_choice:
    if uploaded_model is None:
        st.info("⬆️ Upload a `.pt` file from the sidebar to continue.")
        st.stop()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp.write(uploaded_model.read())
    tmp.flush()
    model_path = tmp.name
elif HAS_BEST:
    model_path = LOCAL_BEST
else:
    model_path = "yolo11n.pt"   # Ultralytics will auto-download

model = load_model(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def classify(label: str) -> str:
    """Return 'no_helmet' or 'helmet' regardless of exact label text."""
    l = label.lower()
    if "no" in l or "without" in l:
        return "no_helmet"
    return "helmet"


def predict_image(bgr: np.ndarray):
    """Run inference on a single BGR frame. Returns annotated BGR + detections."""
    results = model.predict(source=bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
    r = results[0]
    detections = [
        {"class": model.names[int(b.cls[0])], "conf": float(b.conf[0])}
        for b in r.boxes
    ]
    return r.plot(), detections          # plot() → BGR ndarray


def summary_cards(detections: list):
    """Render metric cards + per-detection list."""
    if not detections:
        st.warning("No objects detected. Try lowering the confidence threshold.")
        return

    with_h    = sum(1 for d in detections if classify(d["class"]) == "helmet")
    without_h = sum(1 for d in detections if classify(d["class"]) == "no_helmet")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Detections",  len(detections))
    c2.metric("✅ With Helmet",     with_h)
    c3.metric("🚨 Without Helmet", without_h)

    if without_h:
        st.error(f"⚠️ {without_h} person(s) detected **without** a helmet!")
    else:
        st.success("✅ All detected persons are wearing helmets.")

    st.markdown("#### Per-Detection Details")
    for i, d in enumerate(detections, 1):
        icon = "🚨" if classify(d["class"]) == "no_helmet" else "✅"
        st.markdown(f"{icon} **#{i}** &nbsp; `{d['class']}` &nbsp;— conf: `{d['conf']:.2%}`")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["🖼️ Image Detection", "🎬 Video Detection"])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    st.subheader("Upload an Image")
    img_file = st.file_uploader(
        "JPG / JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        key="img_up",
    )

    if img_file:
        pil_img   = Image.open(img_file).convert("RGB")
        bgr_img   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        col_orig, col_det = st.columns(2, gap="large")
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running detection…"):
            annotated_bgr, detections = predict_image(bgr_img)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        with col_det:
            st.markdown("**Detection Result**")
            st.image(annotated_pil, use_container_width=True)

        st.divider()
        st.subheader("📊 Detection Summary")
        summary_cards(detections)

        # Download
        st.divider()
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name="helmet_result.png",
            mime="image/png",
        )
    else:
        st.info("Upload an image above to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.subheader("Upload a Video")
    vid_file = st.file_uploader(
        "MP4 / AVI / MOV",
        type=["mp4", "avi", "mov"],
        key="vid_up",
    )

    if vid_file:
        # Write uploaded video to a temp file
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        in_tmp.write(vid_file.read())
        in_tmp.flush()
        in_path = in_tmp.name

        st.markdown("**Original Video**")
        st.video(in_path)

        if st.button("▶️ Run Detection on Video", type="primary"):
            out_raw  = tempfile.mktemp(suffix="_raw.mp4")
            out_h264 = tempfile.mktemp(suffix="_out.mp4")

            cap    = cv2.VideoCapture(in_path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            writer = cv2.VideoWriter(
                out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

            prog   = st.progress(0, text="Processing…")
            status = st.empty()
            all_dets = []

            frame_n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ann_frame, dets = predict_image(frame)
                writer.write(ann_frame)
                all_dets.extend(dets)
                frame_n += 1
                pct = frame_n / total if total > 0 else 0
                prog.progress(min(pct, 1.0), text=f"Frame {frame_n}/{total}")

            cap.release()
            writer.release()
            prog.empty()
            status.empty()

            # Re-encode to H.264 so browsers can play it inline
            ffmpeg_ok = os.system(
                f'ffmpeg -y -i "{out_raw}" -vcodec libx264 -pix_fmt yuv420p '
                f'-acodec aac "{out_h264}" 2>nul'
            )
            final_path = out_h264 if (ffmpeg_ok == 0 and os.path.exists(out_h264) and os.path.getsize(out_h264) > 0) else out_raw

            st.success("✅ Detection complete!")

            with open(final_path, "rb") as f:
                vid_bytes = f.read()

            st.markdown("**Annotated Video**")
            st.video(vid_bytes)

            st.download_button(
                "⬇️ Download annotated video",
                data=vid_bytes,
                file_name="helmet_video_result.mp4",
                mime="video/mp4",
            )

            # Summary
            st.divider()
            st.subheader("📊 Video Detection Summary (all frames)")
            summary_cards(all_dets)

            # Cleanup
            for p in [in_path, out_raw, out_h264]:
                try:
                    os.remove(p)
                except OSError:
                    pass
    else:
        st.info("Upload a video above to run frame-by-frame detection.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) · "
    "Built with [Streamlit](https://streamlit.io)"
)
