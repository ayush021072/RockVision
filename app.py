import io
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RockVision | Rock Fragmentation Analysis",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL RESEARCH-STYLE CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.rv-eyebrow {
    color: #a7a99a;
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

.rv-title {
    font-size: clamp(2.1rem, 4vw, 3.1rem);
    line-height: 1.12;
    font-weight: 680;
    letter-spacing: -0.045em;
    margin: 0;
}

.rv-subtitle {
    color: #a9adb5;
    font-size: 1rem;
    line-height: 1.65;
    margin-top: 0.7rem;
    max-width: 800px;
}

.rv-rule {
    border: 0;
    border-top: 1px solid rgba(145, 150, 155, 0.25);
    margin: 1.6rem 0 1.8rem;
}

.rv-section-label {
    color: #a7a99a;
    font-size: 0.75rem;
    font-weight: 650;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.rv-footer {
    color: #92969e;
    font-size: 0.82rem;
    line-height: 1.7;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(145, 150, 155, 0.25);
    border-radius: 9px;
    padding: 1rem;
    background: rgba(145, 150, 155, 0.045);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(145, 150, 155, 0.2);
    border-radius: 8px;
    overflow: hidden;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(145, 150, 155, 0.18);
}

.stDownloadButton button {
    border-radius: 7px;
    font-weight: 550;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    .rv-title {
        font-size: 2.1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "best_maskrcnn_model.pth"

NUM_CLASSES = 2
BACKGROUND_CLASS = 0
ROCK_CLASS = 1

DEFAULT_CONFIDENCE = 0.65
DEFAULT_MASK_THRESHOLD = 0.50


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading the trained RockVision model...")
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    # Replace box predictor for two classes:
    # 0 = background, 1 = rock

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    # Replace mask predictor

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # The saved checkpoint contains model_state_dict.

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel checkpoints.

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image,
    model,
    device,
    confidence_threshold,
    mask_threshold
):

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    image_height, image_width = image_np.shape[:2]

    image_tensor = F.to_tensor(image_rgb).to(device)

    # Run model inference.

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()

    # Filter detections by confidence and rock class.

    keep = (
        (scores >= confidence_threshold)
        &
        (labels == ROCK_CLASS)
    )

    scores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep, 0]

    # Convert probability masks to binary masks.

    binary_masks = masks >= mask_threshold

    # ========================================================
    # ROCK MEASUREMENTS
    # ========================================================

    rows = []

    for i, (score, box, mask) in enumerate(
        zip(scores, boxes, binary_masks),
        start=1
    ):

        area = int(mask.sum())

        if area == 0:
            continue

        x1, y1, x2, y2 = box

        box_width = max(0.0, float(x2 - x1))
        box_height = max(0.0, float(y2 - y1))

        ys, xs = np.where(mask)

        if len(xs) > 0:
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
        else:
            centroid_x = np.nan
            centroid_y = np.nan

        # Equivalent diameter of a circle with the same area.

        equivalent_diameter = (
            2.0 * math.sqrt(area / math.pi)
        )

        bbox_area = box_width * box_height

        fill_ratio = (
            area / bbox_area
            if bbox_area > 0
            else 0.0
        )

        rows.append({
            "Rock ID": len(rows) + 1,
            "Confidence": float(score),
            "Mask Area (px²)": area,
            "BBox Width (px)": box_width,
            "BBox Height (px)": box_height,
            "Centroid X (px)": centroid_x,
            "Centroid Y (px)": centroid_y,
            "Equivalent Diameter (px)": equivalent_diameter,
            "BBox Area (px²)": bbox_area,
            "Mask Fill Ratio": fill_ratio,
        })

    df = pd.DataFrame(rows)

    # ========================================================
    # TOTAL ROCK AREA AND IMAGE COVERAGE
    # ========================================================

    union_mask = np.zeros(
        (image_height, image_width),
        dtype=bool
    )

    for mask in binary_masks:
        union_mask |= mask

    total_rock_area = int(union_mask.sum())

    total_image_area = image_height * image_width

    coverage_percent = (
        100.0 * total_rock_area / total_image_area
        if total_image_area > 0
        else 0.0
    )

    # ========================================================
    # CREATE SEGMENTATION VISUALIZATION
    # ========================================================

    result_image = image_np.copy()
    overlay = result_image.copy()

    rng = np.random.default_rng(42)

    colors = rng.integers(
        40,
        230,
        size=(max(len(binary_masks), 1), 3),
        dtype=np.uint8
    )

    # Draw masks.

    for i, mask in enumerate(binary_masks):

        color = colors[i]

        overlay[mask] = (
            0.55 * overlay[mask]
            +
            0.45 * color
        ).astype(np.uint8)

    result_image = cv2.addWeighted(
        result_image,
        0.55,
        overlay,
        0.45,
        0
    )

    # Draw bounding boxes and confidence labels.

    for i, (score, box) in enumerate(zip(scores, boxes)):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (40, 40, 40),
            2
        )

        label = f"Rock {i + 1}: {score:.2f}"
        text_y = max(25, y1 - 8)

        cv2.putText(
            result_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1
        )

    return (
        result_image,
        df,
        total_rock_area,
        coverage_percent
    )


# ============================================================
# GRAPH FUNCTIONS
# ============================================================

def create_histogram(values, title, xlabel):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(9, 4))

    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#111318")

    if len(values) > 0:

        bins = min(15, max(5, len(values)))

        ax.hist(
            values,
            bins=bins,
            color="#a7ad91",
            edgecolor="#e8e9e3",
            linewidth=0.8
        )

        if "diameter" in title.lower():

            median_value = float(np.median(values))

            ax.axvline(
                median_value,
                color="#e9a85b",
                linestyle="--",
                linewidth=1.7,
                label=f"Median: {median_value:.2f} px"
            )

            ax.legend(
                frameon=False,
                labelcolor="#e7e8eb"
            )

    ax.set_title(
        title,
        color="#f0f0ed",
        pad=12
    )

    ax.set_xlabel(
        xlabel,
        color="#d2d4d8"
    )

    ax.set_ylabel(
        "Number of rocks",
        color="#d2d4d8"
    )

    ax.tick_params(colors="#d2d4d8")
    ax.grid(axis="y", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#42454d")

    fig.tight_layout()

    return fig


def create_cumulative_plot(diameters):

    diameters = np.asarray(diameters, dtype=float)
    diameters = diameters[np.isfinite(diameters)]
    diameters = diameters[diameters > 0]

    fig, ax = plt.subplots(figsize=(9, 4))

    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#111318")

    if len(diameters) > 0:

        sorted_diameters = np.sort(diameters)

        cumulative_percent = (
            np.arange(1, len(sorted_diameters) + 1)
            / len(sorted_diameters)
            * 100.0
        )

        ax.plot(
            sorted_diameters,
            cumulative_percent,
            marker="o",
            markersize=4,
            linewidth=2,
            color="#a7ad91"
        )

    ax.set_title(
        "Cumulative rock-size distribution",
        color="#f0f0ed",
        pad=12
    )

    ax.set_xlabel(
        "Equivalent diameter (pixels)",
        color="#d2d4d8"
    )

    ax.set_ylabel(
        "Cumulative percentage (%)",
        color="#d2d4d8"
    )

    ax.set_ylim(0, 100)
    ax.tick_params(colors="#d2d4d8")
    ax.grid(alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#42454d")

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.markdown("## Analysis settings")

    st.markdown("### Detection")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.30,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.01,
        help="Detections below this confidence are excluded."
    )

    st.caption(
        f"Current confidence: **{confidence_threshold:.2f}**"
    )

    st.markdown("### Segmentation")

    mask_threshold = st.slider(
        "Mask threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_MASK_THRESHOLD,
        step=0.05,
        help="Controls conversion of predicted masks into binary masks."
    )

    st.caption(
        f"Current mask threshold: **{mask_threshold:.2f}**"
    )

    st.info(
        "Recommended starting confidence threshold: 0.65"
    )

    st.caption(
        "Physical measurements require image-scale calibration."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="rv-eyebrow">Mining Engineering · Research Tool</div>'
    '<h1 class="rv-title">RockVision</h1>'
    '<p class="rv-subtitle">'
    'Rock-fragmentation analysis using instance segmentation. '
    'Inspect predicted rock boundaries, estimate fragment sizes, '
    'and export measurements for further analysis.'
    '</p>'
    '<hr class="rv-rule">',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a rock image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# INITIAL SCREEN
# ============================================================

if uploaded_file is None:

    left, right = st.columns(2, gap="large")

    with left:

        with st.container(border=True):

            st.markdown("#### Analysis workflow")

            st.markdown(
                "1. **Upload** a rock image.\n\n"
                "2. **Segment** individual fragments with Mask R-CNN.\n\n"
                "3. **Review** confidence scores and measurements.\n\n"
                "4. **Export** the segmented image and measurement CSV."
            )

    with right:

        with st.container(border=True):

            st.markdown("#### Available measurements")

            st.markdown(
                "- Rock count and mask area\n"
                "- Equivalent diameter\n"
                "- Bounding-box dimensions\n"
                "- Centroid coordinates\n"
                "- Rock coverage percentage\n"
                "- D10, D25, D50, D75, D80 and D90"
            )

    st.caption(
        "Measurements are currently in pixels. "
        "Conversion to mm or cm requires a known reference scale "
        "and appropriate image calibration."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

else:

    try:

        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

    except Exception as error:

        st.error(f"Unable to open the uploaded image: {error}")
        st.stop()

    st.markdown(
        '<div class="rv-section-label">Input image</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption=f"Uploaded image: {uploaded_file.name}",
        use_container_width=True
    )

    try:

        with st.spinner("Loading model and analysing image..."):

            model, device = load_model()

            (
                result_image,
                df,
                total_rock_area,
                coverage_percent
            ) = analyze_image(
                image,
                model,
                device,
                confidence_threshold,
                mask_threshold
            )

    except Exception as error:

        st.error(
            f"Model loading or inference error: {error}"
        )

        st.info(
            "Check that best_maskrcnn_model.pth is present "
            "and contains the trained two-class Mask R-CNN checkpoint."
        )

        st.stop()

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.markdown(
        '<hr class="rv-rule">',
        unsafe_allow_html=True
    )

    st.markdown("## Analysis summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Detected rocks",
        f"{len(df):,}"
    )

    col2.metric(
        "Total rock area",
        f"{total_rock_area:,} px²"
    )

    col3.metric(
        "Image coverage",
        f"{coverage_percent:.2f}%"
    )

    mean_confidence = (
        df["Confidence"].mean()
        if not df.empty
        else 0.0
    )

    col4.metric(
        "Mean confidence",
        f"{mean_confidence:.3f}" if not df.empty else "—"
    )

    # ========================================================
    # SEGMENTATION RESULT
    # ========================================================

    st.markdown("## Segmentation result")

    st.image(
        result_image,
        use_container_width=True
    )

    result_buffer = io.BytesIO()

    Image.fromarray(result_image).save(
        result_buffer,
        format="PNG"
    )

    st.download_button(
        "Download segmented image",
        data=result_buffer.getvalue(),
        file_name="rockvision_segmented.png",
        mime="image/png"
    )

    # ========================================================
    # ROCK-WISE MEASUREMENTS
    # ========================================================

    st.markdown(
        '<hr class="rv-rule">',
        unsafe_allow_html=True
    )

    st.markdown("## Rock-wise measurements")

    if df.empty:

        st.warning(
            "No rocks were detected at the selected thresholds. "
            "Try lowering the confidence threshold."
        )

    else:

        st.dataframe(
            df.round(3),
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download measurements CSV",
            data=csv_data,
            file_name="rock_measurements.csv",
            mime="text/csv"
        )

    # ========================================================
    # SIZE DISTRIBUTION AND STATISTICS
    # ========================================================

    st.markdown(
        '<hr class="rv-rule">',
        unsafe_allow_html=True
    )

    st.markdown("## Rock size distribution")

    if not df.empty:

        # Define diameters BEFORE using them in any graph or statistic.
        # This prevents the previous NameError.

        diameters = pd.to_numeric(
            df["Equivalent Diameter (px)"],
            errors="coerce"
        ).dropna()

        diameters = diameters[
            np.isfinite(diameters) & (diameters > 0)
        ]

        areas = pd.to_numeric(
            df["Mask Area (px²)"],
            errors="coerce"
        ).dropna()

        if len(diameters) > 0:

            # Two graphs: diameter histogram and cumulative distribution.

            graph1, graph2 = st.columns(2, gap="large")

            with graph1:

                fig_histogram = create_histogram(
                    diameters.to_numpy(),
                    "Distribution of detected rock diameters",
                    "Equivalent diameter (pixels)"
                )

                st.pyplot(
                    fig_histogram,
                    use_container_width=True
                )

                plt.close(fig_histogram)

            with graph2:

                fig_cumulative = create_cumulative_plot(
                    diameters.to_numpy()
                )

                st.pyplot(
                    fig_cumulative,
                    use_container_width=True
                )

                plt.close(fig_cumulative)

            # ====================================================
            # D10, D25, D50, D75, D80 AND D90
            # ====================================================

            d10 = float(diameters.quantile(0.10))
            d25 = float(diameters.quantile(0.25))
            d50 = float(diameters.quantile(0.50))
            d75 = float(diameters.quantile(0.75))
            d80 = float(diameters.quantile(0.80))
            d90 = float(diameters.quantile(0.90))

            st.markdown("### Size percentiles")

            p1, p2, p3, p4, p5, p6 = st.columns(6)

            p1.metric("D10", f"{d10:.2f} px")
            p2.metric("D25", f"{d25:.2f} px")
            p3.metric("D50", f"{d50:.2f} px")
            p4.metric("D75", f"{d75:.2f} px")
            p5.metric("D80", f"{d80:.2f} px")
            p6.metric("D90", f"{d90:.2f} px")

            st.caption(
                "D-values are percentiles of the detected equivalent "
                "diameters in this image. They are not calibrated "
                "physical particle sizes."
            )

            # ====================================================
            # STATISTICAL ANALYSIS
            # ====================================================

            st.markdown(
                '<hr class="rv-rule">',
                unsafe_allow_html=True
            )

            st.markdown("## Statistical analysis")

            summary_df = pd.DataFrame({
                "Metric": [
                    "Mean diameter",
                    "Median diameter",
                    "Standard deviation",
                    "Minimum diameter",
                    "Maximum diameter",
                    "D10",
                    "D25",
                    "D50",
                    "D75",
                    "D80",
                    "D90",
                    "Mean area",
                    "Median area"
                ],

                "Value": [
                    diameters.mean(),
                    diameters.median(),
                    diameters.std(),
                    diameters.min(),
                    diameters.max(),
                    d10,
                    d25,
                    d50,
                    d75,
                    d80,
                    d90,
                    areas.mean(),
                    areas.median()
                ]
            })

            st.dataframe(
                summary_df.round(3),
                use_container_width=True,
                hide_index=True
            )

            # Area distribution graph.

            st.markdown("### Rock area distribution")

            fig_area = create_histogram(
                areas.to_numpy(),
                "Distribution of segmented rock areas",
                "Mask area (pixels²)"
            )

            st.pyplot(
                fig_area,
                use_container_width=True
            )

            plt.close(fig_area)

            # ====================================================
            # DOWNLOAD COMPLETE ANALYSIS
            # ====================================================

            st.markdown("### Export analysis")

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                df.to_excel(
                    writer,
                    sheet_name="Rock Measurements",
                    index=False
                )

                summary_df.to_excel(
                    writer,
                    sheet_name="Statistical Analysis",
                    index=False
                )

            st.download_button(
                "Download complete analysis (Excel)",
                data=excel_buffer.getvalue(),
                file_name="rockvision_complete_analysis.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

        else:

            st.info(
                "No valid equivalent diameters are available "
                "for statistical analysis."
            )

    else:

        st.info(
            "Statistical analysis will appear when at least one "
            "rock is detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<hr class="rv-rule">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="rv-footer">'
    'RockVision · Mask R-CNN instance segmentation<br>'
    'Mining Engineering BTP · IIT (BHU), Varanasi'
    '</div>',
    unsafe_allow_html=True
)
