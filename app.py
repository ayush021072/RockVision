import streamlit as st
import torch
import pandas as pd
import numpy as np
import math
import cv2

from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RockVision",
    page_icon="🪨",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================


# ============================================================
# CUSTOM CSS — ROCKVISION RESEARCH INTERFACE
# ============================================================

st.markdown("""
<style>
/* Overall typography and spacing */
.stApp {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1440px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* Restrained research-style header */
.rv-eyebrow {
    color: #a7a99a;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

.rv-title {
    font-size: clamp(2.1rem, 4vw, 3.1rem);
    line-height: 1.12;
    font-weight: 650;
    letter-spacing: -0.045em;
    margin: 0;
}

.rv-subtitle {
    font-size: 1.02rem;
    line-height: 1.65;
    color: #a9adb5;
    margin-top: 0.7rem;
    max-width: 760px;
}

.rv-rule {
    border: 0;
    border-top: 1px solid rgba(145, 150, 155, 0.25);
    margin: 1.6rem 0 1.8rem 0;
}

.rv-section-label {
    color: #a7a99a;
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.rv-section-title {
    font-size: 1.45rem;
    font-weight: 620;
    letter-spacing: -0.025em;
    margin: 0 0 0.5rem 0;
}

.rv-muted {
    color: #a9adb5;
    line-height: 1.65;
    font-size: 0.94rem;
}

/* Subtle upload and metric containers */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(145, 150, 155, 0.25);
    border-radius: 9px;
    padding: 1rem 1.05rem;
    background: rgba(145, 150, 155, 0.045);
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
}

[data-testid="stMetricValue"] {
    font-weight: 650;
}

/* Buttons */
.stDownloadButton button,
.stButton button {
    border-radius: 7px;
    font-weight: 550;
    min-height: 2.65rem;
    transition: border-color 0.15s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(145, 150, 155, 0.18);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    letter-spacing: -0.02em;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(145, 150, 155, 0.22);
    border-radius: 8px;
    overflow: hidden;
}

/* Footer */
.rv-footer {
    color: #92969e;
    font-size: 0.82rem;
    line-height: 1.7;
    padding-top: 0.5rem;
}

/* Small screens */
@media (max-width: 700px) {
    .block-container {
        padding: 1.3rem 1rem 2rem 1rem;
    }

    .rv-title {
        font-size: 2.1rem;
    }
}
</style>
""", unsafe_allow_html=True)




# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="rv-eyebrow">Mining Engineering · Research Tool</div>'
    '<h1 class="rv-title">RockVision</h1>'
    '<p class="rv-subtitle">'
    'Rock-fragmentation analysis using instance segmentation. '
    'Inspect individual rock boundaries, estimate fragment sizes, '
    'and export measurements for further analysis.'
    '</p>'
    '<hr class="rv-rule">',
    unsafe_allow_html=True
)



# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "best_maskrcnn_model.pth"

NUM_CLASSES = 2
BACKGROUND_CLASS = 0
ROCK_CLASS = 1


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the same Mask R-CNN architecture
    # used for the trained model.

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    # --------------------------------------------------------
    # Replace box predictor
    # --------------------------------------------------------

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    # --------------------------------------------------------
    # Replace mask predictor
    # --------------------------------------------------------

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load your trained checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Your checkpoint contains:
    #
    # epoch
    # model_state_dict
    # optimizer_state_dict
    # scheduler_state_dict
    # best_val_loss
    # training_history

    state_dict = checkpoint["model_state_dict"]

    # Handle DataParallel checkpoints if necessary

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

    image_tensor = F.to_tensor(
        image_rgb
    ).to(device)

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.inference_mode():

        prediction = model(
            [image_tensor]
        )[0]

    # --------------------------------------------------------
    # Convert predictions to NumPy
    # --------------------------------------------------------

    scores = (
        prediction["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    boxes = (
        prediction["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    masks = (
        prediction["masks"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        prediction["labels"]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Filter confidence + rock class
    # --------------------------------------------------------

    keep = (
        (scores >= confidence_threshold)
        &
        (labels == ROCK_CLASS)
    )

    scores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep, 0]

    # --------------------------------------------------------
    # Convert masks to binary masks
    # --------------------------------------------------------

    binary_masks = (
        masks >= mask_threshold
    )

    # ========================================================
    # ROCK MEASUREMENTS
    # ========================================================

    rows = []

    for i, (score, box, mask) in enumerate(
        zip(scores, boxes, binary_masks),
        start=1
    ):

        # Mask area

        area = int(
            mask.sum()
        )

        # Bounding box

        x1, y1, x2, y2 = box

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        # Centroid

        ys, xs = np.where(mask)

        if len(xs) > 0:

            centroid_x = float(
                xs.mean()
            )

            centroid_y = float(
                ys.mean()
            )

        else:

            centroid_x = np.nan
            centroid_y = np.nan

        # Equivalent diameter

        if area > 0:

            equivalent_diameter = (
                2 *
                math.sqrt(
                    area / math.pi
                )
            )

        else:

            equivalent_diameter = 0

        # Bounding box area

        bbox_area = (
            width * height
        )

        # Mask fill ratio

        if bbox_area > 0:

            fill_ratio = (
                area / bbox_area
            )

        else:

            fill_ratio = 0

        rows.append({

            "Rock ID": i,

            "Confidence": float(
                score
            ),

            "Mask Area (px²)": area,

            "BBox Width (px)": width,

            "BBox Height (px)": height,

            "Centroid X (px)": centroid_x,

            "Centroid Y (px)": centroid_y,

            "Equivalent Diameter (px)":
                equivalent_diameter,

            "BBox Area (px²)":
                bbox_area,

            "Mask Fill Ratio":
                fill_ratio
        })


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)


    # ========================================================
    # TOTAL ROCK AREA
    # ========================================================

    image_height = image_np.shape[0]
    image_width = image_np.shape[1]

    union_mask = np.zeros(
        (image_height, image_width),
        dtype=bool
    )

    for mask in binary_masks:

        union_mask |= mask


    total_rock_area = int(
        union_mask.sum()
    )


    total_image_area = (
        image_height *
        image_width
    )


    coverage_percent = (
        total_rock_area /
        total_image_area *
        100
    )


    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    result_image = image_np.copy()

    overlay = result_image.copy()


    # Generate consistent colors

    rng = np.random.default_rng(42)

    colors = rng.integers(
        40,
        230,
        size=(
            max(len(binary_masks), 1),
            3
        ),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Draw masks
    # --------------------------------------------------------

    for i, mask in enumerate(
        binary_masks
    ):

        color = colors[i]

        overlay[mask] = (
            0.55 *
            overlay[mask]
            +
            0.45 *
            color
        ).astype(np.uint8)


    result_image = cv2.addWeighted(
        result_image,
        0.55,
        overlay,
        0.45,
        0
    )


    # --------------------------------------------------------
    # Draw bounding boxes + labels
    # --------------------------------------------------------

    for i, (score, box) in enumerate(
        zip(scores, boxes)
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (40, 40, 40),
            2
        )

        label = (
            f"Rock {i + 1}: "
            f"{score:.2f}"
        )

        text_y = max(
            25,
            y1 - 8
        )

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
# SIDEBAR
# ============================================================

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    st.markdown("### 🎯 Detection")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.65,
        step=0.01,
        help="Only detections with confidence equal to or above this value are included."
    )

    st.caption(
        f"Current confidence: **{confidence_threshold:.2f}**"
    )

    st.markdown("### 🎭 Segmentation")

    mask_threshold = st.slider(
        "Mask Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Controls how the predicted mask is converted into a binary rock mask."
    )

    st.caption(
        f"Current mask threshold: **{mask_threshold:.2f}**"
    )

    st.info(
        "Recommended confidence threshold: **0.65**"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a Rock Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        with st.spinner(
            "Loading RockVision AI model..."
        ):

            model, device = load_model()


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing rock image..."
        ):

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


        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        st.subheader(
            "📊 Analysis Summary"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🪨 Rock Count",
                len(df)
            )


        with col2:

            st.metric(
                "📐 Total Rock Area",
                f"{total_rock_area:,} px²"
            )


        with col3:

            st.metric(
                "📊 Rock Coverage",
                f"{coverage_percent:.2f}%"
            )


        with col4:

            if len(df) > 0:

                mean_confidence = (
                    df["Confidence"].mean()
                )

                st.metric(
                    "🎯 Mean Confidence",
                    f"{mean_confidence:.3f}"
                )

            else:

                st.metric(
                    "🎯 Mean Confidence",
                    "—"
                )


        # ====================================================
        # RESULT IMAGE
        # ====================================================

        st.subheader(
            "🔍 AI Segmentation Result"
        )

        st.image(
            result_image,
            use_container_width=True
        )


        # Download processed image

        result_pil = Image.fromarray(
            result_image
        )

        image_buffer = (
            __import__("io")
            .BytesIO()
        )

        result_pil.save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download Segmented Image",
            data=image_buffer.getvalue(),
            file_name="rockvision_result.png",
            mime="image/png"
        )


        # ====================================================
        # ROCK MEASUREMENTS
        # ====================================================

        if len(df) > 0:

            st.subheader(
                "📏 Rock-wise Measurements"
            )

            st.dataframe(
                df.round(3),
                use_container_width=True,
                hide_index=True
            )


            # CSV download

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                "⬇️ Download Measurements CSV",
                data=csv_data,
                file_name="rock_measurements.csv",
                mime="text/csv"
            )


           
# ============================================================
# ROCK SIZE DISTRIBUTION — HISTOGRAM
# ============================================================


            


            

            # =================================================
            # STATISTICAL ANALYSIS
            # =================================================

            st.subheader(
                "📊 Statistical Analysis"
            )


            areas = (
                df[
                    "Mask Area (px²)"
                ]
            )


            d10 = diameters.quantile(0.10)
            d25 = diameters.quantile(0.25)
            d50 = diameters.quantile(0.50)
            d75 = diameters.quantile(0.75)
            d80 = diameters.quantile(0.80)
            d90 = diameters.quantile(0.90)


            summary = pd.DataFrame({

                "Metric": [

                    "Mean Diameter",

                    "Median Diameter",

                    "Standard Deviation",

                    "Minimum Diameter",

                    "Maximum Diameter",

                    "D10",

                    "D25",

                    "D50",

                    "D75",

                    "D80",

                    "D90",

                    "Mean Area",

                    "Median Area"
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
                summary.round(3),
                use_container_width=True,
                hide_index=True
            )


        else:

            st.warning(
                "No rocks were detected at the selected confidence threshold."
            )


    except FileNotFoundError:

        st.error(
            "❌ Model file not found."
        )

        st.info(
            "The file 'best_maskrcnn_model.pth' "
            "must be available with the application."
        )


    except Exception as error:

        st.error(
            f"❌ Model loading/inference error: {error}"
        )

        st.info(
            "Please check that the uploaded checkpoint "
            "is the trained 2-class Mask R-CNN model."
        )



# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.markdown(
        '<div class="rv-section-label">Getting started</div>'
        '<h2 class="rv-section-title">Analyse a rock image</h2>'
        '<p class="rv-muted">'
        'Upload a photograph of a rock pile or fragmented rock surface. '
        'RockVision will identify individual fragments and calculate '
        'image-based measurements from the predicted masks.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Analysis workflow")
            st.markdown(
                "1. **Upload** a JPG, PNG, JPEG or WEBP image.\n\n"
                "2. **Segment** individual rocks with the trained "
                "Mask R-CNN model.\n\n"
                "3. **Review** detections, confidence scores and "
                "rock-wise measurements.\n\n"
                "4. **Export** the segmented image and measurement CSV."
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Measurements available")
            st.markdown(
                "- Detected rock count\n"
                "- Individual mask area and equivalent diameter\n"
                "- Bounding-box dimensions and centroid coordinates\n"
                "- Percentage of image area covered by detected rocks\n"
                "- Size percentiles: D10, D25, D50, D75, D80 and D90"
            )

    st.write("")
    st.caption(
        "Measurement note: results are currently reported in pixels. "
        "Physical dimensions in mm or cm require a known scale and "
        "appropriate image calibration."
    )



# ============================================================
# FOOTER
# ============================================================




