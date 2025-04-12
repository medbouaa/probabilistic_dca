import streamlit as st
import sys
import pathlib

# Add the src directory to the Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

import streamlit as st
import streamlit as st
from PIL import Image
from io import BytesIO
import base64

# Load image and convert to base64
def get_base64_of_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

# Get base64 string
img_base64 = get_base64_of_image("images/dca_workflow_2.png")


# --- Page config ---
st.set_page_config(page_title="Probabilistic DCA — Introduction", layout="wide")

# --- Custom CSS for expander headers ---
def local_css():
    st.markdown(
        """
        <style>
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #1f77b4;
            background-color: #f0f2f6;
            padding: 5px;
            border-radius: 5px;
        }
        .scroll-container {
            overflow-x: auto;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e6e6e6;
        }
        .step {
            margin-bottom: 10px;
            padding: 8px;
            background-color: #f0f2f6;
            border-left: 5px solid #1f77b4;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

st.title("🚀 Probabilistic Decline Curve Analysis (DCA) — Project Overview")

# Introduction
st.markdown(
    """
    Welcome to the **Probabilistic DCA Workflow** based on SPE-194503-PA!  
    This application automates probabilistic decline curve analysis for robust forecasting and uncertainty quantification.

    """
)

# Expander: Goals
with st.expander("🎯 Project Goals"):
    st.markdown(
        """
        - **Enable robust forecasting** of well production and estimated ultimate recovery (EUR).
        - **Quantify uncertainty** by using Monte Carlo sampling and probabilistic methods.
        - **Apply multi-model comparison** to increase reliability (Arps, CRM, SEM, LGM).
        - **Generate professional reports** and visualizations directly from the pipeline.
        """
    )

# Expander: Methodology
with st.expander("🧩 Methodology Overview"):
    st.markdown(
        """
        1. **Data Cleaning:**  
            - Apply outlier detection (LOF algorithm).
        2. **Monte Carlo Sampling:**  
            - Generate thousands of synthetic production profiles.
        3. **Model Fitting:**  
            - Fit multiple decline curve models in parallel.
        4. **Hindcast & Forecasting:**  
            - Test historical fits and predict future performance.
        5. **Model Comparison & Probabilities:**  
            - Calculate marginal posterior probabilities for model selection.
        6. **EUR Estimation:**  
            - Deliver comprehensive EUR forecasts (P10, P50, P90).
        7. **Reporting:**  
            - Export interactive visualizations and tabular results.
        """
    )

# --- Workflow Diagram ---
st.subheader("🔗 Workflow Overview")

# image = Image.open("images/dca_workflow_2.png")

# st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

# st.subheader("Pipeline Workflow")

st.markdown(
    f"""
    <div style="overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
        <img src="data:image/png;base64,{img_base64}" style="display: inline-block; height: auto; width: auto; max-height: 500px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# --- Add Models Summary Section ---
with st.expander("🛠️ Models Used in Probabilistic DCA"):
    st.markdown(
        """
        In this pipeline, we employ a set of robust **decline curve analysis models** to capture a wide range of production behaviors:

        - **📉 Arps Model**  
          Classical decline model (exponential, harmonic, hyperbolic). Ideal for conventional reservoirs.

        - **🔁 Stretched Exponential Model (SEM)**  
          Flexible model capturing transient flow regimes, including tight and unconventional reservoirs.

        - **🔗 Corrected Rate-Time Model (CRM)**  
          Accounts for rate-time distortions, improving accuracy in variable production scenarios.

        - **⚙️ Logistic Growth Model (LGM)**  
          Suitable for constrained growth and development phases, common in unconventional wells.

        ---
        **Combination strategy:** ✅  
        All models are fitted independently, and their predictions are later combined **probabilistically** to derive a robust ensemble forecast.  
        This ensures that the strengths of each model are fully utilized!
        """
    )

# --- Closing Note ---
st.info(
    """
    📢 **Tip:** Use the sidebar to configure your run parameters and launch the analysis.  
    Move to the "Pipeline Run" page to start exploring your production data!
    """
)



st.sidebar.markdown("---")
st.sidebar.markdown("By: Alexis Ortega")
