import streamlit as st
import sys
import pathlib
from PIL import Image
from io import BytesIO
import base64

# ✅ Add the src directory to the Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

# ✅ Load image and convert to base64
def get_base64_of_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

img_base64 = get_base64_of_image("images/dca_workflow_2.png")

# ✅ Streamlit page config — Page title
st.set_page_config(page_title="Streamlit App — Probabilistic DCA", layout="wide")

# ✅ Custom CSS for styling
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
        hr {
            margin-top: 0;
            margin-bottom: 1rem;
            border: 0;
            border-top: 1px solid #e6e6e6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# ✅ Main title and subtitle
st.title("Probabilistic Decline Curve Analysis (DCA)")
st.subheader("Project Overview")

# ✅ Short intro text
st.markdown(
    """
    Welcome to the **Probabilistic DCA Workflow** based on SPE-194503-PA!  
    This application automates probabilistic decline curve analysis for robust forecasting and uncertainty quantification.
    """
)

# ✅ Sidebar navigation + Footer note
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("➡️ Go to **Pipeline Run** to start analysis.")
st.sidebar.markdown("➡️ Go to **Generate Report** to create your final report.")
st.sidebar.markdown("---")
st.sidebar.markdown("By: Alexis Ortega")
st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/alexort74/probabilistic_dca)")

# ✅ Layout: Two columns
col1, col2 = st.columns([1, 2])  # Adjust proportions as needed

# --- Left Column: Workflow diagram ---
with col1:
    st.subheader("🔗 Workflow Overview")
    st.markdown("A step-by-step visualization of the pipeline process.")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; border-radius: 5px; border: 1px solid #ddd; padding: 10px;">
        """,
        unsafe_allow_html=True
    )

# --- Right Column: Expandable Sections ---
with col2:
    st.subheader("🧩 Summary")
    st.markdown("A quick overview of the project's objectives, methodology, and models used.")
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("🎯 Project Goals"):
        st.markdown(
            """
            - **Enable robust forecasting** of well production and estimated ultimate recovery (EUR).
            - **Quantify uncertainty** by using Monte Carlo sampling and probabilistic methods.
            - **Apply multi-model comparison** to increase reliability (Arps, CRM, SEM, LGM).
            - **Generate professional reports** and visualizations directly from the pipeline.
            """
        )

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

    # ✅ Tip inside the right column, below expanders
    st.info(
        """
        📢 **Tip:** Use the sidebar to configure your run parameters and launch the analysis.  
        Move to the "Pipeline Run" page to start exploring your production data!
        """
    )
