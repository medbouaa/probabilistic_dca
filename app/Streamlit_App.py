import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Streamlit App", layout="wide")

# Title and subtitle
st.title("📈 Probabilistic Decline Curve Analysis (DCA)")
st.markdown(
    """
    Welcome to the **Probabilistic DCA Workflow** based on SPE-194503-PA!  
    This application automates probabilistic decline curve analysis for robust forecasting and uncertainty quantification.
    """
)
st.markdown("### 📝 Project Overview")

# Card-like container wrapping both columns
with st.container():
    col1, col_spacer, col2 = st.columns([1, 0.1, 1])

    with col1:
        st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 12px; padding: 20px; background-color: #f9f9f9;'>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔄 Workflow Overview")
        st.markdown("""This diagram outlines the staged execution of the Probabilistic DCA pipeline,
        from data loading and outlier removal to Monte Carlo sampling, model fitting,
        and final EUR reporting.
        """)
        image = Image.open("images/dca_workflow_2.png")
        st.image(image, use_container_width=False)
        st.caption("Hong, Aojie, et al. \"Integrating Model Uncertainty in Probabilistic Decline Curve Analysis for Unconventional Oil Production Forecasting.\" Unconventional Resources Technology Conference, Houston, Texas, 23–25 July 2018. Society of Exploration Geophysicists, American Association of Petroleum Geologists, Society of Petroleum Engineers, 2018.\n")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 12px; padding: 20px; background-color: #f9f9f9;'>
        """, unsafe_allow_html=True)

        st.markdown("#### 🧭 Summary")
        st.markdown("""Use the tabs below to understand the goals, methodology,
        and models implemented in this app.
        """)

        summary_tabs = st.tabs(["🎯 Project Goals", "🧪 Methodology Overview", "📚 Models Used in Probabilistic DCA"])

        with summary_tabs[0]:
            st.markdown("""
            - Provide a user-friendly interface for probabilistic decline curve analysis.
            - Enable model comparison using Monte Carlo sampling.
            - Support multi-model probabilistic forecasting.
            - Facilitate report generation with summaries, plots, and exportable results.
            """)

        with summary_tabs[1]:
            st.markdown("""
            The application executes a staged workflow:

            1. Load and clean oil production data, removing outliers using LOF.
            2. Apply rolling statistical models and LOESS smoothing to estimate standard deviations.
            3. Perform Monte Carlo sampling to generate synthetic production scenarios.
            4. Fit selected decline curve models using multi-start optimization.
            5. Analyze training and hindcast fits, forecast future production.
            6. Estimate EUR per model and combine forecasts probabilistically.
            """)

        with summary_tabs[2]:
            st.markdown("""
            The following models are implemented:

            - **Arps**: Classical exponential/hyperbolic model.
            - **Stretched Exponential Model (SEM)**: Generalized production decay.
            - **Capacitance Resistance Model (CRM)**: Physics-based model accounting for reservoir connectivity.
            - **Logistic Growth Model (LGM)**: Sigmoid-shaped production forecasting.
            """)

        st.markdown("</div>", unsafe_allow_html=True)

# ✅ Sidebar navigation + Footer note
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("➡️ Go to **Pipeline Run** to start analysis.")
st.sidebar.markdown("➡️ Go to **Generate Report** to create your final report.")
st.sidebar.markdown("---")
st.sidebar.markdown("By: Alexis Ortega")
st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/alexort74/probabilistic_dca)")
