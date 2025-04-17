import streamlit as st
from PIL import Image

st.set_page_config(page_title="Probabilistic DCA", layout="wide")

# ─────────────── Banner ───────────────
banner_col1, banner_col2 = st.columns([1, 2])
with banner_col1:
    logo = Image.open("images/dca_workflow_2.png")
    st.image(logo, use_container_width=True)

with banner_col2:
    st.title("📈 Probabilistic Decline Curve Analysis (DCA)")
    st.markdown(
        """
        **Automate** uncertainty‑aware oil‑production forecasting based on SPE‑194503‑PA’s Monte‑Carlo+multi‑model approach.
        """
    )
    st.markdown(
        """
        **Key features:**  
        - 🔄  Staged pipeline: data → sampling → fitting → EUR  
        - 🧠  Multi‑model (ARPS, SEM, CRM, LGM)  
        - 📊  Real‑time UI & exportable CSV/PDF  
        """
    )

    # ───── Tabs under Key Features ─────
    tabs = st.tabs([
        "🎯 Project Goals",
        "🧪 Methodology",
        "📚 Models",
        "⚙️ Tech Details"
    ])

    with tabs[0]:  # Project Goals
        st.markdown(
            """
            - **Integrate model uncertainty** rather than a single “best” model. By treating each model’s goodness‑of‑fit as a probability, you weight forecasts by how likely each model truly represents the underlying physics. 
            - **Propagate measurement & model uncertainty** into production forecasts via Monte Carlo + Bayesian updating. This yields a single probabilistic forecast that inherently accounts for errors in the data and ambiguity in model choice.  
            - **Mitigate over/under‑estimations** that arises when relying on one model. The combined multi‑model forecast reduces the risk of “precisely wrong” point estimates by generating a “vaguely right” distribution.  
            - **Validate on field data**. Show that (a) no single model dominates for all wells, and (b) adding more data tightens uncertainty and improves hindcast performance.  
            """
        )

    with tabs[1]:  # Methodology
        st.markdown(
            """
            1. **Estimate measurement errors**    
                - Use LOESS (“rlowess”) + rolling‑window SD to derive point‑wise standard deviations (rₖ) for each rate datapoint
            2. **Monte Carlo sampling**
               - At each time step, draw N synthetic rate values from 𝒩(q̂, rk), then sort across samples to preserve marginal distributions without bias.
            3. **History matching via (MLE)**  
               - For each sampled dataset, fit each candidate model by minimizing weighted SSE, i.e. Lₘₗₑ(x)=∑(q_model−q_data)²/rk².
            4. **Bayesian model probabilities**   
               - Compute each model’s posterior probability P(m | data) ∝ exp(–½ Lₘₗₑ) normalized across models.
            5. **Forecast aggregation**
               - Weight each model’s forecast by its marginal probability and combine to get a single probabilistic forecast (MM‑P) that integrates both intrinsic and model uncertainty. 
            6. **Analyze & report**
               - Extract percentiles (P10/P50/P90), means, and full EUR distributions. Validate via hindcasts on unseen data and apply to field examples.    
            """
        )

    with tabs[2]:  # Models
        st.markdown(
            """
            **These four models provide a balance of empirical simplicity and physics‑based insight—more can be added as needed.**
            - **Arps (exp/hyp)**:   
                qₜ = q₀(1 + b Dᵢ t)^(-1/b), 0≤b<1      
                Classic, empirical, fast; may mis‑represent multi‑regime fracture flow ​    
            
            - **Stretched Exponential (SEM):**  
                qₜ = q₀ exp[−(t/s)^n]     
                Captures a distribution of characteristic times; fat‑tailed declines   
            
            - **Logistic Growth (LGM):**    
                qₜ = a K t^(g−1)/(a + t^g)^2     
                Sigmoidal, directly estimates carrying capacity (EUR) K   
            - **Pan CRM:**    
                qₜ = ΔP/(b√t + J₁)·exp[−(2 b√t + J₁ t)/(cₜ Vₚ)]     
                Physics‑based, handles linear‑to‑boundary flow; skip first 10 days to avoid singularity   
              
            """
        )

    with tabs[3]:  # Tech Details (unchanged)
        st.markdown("_View package structure, config, and extended technical notes below._")
        with st.expander("Folder structure & config overview", expanded=False):
            st.code(
                """
project_root/
├─ app/pages/1_Pipeline_Run.py
├─ app/pages/2_Generate_Report.py
├─ images/dca_workflow_2.png
├─ src/probabilistic_dca/
│  ├─ my_dca_models/
│  ├─ config.py
│  └─ logging_setup.py
└─ tests/
                """, language="bash"
            )
            st.markdown(
                """
                - **config.py** holds defaults (n_inits, thresholds…)  
                - **logging_setup** for structured logs  
                - **tests/** cover data & fitting modules  
                """
            )

# ─────────────── Sidebar ───────────────
st.sidebar.title("🚀 Jump to")
st.sidebar.markdown("- **Pipeline Run**")  
st.sidebar.markdown("- **Generate Report**")
st.sidebar.markdown("---")
st.sidebar.markdown("By: Alexis Ortega")
st.sidebar.markdown("[🔗 Source on GitHub](https://github.com/alexort74/probabilistic_dca)")


