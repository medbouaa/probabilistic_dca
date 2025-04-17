# 🛢️ Probabilistic Decline Curve Analysis (DCA) Pipeline

**Author:** Alexis Ortega  
**Project Status:** In development — production-ready Streamlit application for forecasting oil production using probabilistic models.

---

## 🚀 Project Overview

This project implements a complete end-to-end **Probabilistic Decline Curve Analysis (DCA)** workflow, based on the SPE-194503-PA methodology.

The goal is to provide **robust production forecasting** with full uncertainty quantification, allowing you to:

- Perform Monte Carlo simulations of oil production profiles.
- Fit multiple DCA models: **Arps, CRM, SEM, LGM**.
- Combine model outputs probabilistically.
- Generate professional `.docx` reports directly from the Streamlit app.
- Export visualizations, forecasts, and EUR statistics.

Built with:

- 🐍 Python 3.10.x (managed via **Pyenv**)
- 📦 Poetry for dependency and environment management
- 🚀 Streamlit for interactive UI
- 📊 Matplotlib for visualizations
- 📄 `python-docx` for automated report generation
- 🔥 Custom DCA models and parallelized fitting logic

---

## 🧩 Project Structure

project-root/ ├── app.py # Streamlit introduction page (project overview) ├── 1_Pipeline_Run.py # Main pipeline execution page ├── 2_Generate_Report.py # Report generation page ├── src/ │ └── probabilistic_dca/ │ ├── my_dca_models/ # Data processing, models, and plotting │ ├── reporting/ # Report generation scripts │ └── logging_setup.py # Logging configuration ├── images/ │ └── dca_workflow_2.png # Workflow diagram ├── poetry.lock # Poetry lock file ├── pyproject.toml # Poetry dependency definition └── README.md # Project documentation

---

## ⚙️ Features  

- **Multiple Models:** Arps, SEM, CRM, LGM — combine strengths of each.
- **Monte Carlo Sampling:** Generate thousands of synthetic production profiles.
- **Parallelized Model Fitting:** Fast execution with multiple initializations.
- **Visualization:** Clean Streamlit interface with interactive tabs.
- **Export Results:** Download CSV and `.docx` professional report.
- **Manual Cache Control:** Option to clear Streamlit cache for fresh runs.

---

## 💻 Installation

### 1. Install Python (with Pyenv)

Ensure you have Python 3.10.x installed via Pyenv:

```bash
pyenv install 3.10.4
pyenv local 3.10.4
```

### 2. Install Poetry

If you haven't installed Poetry yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Check Poetry version:

```bash
poetry --version
```

### 3. Install project dependencies

Install dependencies (Poetry will automatically create the virtual environment):

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

---

## 🚀 Running the Application

Step 1 — Start Streamlit inside Poetry shell

```bash
streamlit run app.py
```

Step 2 — Navigate between pages

Project Overview: Introduction and methodology.

Pipeline Run: Upload data, configure pipeline, execute forecasting.

Generate Report: Create and download .docx report.

➡️ You can navigate pages using the top-left menu inside the Streamlit app.

---

## 🗂️ Input Data Format

Upload your production data in .csv format.

Required columns:

Time Column (default: cum_eff_prod_day)

Rate Column (default: oil_month_bpd)

Cumulative Column (default: cum_oil_bbl)

Customize these names in the sidebar of the Pipeline page.

---

## 📦 Deployment

Option 1: Streamlit Cloud (Recommended for fast deployment)

1. Push your project to a public GitHub repository.

2. Connect your repo to Streamlit Cloud.

3. Define your main entry point as:

```bash
app.py
```

4. Deploy 🚀

Note: Poetry-managed projects work on Streamlit Cloud — ensure your pyproject.toml is complete!

Option 2: Docker (Optional for full control)

(Dockerfile can be provided on request!)

---

## 🧹 Cache Control

The application uses Streamlit cache to improve performance.

You can manually clear the cache anytime using the sidebar 🧹 "Clear Cache" button.

---

## 📈 Sample Output

✅ Outlier detection plot (LOF algorithm)

✅ Monte Carlo sample generation plot

✅ Model fit analysis and forecast plots

✅ Model probability bar charts

✅ Probabilistic EUR boxplots

✅ Professional .docx report with all visuals

---

## 🧭 Roadmap

*Add parallelization to fit_models() and enable caching.

*Optional Docker deployment.

*Expand multi-scenario analysis.

*Option to export full data outputs (forecasts, parameters).

---

## 🤝 Contributing

Contributions are welcome!
If you have suggestions or improvements, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Alexis Ortega
Senior Petroleum Engineer & Data Scientist
