######################################################################
# “report_generator”
######################################################################
from docx import Document
from docx.shared import Inches
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd


def add_heading(document, text, level=1):
    document.add_heading(text, level=level)


def add_paragraph(document, text):
    document.add_paragraph(text)


def add_dataframe(document, df, title=None):
    if title:
        add_heading(document, title, level=2)
    table = document.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, column in enumerate(df.columns):
        hdr_cells[i].text = str(column)
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)


def add_plot(document, fig, title):
    add_heading(document, title, level=2)
    image_stream = BytesIO()
    fig.savefig(image_stream, format='png', bbox_inches='tight')
    image_stream.seek(0)
    document.add_picture(image_stream, width=Inches(6))
    plt.close(fig)


def generate_report(pipeline_results):
    document = Document()

    # Title
    document.add_heading('Probabilistic Decline Curve Analysis Report', 0)

    # Section 1: Overview
    add_heading(document, "1. Overview", level=1)
    add_paragraph(document, "This report summarizes the results of the probabilistic decline curve analysis, including model fitting, probabilistic forecasts, and estimated ultimate recovery (EUR) analysis.")

    # Section 2: Initial Data & Outlier Detection
    add_heading(document, "2. Initial Production Data & Outlier Detection", level=1)
    add_plot(document, pipeline_results['lof_plot'].figure, "Initial Production Data with LOF Outlier Detection")

    # Section 3: Model Probabilities
    add_heading(document, "3. Marginal Posterior Probabilities of Models", level=1)
    add_plot(document, pipeline_results['prob_plot'].figure, "Model Posterior Probabilities")

    # Section 4: Model-specific EUR
    add_heading(document, "4. Model-Specific EUR Statistics", level=1)
    model_eur_df = pd.DataFrame(pipeline_results['model_eur_stats']).T
    
    model_eur_df = model_eur_df.applymap(lambda x: f"{int(round(x, 0)):,}")
    
    desired_columns = ["p10", "p50", "mean", "p90"]
    model_eur_df_clean = model_eur_df[desired_columns].reset_index()
    model_eur_df_clean.rename(columns={'index': 'Model'}, inplace=True)
    
    add_dataframe(document, model_eur_df_clean, title="Per Model EUR Summary")

    # Section 5: Combined Forecast EUR
    add_heading(document, "5. Combined EUR Statistics", level=1)
    combined_eur_stats_df = pd.DataFrame([pipeline_results['combined_eur_stats']])
    combined_eur_stats_df_clean = combined_eur_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
    
    add_dataframe(document, combined_eur_stats_df_clean, title="Combined Model EUR Summary")

    # Conclusion
    add_heading(document, "6. Conclusion", level=1)
    add_paragraph(document, "The analysis demonstrates the range of production forecasts and uncertainties associated with the selected decline curve models. Multi-model probabilistic forecasts provide a robust outlook for future production.")

    # Save document to BytesIO
    doc_io = BytesIO()
    document.save(doc_io)
    doc_io.seek(0)

    return doc_io
