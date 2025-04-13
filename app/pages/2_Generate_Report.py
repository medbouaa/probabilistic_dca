import streamlit as st
import sys
import pathlib

# Add the src directory to the Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from probabilistic_dca.reporting.report_generator import generate_report
from probabilistic_dca.logging_setup import setup_logger

# Logger
logger = setup_logger(__name__)

# ✅ Caching functions
@st.cache_data
def cached_generate_report(_pipeline_results):
    return generate_report(_pipeline_results)

# Page config
st.set_page_config(page_title="Generate Report", layout="wide")
st.title("📄 Generate Report")

# ✅ Sidebar cache control
st.sidebar.header("Utilities")
if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Reloading app...")
    st.experimental_rerun()

st.markdown("""
Generate a full professional `.docx` report based on your pipeline results.
""")

# Check if pipeline has been run
if 'pipeline_results' not in st.session_state or st.session_state.pipeline_results is None:
    st.warning("Please run the pipeline first to generate a report.")
    logger.warning("Attempted to generate report without pipeline results.")
else:
    if st.button("📄 Generate Full Report"):
        with st.spinner("Generating report..."):
            try:
                report_file = cached_generate_report(st.session_state.pipeline_results)
                st.success("Report generated successfully!")
                logger.info("Report generated successfully.")

                # Offer download
                st.download_button(
                    label="📥 Download Report (.docx)",
                    data=report_file.getvalue(),
                    file_name="Probabilistic_DCA_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error("An error occurred while generating the report.")
                logger.exception(f"Error generating report: {e}")