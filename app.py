"""
Root entrypoint for Streamlit application.
Usage:
    streamlit run app.py
"""
from pathlib import Path
import runpy

target = Path(__file__).resolve().parent / "frontend" / "streamlit" / "app.py"
runpy.run_path(str(target), run_name="__main__")
