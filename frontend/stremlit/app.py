"""
ADAS Streamlit Application Proxy.
Points directly to frontend/streamlit/app.py.
"""
from pathlib import Path
import runpy

target = Path(__file__).resolve().parents[1] / "streamlit" / "app.py"
runpy.run_path(str(target), run_name="__main__")
