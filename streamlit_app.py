"""
Wrapper entrypoint for Streamlit Cloud compatibility.
Some deployment UIs expect a file named `streamlit_app.py`.
This file simply imports `app.py` so the app runs the same.
"""
from app import *
