"""
app.py — entry point shim
==========================
This file exists only so that the standard command:

    streamlit run app.py

continues to work after the refactor.
All application logic lives in app/main.py.
"""

from app.main import main

main()