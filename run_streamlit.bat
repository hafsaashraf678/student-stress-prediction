@echo off
set "PYTHONPATH=%CD%"
py -m streamlit run app.py --server.runOnSave true --server.fileWatcherType poll
