import streamlit as st
from pathlib import Path
import json
# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
        with open("settings.json", "r") as f:
            st.session_state.settings = json.load(f)

# Initialize session state for workspace
if "chosen-workspace" not in st.session_state:
    if "workspace" in st.session_state:
        st.session_state["chosen-workspace"] = str(st.session_state.workspace.stem)
    else:
        st.session_state["chosen-workspace"] = "default"

if __name__ == '__main__':
    pages = {
        str(st.session_state.settings["app-name"]) : [
            st.Page(Path("content", "quickstart.py"), title="Quickstart", icon="👋"),
            st.Page(Path("content", "documentation.py"), title="Documentation", icon="📖"),
        ],
        "TOPP Workflow Framework": [
            st.Page(Path("content", "topp_workflow_file_upload.py"), title="File Upload", icon="📁"),
            st.Page(Path("content", "topp_workflow_parameter.py"), title="Configure", icon="⚙️"),
            st.Page(Path("content", "topp_workflow_execution.py"), title="Run", icon="🚀"),
            # st.Page(Path("content", "topp_workflow_results.py"), title="Results", icon="📊"),
        ],
        "Results": [
            st.Page(Path("content", "results_database_search.py"), title="Database Search", icon="🔬"),
            st.Page(Path("content", "results_rescoring.py"), title="Rescoring", icon="📈"),
            st.Page(Path("content", "results_filtered.py"), title="Filtered PSMs", icon="🎯"),
            st.Page(Path("content", "results_abundance.py"), title="Abundance", icon="📋"),
            st.Page(Path("content", "results_volcano.py"), title="Volcano", icon="🌋"),
            st.Page(Path("content", "results_pca.py"), title="PCA", icon="📊"),
            st.Page(Path("content", "results_heatmap.py"), title="Heatmap", icon="🔥"),
            # st.Page(Path("content", "results_library.py"), title="Spectral Library", icon="📚"),
            st.Page(Path("content", "results_pathway.py"), title="Pathway Analysis", icon="🧪"),
        ]
    }

    pg = st.navigation(pages)
    pg.run()
