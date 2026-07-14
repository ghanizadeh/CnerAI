"""
app.py — Entry point for the CnerAI ML Dashboard.

Run with:
    streamlit run app.py

Architecture (post-refactor)
-----------------------------
Model
├── Regression
│   ├── Train      (Model Selection + Validation Method + Training Output)
│   ├── Evaluate
│   ├── Predict
│   └── 🟢 Safe Region & Optimizer  (NEW)
│
└── Classification
    ├── Train      (Model Selection + Validation Method + Training Output)
    ├── Evaluate
    ├── Predict
    └── 🟢 Safe Region & Optimizer  (NEW)

Note: Train/Test Split has been REMOVED from Preprocessing.
      It now lives inside Train → Validation Method.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from config.settings import APP_TITLE, APP_ICON, LOGO_PATH
from state.session import init_state, clear_state, pipeline_status

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    try:

        st.image("logo.png", use_container_width=True)
    except Exception:
        st.title(f"{APP_ICON} Cnergreen ML Dashboard")

    st.markdown("---")

    main_menu = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data", "EDA", "Preprocessing", "Model", "Extra Tool"],
        icons=["house-door", "folder2-open", "bar-chart-line", "gear", "robot", "tools"],
        menu_icon="house",
        default_index=0,
        key="main_menu",
        styles={
            "container": {"padding": "0px"},
            "nav-link-selected": {"font-weight": "bold"},
        },
    )

    sub_selection = None

    if main_menu == "Model":
        st.markdown("#### Model Steps")
        sub_selection = option_menu(
            menu_title=None,
            options=[
                "Train",
                "Evaluate",
                "Predict",
                "Safe Region & Optimizer",
            ],
            icons=[
                "play-circle",
                "clipboard-data",
                "magic",
                "shield-check",
            ],
            default_index=0,
            key="model_sub_menu",
            styles={
                "container": {"padding": "0px"},
                "nav-link-selected": {"font-weight": "bold"},
            },
        )

        # st.markdown(f"#### {model_type} Steps")
        # sub_selection = option_menu(
        #     menu_title=None,
        #     options=["Train", "Evaluate", "Predict", "Safe Region & Optimizer"],
        #     icons=["play-circle", "clipboard-data", "magic", "lightbulb", "shield-check"],
        #     default_index=0,
        #     key=f"model_sub_menu_{model_type}",
        #     styles={
        #         "container": {"padding": "0px"},
        #         "nav-link-selected": {"font-weight": "bold"},
        #     },
        # )

    elif main_menu == "Extra Tool":
        st.markdown("#### Tools")
        sub_selection = option_menu(
            menu_title=None,
            options=[
                "Foam Performance Calculator",
                "Foam Half-Life Calculator",
                "Foam Stability (14/30 days)",
                "Automatic 2D/3D Plotter",
                # "Data Extraction",
                # "Automatic AI Report",
                "Automatic AI Report (Validated)",
            ],
            icons=[
                "calculator",
                "calculator",
                "calculator",
                "graph-up",
                # "file-earmark-spreadsheet",
                # "file-text",
                "shield-check",
            ],
            default_index=0,
            key="extra_tool_menu",
            styles={
                "container": {"padding": "0px"},
                "nav-link-selected": {"font-weight": "bold"},
            },
        )

    st.markdown("---")

    if st.button("🔄 Reset Pipeline", use_container_width=True):
        clear_state()
        st.rerun()

    st.markdown("---")
    st.caption("© Cnergreen Platform")


# ── Routing ────────────────────────────────────────────────────────────────
if main_menu == "Data":
    from pages_content.page_data import render
    render()

elif main_menu == "EDA":
    from pages_content.page_eda import render
    render()

elif main_menu == "Preprocessing":
    from pages_content.page_preprocessing import render
    render()

elif main_menu == "Model":
    if sub_selection == "Train":
        from pages_content.page_train import render
        render()

    elif sub_selection == "Evaluate":
        from pages_content.page_evaluate import render
        render()

    elif sub_selection == "Predict":
        from pages_content.page_predict import render
        render()

    elif sub_selection == "Safe Region & Optimizer":
        from pages_content.page_safe_region import render
        render()

elif main_menu == "Extra Tool":
    if sub_selection == "Foam Performance Calculator":
        from pages_content.foam_performance_calculator import show_foam_performance_calculator
        show_foam_performance_calculator()
    if sub_selection == "Foam Half-Life Calculator":
        from pages_content.half_life_hr_calculator import render
        render()
    if sub_selection == "Foam Stability (14/30 days)":
        from pages_content.dilution_stability_calculator import render
        render()
    if sub_selection == "Automatic 2D/3D Plotter":
        from pages_content.Page_auto_eda import render
        render()
    # if sub_selection == "Data Extraction":
    #     from core.data.extraction.noOil_extraction import render
    #     render()
    # if sub_selection == "Automatic AI Report":
    #     from pages_content.page_auto_ml import render
    #     render()
    if sub_selection == "Automatic AI Report (Validated)":
        from pages_content.page_ai_report_validated import render
        render()
# ── Home ───────────────────────────────────────────────────────────────────
if main_menu == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except Exception:
            pass

    st.title(f"Welcome to {APP_TITLE}")
    st.markdown(
        """
        This application guides you through a **professional machine-learning workflow**:

        | Step | Page | Description |
        |------|------|-------------|
        | 1 | 📂 Data | Upload & preview your dataset |
        | 2 | 📊 EDA | Visualise distributions & correlations |
        | 3 | ⚙️ Preprocessing | Clean, encode & scale features |
        | 4 | 🤖 Train | Select model + validation strategy, then train |
        | 5 | 📈 Evaluate | Metrics, plots & model comparison |
        | 6 | 🔮 Predict | Single-row or batch predictions |
        | 8 | 🟢 Safe Region & Optimizer | Safe region search, Bayesian opt & active learning |

        Use the **left sidebar** to navigate between steps.
        """
    )

    st.divider()

    st.subheader("📊 Pipeline Status")
    status = pipeline_status()

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("📂 Data Loaded",   "✅ Yes" if status["data_loaded"]  else "❌ No")
        c2.metric("⚙️ Preprocessed",  "✅ Yes" if status["preprocessed"] else "❌ No")
        c3.metric("🤖 Model Trained", "✅ Yes" if status["model_trained"] else "❌ No")

    if not status["data_loaded"]:
        st.info("👈 Start by uploading your dataset in **📂 Data**.")
    elif not status["preprocessed"]:
        st.info("👉 Proceed to **⚙️ Preprocessing** to prepare your features.")
    elif not status["model_trained"]:
        st.info("🚀 Head to **🤖 Train** to fit a model.")
    else:
        st.success("🎉 Pipeline complete! Check **📈 Evaluate**, or **🟢 Safe Region & Optimizer**.")