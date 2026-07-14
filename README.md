# CnerAI ML Dashboard

A professional, modular Streamlit application for end-to-end machine learning workflows.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## Project Structure

```
ml_dashboard/
│
├── app.py                        # Entry point — page config + home screen
│
├── pages/                        # Streamlit multi-page routing
│   ├── 1_📂_Data.py             # Upload, preview, validate
│   ├── 2_📊_EDA.py              # Exploratory data analysis
│   ├── 3_⚙️_Preprocessing.py   # Clean, encode, scale, split
│   ├── 4_🤖_Train.py            # Model selection + training
│   ├── 5_📈_Evaluate.py         # Metrics, plots, model comparison
│   └── 6_🔮_Predict.py          # Single-row or batch predictions
│
├── core/                         # Pure Python — zero Streamlit here
│   ├── data/
│   │   ├── loader.py            # load_csv, load_excel, validate_df, extended_describe
│   │   └── preprocessor.py      # impute, outliers, encode, scale, split + EDA helpers
│   ├── models/
│   │   ├── registry.py          # Load models from config/models.yaml
│   │   ├── trainer.py           # train(), cross_validate_model()
│   │   └── evaluator.py         # get_metrics(), get_feature_importance()
│   └── viz/
│       ├── eda.py               # Correlation, boxplots, histograms, scatter, pairplot
│       ├── evaluation.py        # Confusion matrix, ROC, residuals, pred vs actual
│       └── style.py             # Shared palette, fig_to_st(), fig_to_bytes()
│
├── components/                   # Reusable Streamlit UI widgets
│   ├── column_selector.py       # render_column_selector(df) → (features, target)
│   ├── model_picker.py          # render_model_picker(task_type) → (name, params)
│   ├── metrics_card.py          # render_metrics_card(metrics_dict)
│   └── dataset_summary.py       # render_dataset_summary(df)
│
├── state/
│   └── session.py               # init_state, get_value, set_state, clear_state
│
├── config/
│   ├── settings.py              # App-wide constants (RANDOM_STATE, MAX_UPLOAD_MB …)
│   └── models.yaml              # All model classes, default params, param grids
│
├── tests/
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_trainer.py
│   └── test_evaluator.py
│
├── assets/
│   └── logo.png                 # (add your own)
│
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture Principles

**Strict layer separation**

| Layer | Location | Rule |
|-------|----------|------|
| Business logic | `core/` | Pure Python — no `import streamlit` |
| UI widgets | `components/` | Reusable `render_*()` functions |
| Pages | `pages/` | Thin orchestrators — call core + components |
| State | `state/session.py` | Single source of truth via dot-notation API |
| Config | `config/` | Constants and YAML — no logic |

**Reusing components across pages**

`column_selector` and EDA helpers (`extended_describe`, `categorical_summary` …) are
imported on both the EDA page and the Preprocessing page — no code duplication.

**Adding a new model**

1. Add an entry to `config/models.yaml` under `classification:` or `regression:`.
2. That's it — the registry, model picker, and trainer all pick it up automatically.

## Running Tests

```bash
# From the ml_dashboard/ directory
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=core --cov-report=term-missing
```

## Workflow

```
📂 Data → 📊 EDA → ⚙️ Preprocessing → 🤖 Train → 📈 Evaluate → 🔮 Predict
```

All intermediate results are stored in `st.session_state.ml` and persist across
page navigations within the same browser session.
