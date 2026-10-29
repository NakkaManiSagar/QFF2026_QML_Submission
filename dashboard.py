
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS_FILE = ROOT / "predictions_q2.csv"

st.set_page_config(
    page_title="QFF 2026 | Quantum ML",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(29, 78, 137, 0.22),
                transparent 35%
            ),
            linear-gradient(145deg, #09111f 0%, #101b2d 100%);
        color: #e8eef8;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 17, 31, 0.75);
    }

    [data-testid="stSidebar"] {
        background: #0a1424;
        border-right: 1px solid #263a54;
    }

    [data-testid="stSidebar"] * {
        color: #e1eaf8;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    h1 {
        color: #f2f7ff !important;
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.7px;
    }

    h2 {
        color: #c9ddff !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #b9d4fa !important;
        font-weight: 650 !important;
    }

    p, label {
        color: #d3deed;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(140deg, #17283e, #122238);
        border: 1px solid #304966;
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 7px 22px rgba(0, 0, 0, 0.16);
        min-height: 115px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfdc !important;
    }

    [data-testid="stMetricValue"] {
        color: #73d7ff !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #9ee6c1 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #2a405b;
        border-radius: 11px;
        overflow: hidden;
    }

    [data-testid="stVegaLiteChart"],
    [data-testid="stPlotlyChart"] {
        background: #111f32;
        border: 1px solid #293e59;
        border-radius: 12px;
        padding: 8px;
    }

    .hero {
        background: linear-gradient(
            120deg,
            rgba(25, 54, 87, 0.95),
            rgba(17, 37, 62, 0.90)
        );
        border: 1px solid #355777;
        border-radius: 18px;
        padding: 25px 28px;
        margin: 10px 0 24px 0;
    }

    .hero-title {
        color: #f3f8ff;
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #b6cbe5;
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-note {
        color: #9fb4ce;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .status-card {
        background: #132238;
        border: 1px solid #2a405a;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }

    .stButton button,
    .stDownloadButton button {
        background: #16446b;
        border: 1px solid #3d83b4;
        color: #ffffff;
        border-radius: 9px;
        font-weight: 650;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background: #1c5987;
        border-color: #76d9ff;
        color: #ffffff;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #2a405a;
    }

    code {
        color: #90ddff;
    }

    footer {
        visibility: hidden;
    }

    
/* Keep the app and main content inside the viewport */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #0b1120 !important;
    color: #e8eef8 !important;
    max-width: 100%;
}

[data-testid="stMainBlockContainer"] {
    padding-left: clamp(1rem, 3vw, 2.5rem);
    padding-right: clamp(1rem, 3vw, 2.5rem);
}

/* Sidebar width and wrapping */
[data-testid="stSidebar"] {
    min-width: 235px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    white-space: normal !important;
    overflow-wrap: anywhere;
}

/* Consistent text colors */
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: #dce7f5;
}

.stApp h1,
.stApp h2,
.stApp h3 {
    color: #f0f6ff !important;
}

/* Inputs, selectors and expanders */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] > div {
    background-color: #142238 !important;
    color: #e8eef8 !important;
}

[data-testid="stExpander"] {
    background: #111f32 !important;
    border: 1px solid #30445e !important;
    border-radius: 10px;
}

[data-testid="stAlert"] {
    background-color: #15263b !important;
    color: #e8eef8 !important;
}

/* Avoid excessive horizontal overflow */
[data-testid="stDataFrame"] {
    max-width: 100%;
}



    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA HELPERS
# =========================================================
def load_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
    return pd.DataFrame()


def load_json(path):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
    return {}


def show_table(df, title):
    st.subheader(title)

    if df.empty:
        st.info("No saved data is available for this table.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


def find_number(df, column, model_contains=None):
    if df.empty or column not in df.columns:
        return None

    rows = df.copy()

    if model_contains and "model" in rows.columns:
        rows = rows[
            rows["model"].astype(str).str.contains(
                model_contains,
                case=False,
                regex=False,
            )
        ]

    if rows.empty:
        return None

    try:
        value = float(rows.iloc[0][column])
        return value if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value, decimals=3):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def show_image(filename, caption=None):
    path = FIGURES / filename

    if path.exists():
        st.image(
            str(path),
            caption=caption or filename,
            use_container_width=True,
        )
        return True

    return False


def show_page_header(title, subtitle):
    st.header(title)
    st.markdown(
        f'<p class="section-note">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# =========================================================
# LOAD SAVED RESULTS
# =========================================================
final_comparison = load_csv(TABLES / "final_comparison.csv")
classical = load_csv(TABLES / "classical_baselines.csv")

noisy_comparison = load_csv(
    TABLES / "noisy_kernel_accuracy_comparison.csv"
)

processor_d = load_csv(TABLES / "processor_d_candidates.csv")
task_a = load_csv(TABLES / "task_a.csv")

predictions = load_csv(PREDICTIONS_FILE)

noisy_diagnostics = load_json(
    TABLES / "noisy_kernel_diagnostics.json"
)

final_selection = load_json(
    RESULTS / "final_model_selection.json"
)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ⚛️ QFF 2026")
st.sidebar.caption("Quantum ML Research Dashboard")
st.sidebar.divider()

page_options = {
    "Overview": "Executive Summary",
    "Model Results": "Model Comparison",
    "Kernel & Noise": "Quantum Kernel & Noise",
    "Task A: Bell State": "Task A — Bell State",
    "Processor D": "Task B — Processor D",
    "Predictions": "Test Predictions",
    "Figures": "Saved Figures",
}

selected_page = st.sidebar.radio(
    "EXPLORE DASHBOARD",
    list(page_options.keys()),
)

page = page_options[selected_page]

st.sidebar.divider()
st.sidebar.caption("Problem statement: PS-Q2")
st.sidebar.caption("High-dimensional kernel classification")
st.sidebar.caption("Results loaded from local project files")


# =========================================================
# HERO BANNER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            ⚛️ Quantum Machine Learning
        </div>
        <div class="hero-subtitle">
            Qiskit Fall Fest 2026 · PS-Q2<br>
            High-Dimensional Quantum-Kernel Classification<br>
            Classical baselines · Quantum kernels · Noise analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1: EXECUTIVE SUMMARY
# =========================================================
if page == "Executive Summary":

    show_page_header(
        "Executive Summary",
        "A high-level view of the saved model evaluations, "
        "quantum experiments, and submission artifacts.",
    )

    classical_ba = find_number(
        classical,
        "balanced_accuracy_mean",
        "RBF-SVM all 50 features",
    )

    ideal_ba = find_number(
        noisy_comparison,
        "balanced_accuracy",
        "ideal_kernel",
    )

    noisy_ba = find_number(
        noisy_comparison,
        "balanced_accuracy",
        "noisy_kernel_psd_clipped",
    )

    noisy_fidelity = find_number(
        task_a,
        "fidelity",
    )

    prediction_count = (
        len(predictions) if not predictions.empty else None
    )

    # Main metric cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Classical RBF-SVM",
        format_percent(classical_ba),
    )

    c2.metric(
        "Ideal Kernel",
        format_percent(ideal_ba),
    )

    c3.metric(
        "Noisy Kernel",
        format_percent(noisy_ba),
    )

    c4.metric(
        "Test Predictions",
        f"{prediction_count:,}"
        if prediction_count is not None
        else "N/A",
    )

    st.caption(
        "Evaluation protocols differ: classical metrics are from "
        "saved cross-validation results; kernel metrics are from "
        "a separate subset experiment."
    )

    st.divider()

    # Status cards
    st.subheader("Project Artifacts")

    status_items = [
        ("Classical baseline results", not classical.empty),
        ("Final model comparison", not final_comparison.empty),
        ("Quantum kernel comparison", not noisy_comparison.empty),
        ("Task A measurements", not task_a.empty),
        ("Processor D benchmark", not processor_d.empty),
        ("Prediction file", not predictions.empty),
    ]

    status_columns = st.columns(3)

    for index, (label, available) in enumerate(status_items):
        with status_columns[index % 3]:
            symbol = "✅" if available else "⚠️"
            message = "Available" if available else "Not found"

            st.markdown(
                f"""
                <div class="status-card">
                    <strong>{symbol} {label}</strong><br>
                    <span class="section-note">{message}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Model comparison chart
    st.subheader("Cross-Validation Model Comparison")

    if (
        not final_comparison.empty
        and {"model", "balanced_accuracy_cv"}.issubset(
            final_comparison.columns
        )
    ):
        chart_data = final_comparison[
            ["model", "balanced_accuracy_cv"]
        ].copy()

        chart_data["balanced_accuracy_cv"] = (
            pd.to_numeric(
                chart_data["balanced_accuracy_cv"],
                errors="coerce",
            )
            * 100
        )

        chart_data = chart_data.dropna().set_index("model")

        st.bar_chart(
            chart_data,
            y="balanced_accuracy_cv",
            x_label="Model",
            y_label="Balanced accuracy (%)",
        )
    else:
        st.info("Saved final comparison chart data is unavailable.")

    show_table(
        final_comparison,
        "Saved final comparison results",
    )

    st.info(
        "Interpretation note: the ideal/noisy kernel subset scores "
        "must not be presented as a direct replacement for the "
        "final model's cross-validation results."
    )


# =========================================================
# PAGE 2: MODEL COMPARISON
# =========================================================
elif page == "Model Comparison":

    show_page_header(
        "Model Comparison",
        "Review saved classical baselines and final model-selection "
        "results using their documented evaluation protocols.",
    )

    # Compact summary table: show the most useful metrics first.
    st.subheader("Classical Baseline Results")

    if classical.empty:
        st.info("No classical baseline results are available.")
    else:
        compact_columns = [
            "model",
            "pca_components",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "f1_mean",
            "f1_std",
        ]
        available_columns = [
            column for column in compact_columns
            if column in classical.columns
        ]
        compact = classical[available_columns].copy()

        # Display metrics as percentages for readability.
        for column in [
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "f1_mean",
            "f1_std",
        ]:
            if column in compact.columns:
                compact[column] = pd.to_numeric(
                    compact[column], errors="coerce"
                ).map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
                )

        st.dataframe(
            compact,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View complete baseline details"):
            st.caption(
                "Includes fold-level metrics and saved model parameters. "
                "Use the horizontal scrollbar to inspect all columns."
            )
            st.dataframe(
                classical,
                use_container_width=True,
                hide_index=True,
            )

    if (
        not classical.empty
        and {"model", "balanced_accuracy_mean"}.issubset(
            classical.columns
        )
    ):
        chart_data = classical[
            ["model", "balanced_accuracy_mean"]
        ].copy()
        chart_data["balanced_accuracy_mean"] = (
            pd.to_numeric(
                chart_data["balanced_accuracy_mean"],
                errors="coerce",
            )
            * 100
        )
        chart_data = chart_data.dropna().set_index("model")

        st.subheader("Classical Balanced Accuracy")
        st.bar_chart(
            chart_data,
            y="balanced_accuracy_mean",
            x_label="Model",
            y_label="Balanced accuracy (%)",
        )

    if (
        not classical.empty
        and {"model", "f1_mean"}.issubset(classical.columns)
    ):
        f1_data = classical[["model", "f1_mean"]].copy()
        f1_data["f1_mean"] = (
            pd.to_numeric(f1_data["f1_mean"], errors="coerce") * 100
        )
        f1_data = f1_data.dropna().set_index("model")

        st.subheader("Classical F1 Scores")
        st.bar_chart(
            f1_data,
            y="f1_mean",
            x_label="Model",
            y_label="F1 score (%)",
        )

    st.divider()
    st.subheader("Final Model Selection")

    if final_selection:
        st.json(final_selection)
    else:
        st.info("No saved final model-selection JSON was found.")

    show_table(
        final_comparison,
        "Final Cross-Validation Comparison",
    )

    st.warning(
        "Use the documented evaluation protocol when comparing models. "
        "Subset experiments and full cross-validation results may not "
        "be directly comparable."
    )


# =========================================================
# PAGE 3: QUANTUM KERNEL & NOISE
# =========================================================
elif page == "Quantum Kernel & Noise":

    show_page_header(
        "Quantum Kernel & Noise",
        "Explore saved ideal and noisy kernel results and kernel-matrix "
        "diagnostics from the experiment.",
    )

    show_table(
        noisy_comparison,
        "Ideal and Noisy Kernel Classification Results",
    )

    if (
        not noisy_comparison.empty
        and {"model", "balanced_accuracy"}.issubset(
            noisy_comparison.columns
        )
    ):
        chart_data = noisy_comparison[
            ["model", "balanced_accuracy"]
        ].copy()

        chart_data["balanced_accuracy"] = (
            pd.to_numeric(
                chart_data["balanced_accuracy"],
                errors="coerce",
            )
            * 100
        )

        chart_data = chart_data.dropna().set_index("model")

        st.subheader("Kernel Balanced Accuracy")

        st.bar_chart(
            chart_data,
            y="balanced_accuracy",
            x_label="Experiment",
            y_label="Balanced accuracy (%)",
        )

    if (
        not noisy_comparison.empty
        and {"model", "f1"}.issubset(noisy_comparison.columns)
    ):
        f1_data = noisy_comparison[["model", "f1"]].copy()

        f1_data["f1"] = (
            pd.to_numeric(f1_data["f1"], errors="coerce") * 100
        )

        f1_data = f1_data.dropna().set_index("model")

        st.subheader("Kernel F1 Scores")

        st.bar_chart(
            f1_data,
            y="f1",
            x_label="Experiment",
            y_label="F1 score (%)",
        )

    st.divider()
    st.subheader("Kernel Diagnostics")

    if noisy_diagnostics:
        st.json(noisy_diagnostics)
    else:
        st.info("No saved noisy-kernel diagnostics were found.")

    st.subheader("Saved Kernel Visualizations")

    kernel_figures = [
        (
            "kernel_matrices_ideal_vs_noisy.png",
            "Ideal versus noisy kernel matrices",
        ),
        (
            "kernel_noisy_vs_ideal_scatter.png",
            "Noisy versus ideal kernel entries",
        ),
        (
            "ideal_kernel_eigenvalue_spectrum.png",
            "Ideal kernel eigenvalue spectrum",
        ),
        (
            "ideal_kernel_heatmap_k6.png",
            "Ideal kernel heatmap",
        ),
    ]

    found_figure = False

    for filename, caption in kernel_figures:
        if show_image(filename, caption):
            found_figure = True

    if not found_figure:
        st.info("No saved kernel figures were found.")

    st.caption(
        "Kernel scores shown here are from a subset experiment. "
        "Noise, shot count, preprocessing, and PSD correction can "
        "affect the resulting kernel and classifier."
    )


# =========================================================
# PAGE 4: TASK A — BELL STATE
# =========================================================
elif page == "Task A — Bell State":

    show_page_header(
        "Task A — Bell-State Measurements",
        "Saved ideal and noisy correlation measurements and "
        "fidelity estimates.",
    )

    show_table(
        task_a,
        "Saved Task A Measurements",
    )

    if not task_a.empty and "condition" in task_a.columns:

        for condition in task_a["condition"].dropna().unique():

            row = task_a[
                task_a["condition"].astype(str) == str(condition)
            ].iloc[0]

            st.subheader(f"{str(condition).title()} Condition")

            cols = st.columns(4)

            metric_names = ["XX", "YY", "ZZ", "fidelity"]

            for col, name in zip(cols, metric_names):
                if name in row.index:
                    try:
                        value = float(row[name])
                        col.metric(name, f"{value:.4f}")
                    except (ValueError, TypeError):
                        col.metric(name, str(row[name]))

            if "shots" in row.index:
                st.caption(f"Shots: {row['shots']}")

    st.info(
        "Ideal and noisy results are displayed separately. "
        "Noisy measurements are affected by finite-shot sampling "
        "and the configured noise model."
    )


# =========================================================
# PAGE 5: TASK B — PROCESSOR D
# =========================================================
elif page == "Task B — Processor D":

    show_page_header(
        "Task B — Processor D",
        "Review QFT-3 transpilation costs, noisy fidelity, and "
        "kernel circuit metrics for the saved candidates.",
    )

    if processor_d.empty:
        st.info("No Processor D candidate results were found.")
    else:

        # Show metrics for the first saved candidate
        first = processor_d.iloc[0]

        metric_specs = [
            ("n_qubits", "Qubits", False),
            (
                "qft3_circuit_depth_after_transpile",
                "QFT-3 Depth",
                False,
            ),
            (
                "qft3_two_qubit_gate_count",
                "QFT-3 Two-Qubit Gates",
                False,
            ),
            (
                "qft3_fidelity_noisy",
                "Noisy QFT-3 Fidelity",
                True,
            ),
        ]

        cols = st.columns(4)

        for col, (field, label, is_percentage) in zip(
            cols,
            metric_specs,
        ):
            if field not in processor_d.columns:
                col.metric(label, "N/A")
                continue

            try:
                value = float(first[field])
                display = (
                    format_percent(value)
                    if is_percentage
                    else format_number(value, 0)
                )
                col.metric(label, display)
            except (TypeError, ValueError):
                col.metric(label, "N/A")

        st.caption(
            "The cards above summarize the first saved candidate. "
            "Use the full table below to inspect every candidate."
        )

        show_table(
            processor_d,
            "Processor D Candidate Results",
        )

        circuit_columns = [
            "processor_id",
            "topology",
            "n_qubits",
            "qft3_circuit_depth_before_transpile",
            "qft3_circuit_depth_after_transpile",
            "qft3_two_qubit_gate_count",
            "qft3_swap_overhead",
            "qft3_fidelity_ideal",
            "qft3_fidelity_noisy",
            "kernel_depth_after",
            "kernel_two_qubit_gates_after_prepare",
        ]

        available_columns = [
            name
            for name in circuit_columns
            if name in processor_d.columns
        ]

        if available_columns:
            st.subheader("Circuit Cost Details")
            st.dataframe(
                processor_d[available_columns],
                use_container_width=True,
                hide_index=True,
            )

        if (
            {"processor_id", "qft3_fidelity_noisy"}.issubset(
                processor_d.columns
            )
        ):
            fidelity_data = processor_d[
                ["processor_id", "qft3_fidelity_noisy"]
            ].copy()

            fidelity_data["qft3_fidelity_noisy"] = (
                pd.to_numeric(
                    fidelity_data["qft3_fidelity_noisy"],
                    errors="coerce",
                )
                * 100
            )

            fidelity_data = fidelity_data.dropna().set_index(
                "processor_id"
            )

            st.subheader("Noisy QFT-3 Fidelity by Candidate")

            st.bar_chart(
                fidelity_data,
                y="qft3_fidelity_noisy",
                x_label="Processor candidate",
                y_label="Noisy fidelity (%)",
            )

        show_image(
            "processor_d_candidates.png",
            "Processor D Candidate Analysis",
        )

    st.caption(
        "QFT-3 benchmark costs and kernel-circuit costs are different "
        "measurements. The dashboard keeps their fields separate."
    )


# =========================================================
# PAGE 6: TEST PREDICTIONS
# =========================================================
elif page == "Test Predictions":

    show_page_header(
        "Test Predictions",
        "Inspect and download the saved predictions for the unlabeled "
        "test dataset.",
    )

    if predictions.empty:
        st.error(
            "predictions_q2.csv was not found or could not be read."
        )
    else:
        st.metric("Prediction Rows", f"{len(predictions):,}")

        # Optional prediction-column distribution
        possible_label_columns = [
            "prediction",
            "predicted_label",
            "label",
            "target",
            "class",
        ]

        label_column = next(
            (
                column
                for column in possible_label_columns
                if column in predictions.columns
            ),
            None,
        )

        if label_column:
            st.subheader("Predicted Label Distribution")

            distribution = (
                predictions[label_column]
                .astype(str)
                .value_counts()
                .rename_axis("Predicted label")
                .to_frame("Count")
            )

            st.bar_chart(distribution)

        st.subheader("Prediction Table")

        search = st.text_input(
            "Search prediction rows",
            placeholder="Enter a value to filter the table...",
        )

        visible_predictions = predictions.copy()

        if search.strip():
            mask = visible_predictions.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    regex=False,
                    na=False,
                )
            ).any(axis=1)

            visible_predictions = visible_predictions[mask]

        st.dataframe(
            visible_predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download predictions_q2.csv",
            data=PREDICTIONS_FILE.read_bytes(),
            file_name="predictions_q2.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "These are predictions for unlabeled test rows. "
            "Their true labels and hidden-test performance are "
            "not available from this prediction file."
        )


# =========================================================
# PAGE 7: SAVED FIGURES
# =========================================================
elif page == "Saved Figures":

    show_page_header(
        "Experiment Figures",
        "Browse the PNG visualizations generated by your saved "
        "experiments.",
    )

    if not FIGURES.exists():
        st.info("The figures directory does not exist.")
    else:
        images = sorted(FIGURES.glob("*.png"))

        if not images:
            st.info("No PNG figures were found.")
        else:
            st.caption(f"{len(images)} saved PNG figure(s) found.")

            for path in images:
                with st.expander(path.stem.replace("_", " ").title()):
                    st.image(
                        str(path),
                        caption=path.name,
                        use_container_width=True,
                    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0 20px 0;">
        <span style="color:#83bfe8; font-weight:700;">
            QFF 2026 · PS-Q2
        </span>
        <br>
        <span style="color:#94a9c2; font-size:0.85rem;">
            Quantum Machine Learning Research Dashboard
            · Local experiment artifacts
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
