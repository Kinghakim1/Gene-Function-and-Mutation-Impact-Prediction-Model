import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Gene Function & Mutation Impact Prediction",
    page_icon="🧬",
    layout="wide",
)

# -------------------------------------------------
# Loaders
# -------------------------------------------------
@st.cache_data
def load_dataset():
    path = "cleaned_features.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    path = "models/function_rf.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def get_numeric_features(df: pd.DataFrame) -> list:
    """Return numeric columns only."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def prepare_input_from_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns for model prediction."""
    return row_df.select_dtypes(include=["number"])


# -------------------------------------------------
# Load assets
# -------------------------------------------------
df = load_dataset()
model = load_model()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🧬 Gene Function & Mutation Impact Prediction")
st.subheader("Interactive portfolio demo for exploring engineered sequence features and model-based prediction")

st.markdown(
    """
This app presents my **Gene Function and Mutation Impact Prediction** project.

It highlights:
- feature-engineered biological sequence data
- exploratory analysis across genes and organisms
- interactive row-level model prediction
- batch CSV prediction support for recruiter/demo use
"""
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Choose a section",
    ["Project Overview", "Explore Dataset", "Run Prediction", "Batch Prediction"]
)

# -------------------------------------------------
# Project Overview
# -------------------------------------------------
if view == "Project Overview":
    st.markdown("## Project Overview")

    st.write(
        """
This project focuses on using engineered biological sequence features to support
gene function and mutation impact analysis. The workflow includes data cleaning,
feature extraction, model development, and interactive visualization for easier interpretation.
"""
    )

    st.markdown("### Core pipeline")
    st.markdown(
        """
- Collected and cleaned gene-related sequence data  
- Engineered nucleotide, codon, k-mer, and amino acid composition features  
- Trained machine learning models for classification  
- Built a lightweight interactive interface for exploration and demo presentation  
"""
    )

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Unique Genes", df["Gene Name"].nunique() if "Gene Name" in df.columns else "N/A")
        col4.metric("Unique Organisms", df["Organism"].nunique() if "Organism" in df.columns else "N/A")

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.error("`cleaned_features.csv` was not found in the repo root.")

    if model is not None:
        st.success("Model loaded successfully from `models/function_rf.pkl`.")
    else:
        st.warning("Model file not found or could not be loaded from `models/function_rf.pkl`.")

# -------------------------------------------------
# Explore Dataset
# -------------------------------------------------
elif view == "Explore Dataset":
    st.markdown("## Explore Dataset")

    if df is None:
        st.error("Dataset not found.")
    else:
        filtered_df = df.copy()

        col_a, col_b = st.columns(2)

        with col_a:
            if "Gene Name" in df.columns:
                gene_options = sorted(df["Gene Name"].dropna().astype(str).unique())
                selected_genes = st.multiselect("Filter by Gene Name", gene_options)
                if selected_genes:
                    filtered_df = filtered_df[filtered_df["Gene Name"].astype(str).isin(selected_genes)]

        with col_b:
            if "Organism" in df.columns:
                organism_options = sorted(df["Organism"].dropna().astype(str).unique())
                selected_organisms = st.multiselect("Filter by Organism", organism_options)
                if selected_organisms:
                    filtered_df = filtered_df[filtered_df["Organism"].astype(str).isin(selected_organisms)]

        st.write(f"Showing **{len(filtered_df)}** rows")
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### Visual Summary")

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            if "Sequence_length" in filtered_df.columns and not filtered_df.empty:
                fig, ax = plt.subplots()
                ax.hist(filtered_df["Sequence_length"].dropna(), bins=20)
                ax.set_title("Sequence Length Distribution")
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.info("`Sequence_length` column not available.")

        with plot_col2:
            if "GC_content" in filtered_df.columns and not filtered_df.empty:
                fig, ax = plt.subplots()
                ax.hist(filtered_df["GC_content"].dropna(), bins=20)
                ax.set_title("GC Content Distribution")
                ax.set_xlabel("GC Content")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.info("`GC_content` column not available.")

        if "Gene Name" in filtered_df.columns and not filtered_df.empty:
            gene_counts = filtered_df["Gene Name"].value_counts().head(10)
            st.markdown("### Top Gene Counts")
            fig, ax = plt.subplots()
            gene_counts.plot(kind="bar", ax=ax)
            ax.set_title("Top Genes in Filtered View")
            ax.set_xlabel("Gene Name")
            ax.set_ylabel("Count")
            st.pyplot(fig)

# -------------------------------------------------
# Run Prediction
# -------------------------------------------------
elif view == "Run Prediction":
    st.markdown("## Run Prediction")

    if df is None:
        st.error("Dataset not found.")
    elif model is None:
        st.warning("Model not loaded. Check `models/function_rf.pkl`.")
    else:
        st.write(
            """
Select a row from your existing dataset and use its engineered numeric features
as model input. This is the safest way to demo predictions using your real feature schema.
"""
        )

        working_df = df.copy()

        if "Gene Name" in working_df.columns:
            gene_options = sorted(working_df["Gene Name"].dropna().astype(str).unique())
            selected_gene = st.selectbox("Select Gene Name", gene_options)
            working_df = working_df[working_df["Gene Name"].astype(str) == selected_gene]

        row_index = st.selectbox(
            "Select a row to score",
            options=working_df.index.tolist(),
            format_func=lambda x: f"Row {x}"
        )

        selected_row = df.loc[[row_index]]

        st.markdown("### Selected Record")
        preview_cols = [c for c in ["Gene Name", "Accession", "Organism", "Sequence_length", "GC_content", "protein length"] if c in selected_row.columns]
        if preview_cols:
            st.dataframe(selected_row[preview_cols], use_container_width=True)
        else:
            st.dataframe(selected_row, use_container_width=True)

        model_input = prepare_input_from_row(selected_row)

        st.markdown("### Numeric Feature Input")
        st.write(f"Using **{model_input.shape[1]}** numeric features.")
        st.dataframe(model_input, use_container_width=True)

        if st.button("Predict"):
            try:
                prediction = model.predict(model_input)[0]
                st.success(f"Predicted Class / Output: **{prediction}**")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(model_input)[0]
                    prob_df = pd.DataFrame({
                        "Class Index": list(range(len(probs))),
                        "Probability": probs
                    })
                    st.markdown("### Prediction Confidence")
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                st.exception(e)
                st.warning(
                    "The model loaded, but the exact feature schema may not match this input. "
                    "If that happens, align the prediction input to the exact training columns."
                )

# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------
elif view == "Batch Prediction":
    st.markdown("## Batch Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file for batch scoring", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data Preview")
        st.dataframe(user_df.head(20), use_container_width=True)

        if model is None:
            st.warning("Model is not available, so batch prediction is disabled.")
        else:
            if st.button("Run Batch Prediction"):
                try:
                    batch_input = user_df.select_dtypes(include=["number"])
                    preds = model.predict(batch_input)

                    output_df = user_df.copy()
                    output_df["predicted_output"] = preds

                    st.success("Batch prediction complete.")
                    st.dataframe(output_df.head(20), use_container_width=True)

                    csv_bytes = output_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_bytes,
                        file_name="gene_function_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.exception(e)
                    st.warning(
                        "Uploaded CSV numeric columns may not match the model's expected training features."
                    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
**Built by Hakim Adam**  
**Project:** Gene Function and Mutation Impact Prediction  
**Tech:** Python, pandas, scikit-learn, Streamlit
"""
)
