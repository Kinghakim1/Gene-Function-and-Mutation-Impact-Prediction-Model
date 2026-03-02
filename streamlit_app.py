import os
import pickle
import pandas as pd
import streamlit as st

# Optional plotting libs
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Gene Mutation Impact Prediction",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_dataset():
    """
    Load your processed dataset from the repo.
    Change the file path to match your project structure.
    """
    possible_paths = [
        "data/processed_gene_data.csv",
        "processed_gene_data.csv",
        "data/final_dataset.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


@st.cache_resource
def load_model():
    """
    Load your trained ML model from the repo.
    Change the file path to match your actual model file.
    """
    possible_paths = [
        "models/gene_model.pkl",
        "gene_model.pkl",
        "models/random_forest_model.pkl"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    return None


# -----------------------------
# Header
# -----------------------------
st.title("🧬 Gene Mutation Impact Prediction")
st.subheader("Interactive demo of my machine learning pipeline for predicting mutation impact")

st.markdown("""
This app showcases my gene mutation / gene function project.  
It demonstrates how biological sequence-derived features and engineered attributes
can be used to analyze mutations and support impact prediction.

**What this demo shows:**
- Project overview
- Dataset exploration
- Quick mutation filtering
- Model prediction interface (if model file is included)
- Visual summaries for employers/recruiters
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

view = st.sidebar.radio(
    "Choose a section",
    ["Project Overview", "Explore Dataset", "Run Prediction", "Upload CSV"]
)

# -----------------------------
# Load assets
# -----------------------------
df = load_dataset()
model = load_model()

# -----------------------------
# Section: Project Overview
# -----------------------------
if view == "Project Overview":
    st.markdown("## Project Overview")

    st.write("""
    This project focuses on predicting the functional impact of gene mutations using
    machine learning. The pipeline includes:
    - biological data collection / preprocessing
    - feature engineering
    - model training and evaluation
    - interpretable outputs for downstream analysis
    """)

    st.markdown("### Suggested highlights to mention")
    st.markdown("""
    - Built an end-to-end ML pipeline for gene mutation analysis  
    - Engineered sequence-based and mutation-related features  
    - Compared models and evaluated predictive performance  
    - Designed an interactive visualization layer for sharing insights  
    """)

    if df is not None:
        st.markdown("### Quick Dataset Snapshot")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Unique Genes", df["gene"].nunique() if "gene" in df.columns else "N/A")

        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Dataset file not found yet. Add your processed CSV to your repo and update the file path.")

# -----------------------------
# Section: Explore Dataset
# -----------------------------
elif view == "Explore Dataset":
    st.markdown("## Explore Dataset")

    if df is None:
        st.warning("No dataset found. Add your CSV file and update the path in load_dataset().")
    else:
        filtered_df = df.copy()

        if "gene" in df.columns:
            gene_options = sorted(df["gene"].dropna().astype(str).unique())
            selected_genes = st.multiselect("Filter by gene", gene_options)
            if selected_genes:
                filtered_df = filtered_df[filtered_df["gene"].astype(str).isin(selected_genes)]

        if "mutation_type" in df.columns:
            mutation_options = sorted(df["mutation_type"].dropna().astype(str).unique())
            selected_mutations = st.multiselect("Filter by mutation type", mutation_options)
            if selected_mutations:
                filtered_df = filtered_df[filtered_df["mutation_type"].astype(str).isin(selected_mutations)]

        st.write(f"Showing **{len(filtered_df)}** rows")
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### Basic Distribution")

        # Example plot if mutation_type exists
        if "mutation_type" in filtered_df.columns and not filtered_df.empty:
            counts = filtered_df["mutation_type"].value_counts().head(10)

            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Top Mutation Types")
            ax.set_xlabel("Mutation Type")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("Add a `mutation_type` column to visualize mutation distribution.")

# -----------------------------
# Section: Run Prediction
# -----------------------------
elif view == "Run Prediction":
    st.markdown("## Run Prediction")

    st.write("Use this section to demonstrate a single prediction.")

    # These inputs are placeholders — align them to your actual trained model features
    gene_name = st.text_input("Gene Name", value="BRCA1")
    mutation_type = st.selectbox(
        "Mutation Type",
        ["Missense", "Nonsense", "Insertion", "Deletion", "Frameshift"]
    )
    conservation_score = st.slider("Conservation Score", 0.0, 1.0, 0.75)
    gc_content = st.slider("GC Content", 0.0, 1.0, 0.50)
    codon_bias = st.slider("Codon Bias Score", 0.0, 1.0, 0.40)

    if st.button("Predict Impact"):
        if model is None:
            st.error("Model file not found. Add your trained `.pkl` model to the repo and update load_model().")
        else:
            try:
                # Replace this with the exact feature order your model expects
                input_df = pd.DataFrame([{
                    "conservation_score": conservation_score,
                    "gc_content": gc_content,
                    "codon_bias": codon_bias
                }])

                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Impact: **{prediction}**")

                # If your model supports probabilities:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    st.write("Prediction confidence:")
                    st.write(probs)

                st.markdown("### Input Summary")
                st.write({
                    "gene_name": gene_name,
                    "mutation_type": mutation_type,
                    "conservation_score": conservation_score,
                    "gc_content": gc_content,
                    "codon_bias": codon_bias
                })

            except Exception as e:
                st.exception(e)
                st.warning("Your model loaded, but the input schema likely doesn’t match. Update the feature names/order.")

# -----------------------------
# Section: Upload CSV
# -----------------------------
elif view == "Upload CSV":
    st.markdown("## Upload CSV for Batch Review")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(user_df.head(20), use_container_width=True)

        if model is not None and st.button("Run Batch Prediction"):
            try:
                preds = model.predict(user_df)
                output_df = user_df.copy()
                output_df["predicted_impact"] = preds

                st.success("Batch prediction complete.")
                st.dataframe(output_df.head(20), use_container_width=True)

                csv = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv,
                    file_name="gene_mutation_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.exception(e)
                st.warning("Uploaded CSV columns may not match the model input features.")
        elif model is None:
            st.info("Model file not found yet, so batch prediction is disabled.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
**Built by Hakim Adam**  
GitHub repo: Add your GitHub repo link here  
Tech: Python, pandas, scikit-learn, Streamlit
""")
