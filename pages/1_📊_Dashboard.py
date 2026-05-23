import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Ingestion & Overview")

st.markdown("""
Welcome to the **ORCA Dashboard**. To get started, you can either:
1. **Upload your own customer CSV dataset** (which should contain a target outcome and a treatment group indicator).
2. **Generate a synthetic Demo Dataset** to see how the system operates immediately.
""")


# SECTION 1: DATA LOADING

st.header("📤 Load Dataset")

col_upload, col_demo = st.columns(2)

with col_upload:
    st.subheader("Option A: Upload Your CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="dashboard_uploader")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success(f" Successfully loaded **{uploaded_file.name}** ({df.shape[0]} rows, {df.shape[1]} columns)")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with col_demo:
    st.subheader("Option B: Load Synthetic Demo")
    st.write("Click below to instantly create a simulated customer retention dataset.")
    if st.button("🚀 Load Demo Dataset"):
        # We generate simulated customer data with known causal uplift patterns
        np.random.seed(42)
        n = 1000
        
        # Demographic characteristics
        age = np.random.randint(18, 65, n)
        income = np.random.normal(50000, 15000, n).astype(int)
        tenure = np.random.randint(1, 60, n)
        usage_freq = np.random.randint(1, 30, n)
        
        # 50% of people are randomly selected for treatment (e.g. got a promotional discount)
        treatment = np.random.binomial(1, 0.5, n)

        # Baseline retention probability (without treatment)
        # Based on age, income, tenure, and usage frequency
        base_prob = 1 / (1 + np.exp(-(0.01 * age - 0.00001 * income + 0.02 * tenure + 0.03 * usage_freq - 2)))
        
        # Causal effect of treatment (uplift): how much more likely they are to stay if treated
        # Notice that younger people with high usage frequency get the highest uplift from treatment!
        uplift_effect = treatment * (0.1 + 0.002 * usage_freq - 0.001 * age)
        
        # Combined final probability of staying
        prob = np.clip(base_prob + uplift_effect, 0, 1)
        
        # Binary target: 1 if retained, 0 if churned
        outcome = np.random.binomial(1, prob)

        df = pd.DataFrame({
            'age': age,
            'income': income,
            'tenure_months': tenure,
            'usage_frequency': usage_freq,
            'treatment': treatment,
            'retained': outcome # our target variable
        })
        st.session_state['df'] = df
        st.success(f" Demo dataset loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")


# SECTION 2: DATA PREVIEW & EXPLORATION

if st.session_state.get('df') is not None:
    df = st.session_state['df']

    st.divider()
    st.header(" Data Inspection & Summary")
    
    st.write("##### First 10 Rows:")
    st.dataframe(df.head(10), use_container_width=True)

    # Show high-level metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns", f"{df.shape[1]}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")

    st.divider()
    st.write("##### Numerical Summary Statistics:")
    st.dataframe(df.describe(), use_container_width=True)

    st.divider()
    st.write("##### Columns Info:")
    col_info = pd.DataFrame({
        'Data Type': df.dtypes.astype(str),
        'Filled Rows': df.notnull().sum(),
        'Missing Rows': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info.T, use_container_width=True)

  
    # SECTION 3: EXPLORATORY VISUALIZATIONS
  
    st.divider()
    st.header(" Exploratory Charts")
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if numeric_cols:
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Feature Correlation Heatmap")
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Heatmap requires at least 2 numerical columns.")
                
        with col_viz2:
            st.subheader("Distribution Plots")
            selected_feature = st.selectbox("Select a column to plot distribution:", numeric_cols)
            if selected_feature:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.histplot(df[selected_feature], kde=True, color='#1f77b4', ax=ax)
                ax.set_title(f"Distribution of {selected_feature}", fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No numeric columns found for plotting.")

    st.divider()
    st.info("👉 Dataset is ready! Go to **Preprocessing** in the sidebar to clean and prepare it for modeling.")
else:
    st.divider()
    st.warning("⬆ Please upload a CSV file or load the synthetic demo dataset above to begin.")
