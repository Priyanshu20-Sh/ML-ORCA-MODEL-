import streamlit as st

# Configure the Streamlit page header and layout
st.set_page_config(
    page_title="ORCA: Optimising Retention via Causal Analysis",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render main landing page introduction
st.title("🐋 ORCA: Optimising Retention via Causal Analysis")
st.subheader("Causal Inference & Uplift Modeling Dashboard")

st.markdown("""
Welcome to **ORCA**! 
This is a fully-featured, beginner-friendly **Causal Inference and Uplift Modeling** application. 

Instead of traditional predictive models (which predict *what* a user will do), ORCA uses causal meta-learners to predict **how much a customer's behavior will change in response to a specific action or campaign** (e.g. offering a discount).
""")

# 
# PIPELINE WALKTHROUGH DIAGRAM

st.divider()
st.header("🏁 The Causal Inference Pipeline")
st.write("ORCA takes you step-by-step through a complete industrial machine learning pipeline:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### **1. Data Ingestion**
    * Load your own CSV or generate our built-in **Demo Dataset**.
    * Preview columns and exploratory correlation heatmaps.
    """)

with col2:
    st.markdown("""
    ### **2. Preprocessing**
    * Declare target, treatment, and features.
    * Handle label encodings of categorical columns automatically.
    """)

with col3:
    st.markdown("""
    ### **3. Model Training**
    * Train **XGBoost** meta-learners.
    * Compare separate models (**T-Learner**) vs. single models (**S-Learner**).
    """)

with col4:
    st.markdown("""
    ### **4. Uplift Segmenting**
    * Calculate individual uplift scores.
    * Group users into **Persuadables**, **Sure Things**, **Lost Causes**, or **Sleeping Dogs**.
    """)

st.divider()

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.markdown("""
    ### **5. Curve Evaluation**
    * Plot actual **Qini Curves** and **Uplift Curves**.
    * Compare model success directly against random targeting.
    """)

with col6:
    st.markdown("""
    ### **6. Visualizations**
    * Density plots of predicted scores.
    * Analyze demographic differences and feature relationships.
    """)

with col7:
    st.markdown("""
    ### **7. What-If Simulator**
    * Run a financial campaign simulator.
    * Enter campaign costs and value to find the **mathematically optimal ROI**.
    """)

with col8:
    st.markdown("""
    ### **8. Executive Reports**
    * Read consolidated briefings.
    * Download cleaned modeling datasets and final CSV predictions.
    """)

st.divider()

st.info("*Ready to begin?** Open the **Sidebar Navigation** on the left and click **1__Dashboard** to load your data!")
