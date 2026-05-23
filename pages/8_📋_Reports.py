import streamlit as st
import pandas as pd
import numpy as np

st.title("📋 Executive Causal Brief & Exports")

st.markdown("""
Consolidate the final outputs of the causal inference pipeline into an executive brief. 
You can also download clean preprocessed data and customer predictions with predicted uplift scores below.
""")

if st.session_state.get('results') is not None:
    results = st.session_state['results']
    model_type = st.session_state.get('model_type', 'Model')
    features = st.session_state.get('features', [])
    target_col = st.session_state.get('target_col', 'Target')
    treatment_col = st.session_state.get('treatment_col', 'Treatment')
    
    # Pre-calculate key metrics for the brief
    total_users = len(results)
    persuadables_pct = (results['Segment'] == 'Persuadables').sum() / total_users
    sleeping_dogs_pct = (results['Segment'] == 'Sleeping Dogs').sum() / total_users
    sure_things_pct = (results['Segment'] == 'Sure Things').sum() / total_users
    lost_causes_pct = (results['Segment'] == 'Lost Causes').sum() / total_users
    
    mean_uplift = results['Uplift_Score'].mean()
    max_uplift = results['Uplift_Score'].max()
    min_uplift = results['Uplift_Score'].min()
    
    st.divider()
    st.subheader("📰 Project ORCA Causal Brief")
    
    st.markdown(f"""
    ### **Executive Summary Report**
    
    #### **1. Project Parameters**
    * **Causal Model Structure**: {model_type} (Base Estimator: XGBoost)
    * **Analyzed Demographic Predictors**: {', '.join([f'`{f}`' for f in features])}
    * **Target Retention Metric**: `{target_col}` (Binary)
    * **Treatment Intervention Indicator**: `{treatment_col}` (Binary)
    
    #### **2. Population Composition Breakdown**
    * **Total Customers Evaluated**: **{total_users:,}**
    * **🟢 Persuadables (Target Priority #1)**: **{persuadables_pct:.1%}** — These users respond highly positively to treatment.
    * **🔵 Sure Things (Do Not Target)**: **{sure_things_pct:.1%}** — These users convert anyway; targeting them is redundant cost.
    * **⚪ Lost Causes (Do Not Target)**: **{lost_causes_pct:.1%}** — These users will not convert; targeting them is a sunk cost.
    * **🔴 Sleeping Dogs (Never Target)**: **{sleeping_dogs_pct:.1%}** — These users convert *only* if left alone. Targeting triggers churn!
    
    #### **3. Uplift Value Statistics**
    * **Average Treatment Uplift Effect**: **{mean_uplift:+.2%}**
    * **Maximum Customer Uplift**: **{max_uplift:+.2%}**
    * **Minimum Customer Uplift (Sleeping Dog)**: **{min_uplift:+.2%}**
    """)

    # EXPORTS & DOWNLOADS
   
    st.divider()
    st.header("📥 Export Causal Datasets")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.write("##### 📊 Clean Preprocessed Data:")
        st.write("Download the clean dataset containing label-encoded categorical columns.")
        if st.session_state.get('df_clean') is not None:
            csv_clean = st.session_state['df_clean'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Preprocessed Data (CSV)",
                data=csv_clean,
                file_name="orca_preprocessed_data.csv",
                mime="text/csv"
            )
            
    with col_dl2:
        st.write("##### 🎯 Customer Causal Predictions:")
        st.write("Download the complete test population dataset containing predicted uplift scores and segment assignments.")
        csv_results = results.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download Causal Predictions (CSV)",
            data=csv_results,
            file_name="orca_uplift_predictions.csv",
            mime="text/csv"
        )
        
    st.divider()
    st.subheader("📋 Top 100 Highest Uplift Prospects")
    st.write("These represent your prime candidates for targeted campaigns, sorted by highest uplift score:")
    st.dataframe(results.sort_values(by='Uplift_Score', ascending=False).head(100), use_container_width=True)
    
    st.success("🎉 Full Causal Inference Pipeline executed successfully! Your project is ready for review.")
else:
    st.warning("⚠️ No reports available. Please load data, preprocess, train the model, and calculate uplift first.")
