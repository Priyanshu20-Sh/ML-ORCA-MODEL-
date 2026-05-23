import streamlit as st
from data_utils import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

st.title(" Preprocessing & Variable Selection")

st.markdown("""
Before building causal models, we need to declare:
1. **Target Variable (y)**: A binary column (1/0) indicating whether the customer stayed or converted.
2. **Treatment Variable (T)**: A binary column (1/0) indicating whether they received the special action or offer.
3. **Features (X)**: Characteristics of the customer used to make predictions (e.g. age, usage metrics, history).
""")

if st.session_state.get('df') is not None:
    df = st.session_state['df']
    cols = list(df.columns)
    
    # Pre-populate defaults if standard columns exist (helpful for our demo dataset)
    default_target_idx = cols.index('retained') if 'retained' in cols else (cols.index('target') if 'target' in cols else 0)
    default_treatment_idx = cols.index('treatment') if 'treatment' in cols else 0
    
    # Create selection widgets
    st.header(" Variable Selection")
    target_col = st.selectbox("Select Target Column (Binary outcome, e.g. 'retained'):", cols, index=default_target_idx)
    treatment_col = st.selectbox("Select Treatment Column (Binary group, e.g. 'treatment'):", cols, index=default_treatment_idx)
    
    # Exclude target and treatment from features list by default
    available_features = [c for c in cols if c not in [target_col, treatment_col]]
    default_features = available_features.copy()
    
    features = st.multiselect("Select Feature Columns (Predictors):", available_features, default=default_features)
    
    # Process data when button is clicked
    if st.button(" Clean & Encode Data"):
        if features and target_col and treatment_col:
            with st.spinner("Cleaning rows and converting categorical features to numbers..."):
                # Call our core data_utils preprocessing logic
                df_clean, encoders = preprocess_data(df, target_col, treatment_col, features)
                
                # Save results in Streamlit's session state so other pages can access it
                st.session_state['df_clean'] = df_clean
                st.session_state['target_col'] = target_col
                st.session_state['treatment_col'] = treatment_col
                st.session_state['features'] = features
                st.session_state['encoders'] = encoders
                
                st.success(" Preprocessing completed!")
                
                # Report any label encodings done on text columns
                if encoders:
                    st.write("#####  Categorical Mappings Created:")
                    for col, encoder in encoders.items():
                        st.info(f"Column **{col}** successfully encoded with categories: `{list(encoder.classes_)}`")
                else:
                    st.info("Information: No categorical features (text columns) required label encoding.")
                
                st.write("##### Cleaned Preprocessed Data Preview:")
                st.dataframe(df_clean.head(10), use_container_width=True)
                
                # Plot sample class and treatment balance to check for skewness
                st.divider()
                st.header(" Group Representation Balance")
                
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.subheader("Target Distribution")
                    target_counts = df_clean[target_col].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.pie(target_counts, labels=[f"Class {k}\n({v:,} rows)" for k, v in target_counts.items()],
                           autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=90, 
                           wedgeprops=dict(width=0.4, edgecolor='w'))
                    ax.set_title(f"Target distribution: {target_col}", fontsize=11, fontweight='bold')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with col_c2:
                    st.subheader("Treatment Balance")
                    treatment_counts = df_clean[treatment_col].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.pie(treatment_counts, labels=[f"Group {k}\n({v:,} rows)" for k, v in treatment_counts.items()],
                           autopct='%1.1f%%', colors=['#1f77b4', '#aec7e8'], startangle=90,
                           wedgeprops=dict(width=0.4, edgecolor='w'))
                    ax.set_title(f"Treatment distribution: {treatment_col}", fontsize=11, fontweight='bold')
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.info("👉 Preprocessing complete! Now navigate to **Model Training** in the sidebar.")
        else:
            st.error("Please make sure you have selected a target column, treatment column, and at least one feature.")
            
    elif st.session_state.get('df_clean') is not None:
        st.info(" Data has already been preprocessed! You can review a preview below or change settings and preprocess again.")
        st.dataframe(st.session_state['df_clean'].head(5), use_container_width=True)
else:
    st.warning("⚠️ Please load a dataset on the **Data Ingestion** page first.")
