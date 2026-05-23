import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Demographic & Uplift Visualizations")

st.markdown("""
Explore how the predicted causal uplift scores relate to customer demographics and attributes. 
This helps identify *why* certain customer segments respond better to treatments.
""")

if st.session_state.get('results') is not None:
    results = st.session_state['results']
    features = st.session_state.get('features', [])
    
    tab1, tab2, tab3 = st.tabs([
        "📈 Uplift Score Density", 
        "📦 Uplift by Demographic Segments", 
        "🔍 Feature vs. Uplift Relationships"
    ])
    
    with tab1:
        st.subheader("Predicted Uplift Score Distribution")
        st.write("Shows how uplift scores are distributed across the test population. We can group them by actual treatment status.")
        
        show_by_treatment = st.checkbox("Overlay by Treatment & Control Groups", value=True)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if show_by_treatment:
            sns.kdeplot(data=results, x='Uplift_Score', hue='Treatment', fill=True, common_norm=False, palette='coolwarm', ax=ax)
            ax.legend(labels=["Treatment", "Control"])
        else:
            sns.histplot(data=results, x='Uplift_Score', kde=True, color='#1f77b4', ax=ax)
            
        ax.set_title("Uplift Score Density Plot", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Causal Uplift (Probability Difference)")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with tab2:
        st.subheader("Uplift Scores Across Demographic Segments")
        st.write("Select a feature to segment customers and see their uplift score distributions:")
        
        if features:
            segment_col = st.selectbox("Select Segment Feature Column:", features)
            
            # Detect if feature is continuous or categorical to bin it
            if results[segment_col].nunique() > 10:
                st.write(f"Note: Column **{segment_col}** is continuous. Bins are generated automatically.")
                results_temp = results.copy()
                results_temp[f'{segment_col}_binned'] = pd.qcut(results_temp[segment_col], q=4, duplicates='drop')
                plot_col = f'{segment_col}_binned'
            else:
                results_temp = results
                plot_col = segment_col
                
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=results_temp, x=plot_col, y='Uplift_Score', palette='Set2', ax=ax)
            sns.stripplot(data=results_temp, x=plot_col, y='Uplift_Score', color='black', alpha=0.15, size=4, ax=ax)
            
            ax.set_title(f"Predicted Uplift Score by {segment_col}", fontsize=11, fontweight='bold')
            ax.set_ylabel("Uplift Score")
            ax.set_xlabel(segment_col)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No features available for segmentation.")
            
    with tab3:
        st.subheader("Feature vs. Uplift Relationships")
        st.write("Understand linear or continuous correlations between features and causal scores:")
        
        numeric_features = results[features].select_dtypes(include='number').columns.tolist()
        if numeric_features:
            x_feature = st.selectbox("Select Continuous X-axis Feature:", numeric_features)
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.regplot(data=results, x=x_feature, y='Uplift_Score', 
                        scatter_kws={'alpha':0.3, 'color': '#2ca02c'}, 
                        line_kws={'color': '#d62728', 'linewidth': 2}, ax=ax)
            
            ax.set_title(f"{x_feature} vs. Predicted Uplift Score", fontsize=11, fontweight='bold')
            ax.set_xlabel(x_feature)
            ax.set_ylabel("Uplift Score")
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numeric features found for regression scatter plot.")
            
    st.info("👉 Demographic distributions verified! Proceed to **Simulator** in the sidebar to simulate campaigns.")
else:
    st.warning("⚠️ Please calculate uplift scores on the **Uplift Analysis** page first.")
