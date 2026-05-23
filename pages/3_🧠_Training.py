import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from model import TLearner, SLearner
import matplotlib.pyplot as plt
import seaborn as sns

st.title(" Causal Model Training")

st.markdown("""
Now we will train causal inference estimators (Meta-Learners) using **XGBoost** as our core classifier. 

We support two major architectures:
1. **T-Learner (Two Models)**: Trains one XGBoost on control users and a separate XGBoost on treatment users.
2. **S-Learner (Single Model)**: Trains one single global XGBoost with the treatment indicator as a feature.
""")

if st.session_state.get('df_clean') is not None:
    df = st.session_state['df_clean']
    target_col = st.session_state['target_col']
    treatment_col = st.session_state['treatment_col']
    features = st.session_state['features']
    
    st.write(f"**Target Outcome:** `{target_col}` | **Treatment Indicator:** `{treatment_col}`")
    st.write(f"**Predictor Features:** {', '.join([f'`{f}`' for f in features])}")
    
    model_type = st.radio("Choose Meta-Learner Architecture:", ["T-Learner", "S-Learner"], 
                          help="T-Learner trains separate models. S-Learner trains one model with treatment as a column.")
    
    if st.button("🚀 Train Causal Model"):
        X = df[features]
        y = df[target_col]
        treatment = df[treatment_col]
        
        # Split into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(
            X, y, treatment, test_size=0.3, random_state=42
        )
        
        with st.spinner(f"Training XGBoost base classifiers for {model_type}..."):
            # Initialize the chosen learner
            if model_type == "T-Learner":
                model = TLearner()
            else:
                model = SLearner(treatment_col=treatment_col)
            
            # Fit model on training partition
            model.fit(X_train, y_train, trt_train)
            
            # Keep model, test data, and type in session state
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
            st.session_state['test_data'] = (X_test, y_test, trt_test)
            
            st.success(f"✅ Trained {model_type} on {len(X_train)} training records!")
            
           
            # DIAGNOSTIC METRICS
           
            st.divider()
            st.header("🎯 Classifier Diagnostic Metrics")
            st.write("These metrics evaluate the base classification accuracy of the underlying models on the holdout test set:")
            
            metrics = model.get_accuracy_metrics(X_test, y_test, trt_test)
            
            for model_name, score_dict in metrics.items():
                st.subheader(f"📈 {model_name} Scores")
                
                # Render metrics in clean columns
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{score_dict['Accuracy']:.1%}")
                col2.metric("Precision", f"{score_dict['Precision']:.1%}")
                col3.metric("Recall", f"{score_dict['Recall']:.1%}")
                col4.metric("F1-Score", f"{score_dict['F1-Score']:.1%}")
                col5.metric("ROC-AUC", f"{score_dict['ROC-AUC']:.3f}")
                
                # Render metric summary dataframe
                df_metrics = pd.DataFrame(score_dict.items(), columns=["Metric", "Score"])
                df_metrics['Score'] = df_metrics.apply(lambda r: f"{r['Score']:.3f}" if r['Metric'] == 'ROC-AUC' else f"{r['Score']:.1%}", axis=1)
                st.dataframe(df_metrics.T, use_container_width=True)
                
          
            # FEATURE IMPORTANCES
           
            st.divider()
            st.header(" Feature Importance")
            st.write("Understand which features played the largest roles in the model's predictions:")
            
            df_imp = model.get_feature_importances(features)
            
            if not df_imp.empty:
                col_chart, col_data = st.columns([2, 1])
                
                with col_chart:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    importance_col = 'Average Importance' if 'Average Importance' in df_imp.columns else df_imp.columns[0]
                    sns.barplot(x=df_imp[importance_col], y=df_imp.index, palette='viridis', ax=ax)
                    ax.set_title(f"Feature Importance ({model_type})", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Relative Score")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with col_data:
                    st.write("##### Raw Scores")
                    st.dataframe(df_imp, use_container_width=True)
            else:
                st.info("No feature importances available.")
                
            st.info("👉 Model is trained! Navigate to **Uplift Analysis** in the sidebar to generate customer uplift predictions.")
            
    elif st.session_state.get('model') is not None:
        st.info(f" A {st.session_state.get('model_type', 'model')} is already trained! You can re-train with a different learner above if desired.")
else:
    st.warning("⚠️ Please preprocess data on the **Preprocessing** page first.")
