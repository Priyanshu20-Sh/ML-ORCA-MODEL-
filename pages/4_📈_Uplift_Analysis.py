import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📈 Uplift Analysis & Segmentation")

st.markdown("""
Using our trained model, we will calculate **Individual Uplift Scores** on our test dataset.
We then apply causal segmentation to place each customer into one of four marketing quadrants:

1. **🟢 Persuadables** (High Uplift): Customers who buy/stay **only if they are treated**. Targeting these users provides maximum ROI!
2. **🔵 Sure Things** (Low Uplift, stayed anyway): Customers who buy/stay **whether treated or not**. Do not waste budget targeting them.
3. **⚪ Lost Causes** (Low Uplift, left anyway): Customers who **will not convert** regardless of treatment. Do not waste budget here.
4. **🔴 Sleeping Dogs** (Negative Uplift): Customers who are **annoyed by marketing/intervention** and churn *because* of it. Never target these!
""")

if st.session_state.get('model') is not None and st.session_state.get('test_data') is not None:
    model = st.session_state['model']
    X_test, y_test, trt_test = st.session_state['test_data']
    model_type = st.session_state.get('model_type', 'Model')
    
    st.write(f"Evaluating test partition containing **{len(X_test)}** records using the trained **{model_type}**.")
    
    st.header("🎛️ Adjust Segmentation Boundaries")
    st.write("You can adjust the thresholds used to classify users into Persuadables or Sleeping Dogs based on their uplift scores:")
    
    col_t1, col_t2 = st.columns(2)
    upper_threshold = col_t1.number_input("Persuadable Threshold (Uplift > X):", value=0.05, step=0.01)
    lower_threshold = col_t2.number_input("Sleeping Dog Threshold (Uplift < Y):", value=-0.05, step=0.01)
    
    if st.button("🚀 Calculate Uplift & Quadrants"):
        with st.spinner("Predicting individual uplift scores and running segmentation..."):
            # Predict uplift score using our meta-learner
            uplift_scores = model.predict_uplift(X_test)
            
            # Combine test features, actual outcome, treatment, and scores into one dataframe
            results = X_test.copy()
            results['Uplift_Score'] = uplift_scores
            results['Target'] = y_test
            results['Treatment'] = trt_test
            
            # Segmentation logic function
            def assign_segment(row):
                score = row['Uplift_Score']
                target = row['Target']
                
                # Check for positive/negative treatment effect
                if score > upper_threshold:
                    return 'Persuadables'
                elif score < lower_threshold:
                    return 'Sleeping Dogs'
                else:
                    # Neutral uplift users are split by whether they stayed or left
                    if target == 1:
                        return 'Sure Things'
                    else:
                        return 'Lost Causes'
            
            # Apply function
            results['Segment'] = results.apply(assign_segment, axis=1)
            
            # Save results and thresholds in session state
            st.session_state['results'] = results
            st.session_state['upper_threshold'] = upper_threshold
            st.session_state['lower_threshold'] = lower_threshold
            
            st.success("✅ Calculated uplift scores and quadrants successfully!")
            
            st.write("##### Sample Predictions DataFrame (Preview):")
            st.dataframe(results.head(10), use_container_width=True)
            
        
            # SEGMENTATION CHART & TABLES
          
            st.divider()
            st.header("🧩 Customer Quadrant Breakdown")
            
            col_desc, col_pie = st.columns([1, 1])
            
            with col_desc:
                st.write("##### Quadrant Representation:")
                
                counts = results['Segment'].value_counts()
                df_counts = pd.DataFrame({
                    'Causal Quadrant': counts.index,
                    'Customer Count': counts.values,
                    'Percentage': (counts.values / len(results))
                })
                df_counts['Percentage'] = df_counts['Percentage'].map(lambda p: f"{p:.1%}")
                st.dataframe(df_counts, use_container_width=True)
                
                st.markdown(f"""
                * **🟢 Persuadables**: **{counts.get('Persuadables', 0):,}** users (respond positively).
                * **🔵 Sure Things**: **{counts.get('Sure Things', 0):,}** users (already loyal).
                * **⚪ Lost Causes**: **{counts.get('Lost Causes', 0):,}** users (won't stay).
                * **🔴 Sleeping Dogs**: **{counts.get('Sleeping Dogs', 0):,}** users (negative reaction).
                """)
                
            with col_pie:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                colors = {
                    'Persuadables': '#2ca02c', # green
                    'Sure Things': '#1f77b4',  # blue
                    'Lost Causes': '#7f7f7f',   # gray
                    'Sleeping Dogs': '#d62728'  # red
                }
                segment_colors = [colors.get(x, '#d3d3d3') for x in counts.index]
                
                ax.pie(counts, labels=[f"{x}\n({counts[x]:,})" for x in counts.index], autopct='%1.1f%%',
                       colors=segment_colors, startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
                ax.set_title("Customer Quadrant Breakdown", fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            st.info("👉 Uplift segmentation complete! Proceed to **Evaluation** in the sidebar to view Qini curves.")
            
    elif st.session_state.get('results') is not None:
        st.info("✅ Uplift scores have already been calculated! You can recalculate with different thresholds above or review the preview.")
        st.dataframe(st.session_state['results'].head(5), use_container_width=True)
else:
    st.warning("⚠️ Please train a model on the **Model Training** page first.")
