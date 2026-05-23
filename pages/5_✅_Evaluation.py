import streamlit as st
import matplotlib.pyplot as plt
from sklift.viz import plot_qini_curve, plot_uplift_curve
from metrics import evaluate_model

st.title("✅ Causal Performance Curves")

st.markdown("""
In causal inference, we cannot use standard ROC-AUC or F1-scores to judge "causal accuracy" because we don't have a ground truth uplift label for individual users. 
Instead, we sort users by their predicted uplift score and plot their **cumulative conversions** under treatment vs. control.

We support two major evaluation curves:
1. **Qini Curve**: Plots cumulative incremental conversions against targeted users.
2. **Uplift Curve (Cumulative Gain)**: Plots cumulative lift in conversion probability against the percentage of the targeted population.
""")

if st.session_state.get('results') is not None:
    results = st.session_state['results']
    
    st.write("Evaluating predicted scores on the holdout test set:")
    
    # Run evaluation function
    metrics = evaluate_model(results['Target'], results['Uplift_Score'], results['Treatment'])
    
    # Show key metrics in columns
    col1, col2 = st.columns(2)
    col1.metric("Qini Area Under Curve (Qini AUC):", f"{metrics['Qini']:.4f}", 
                help="Qini AUC measures how much better your uplift model performs compared to a random targeting strategy. Higher is better.")
    col2.metric("Area Under Uplift Curve (AUUC):", f"{metrics['AUUC']:.4f}",
                help="AUUC evaluates cumulative outcome differences between treatment and control groups. Higher is better.")
    
    st.divider()
    st.header("📈 Evaluation Curve Graphs")
    st.write("A model is effective if its curve climbs high above the diagonal 'Random' baseline:")
    
    tab1, tab2 = st.tabs(["📊 Qini Curve", "📈 Uplift Curve (Cumulative Gain)"])
    
    with tab1:
        st.subheader("Qini Curve Graph")
        st.write("The **Qini Curve** shows the cumulative number of incremental conversions as we target more users, sorted from highest uplift to lowest.")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_qini_curve(y_true=results['Target'], uplift=results['Uplift_Score'], treatment=results['Treatment'], ax=ax)
        ax.set_title("Qini Curve vs. Random Targeting", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with tab2:
        st.subheader("Uplift Curve Graph")
        st.write("The **Uplift Curve** plots cumulative lift in conversion probability against the percentage of the population targeted.")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_uplift_curve(y_true=results['Target'], uplift=results['Uplift_Score'], treatment=results['Treatment'], ax=ax)
        ax.set_title("Uplift Curve vs. Random Targeting", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    st.info("👉 Performance curves verified! Proceed to **Visualizations** in the sidebar to view demographic segments.")
else:
    st.warning("⚠️ Please calculate uplift scores on the **Uplift Analysis** page first.")
