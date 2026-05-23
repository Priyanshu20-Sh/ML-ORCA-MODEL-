import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title(" What-If Campaign Simulator")

st.markdown("""
Uplift modeling is powerful because it lets us **optimize marketing budgets**. 
Rather than blindly targeting everyone (which wastes budget on *Sure Things* and *Lost Causes*) or annoying customers (*Sleeping Dogs*), 
we target only customers whose predicted uplift score is high enough to justify the treatment cost.

Use this simulator to estimate the ROI of your next campaign.
""")

if st.session_state.get('results') is not None:
    results = st.session_state['results']
    
    st.header("🎛️ Simulation Parameters")
    col_p1, col_p2 = st.columns(2)
    cost_per_treatment = col_p1.number_input("Cost per Treated Customer ($):", value=1.00, min_value=0.01, step=0.10)
    revenue_per_success = col_p2.number_input("Value per Successful Conversion ($):", value=10.00, min_value=0.10, step=1.00)
    
    st.divider()
    
  
    # SIMULATION ROI GRAPH CALCULATIONS

    thresholds = np.linspace(-0.5, 0.5, 101)
    costs = []
    revenues = []
    rois = []
    targeted_sizes = []
    
    for t in thresholds:
        targeted = results[results['Uplift_Score'] >= t]
        c = len(targeted) * cost_per_treatment
        # Cumulative revenue is sum of uplift scores * conversion value
        r = targeted['Uplift_Score'].sum() * revenue_per_success
        net_roi = r - c
        
        costs.append(c)
        revenues.append(r)
        rois.append(net_roi)
        targeted_sizes.append(len(targeted))
        
    df_sim = pd.DataFrame({
        'Threshold': thresholds,
        'Cost': costs,
        'Revenue': revenues,
        'ROI': rois,
        'Targeted_Size': targeted_sizes
    })
    
    # Calculate mathematically optimal threshold
    optimal_idx = df_sim['ROI'].idxmax()
    optimal_threshold = df_sim.loc[optimal_idx, 'Threshold']
    max_roi = df_sim.loc[optimal_idx, 'ROI']
    optimal_targeted = df_sim.loc[optimal_idx, 'Targeted_Size']
    
    st.header("📈 ROI Optimization Curve")
    st.markdown(f"""
    By simulating across all thresholds, we find that the **mathematical peak Net ROI** occurs 
    at an uplift threshold of **{optimal_threshold:.2f}**. 
    At this threshold, targeting customers with scores above **{optimal_threshold:.2f}** yields an estimated Net ROI of **${max_roi:.2f}**.
    """)
    
    threshold = st.slider("Select Causal Uplift Target Threshold:", 
                          min_value=-0.5, max_value=0.5, value=float(optimal_threshold), step=0.01,
                          help="Slide to evaluate the financial impact of targeting at different thresholds. The default is set to the optimal peak.")
    
    # Retrieve selected threshold performance
    sel_idx = (df_sim['Threshold'] - threshold).abs().idxmin()
    sel_row = df_sim.loc[sel_idx]
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Users Targeted", f"{int(sel_row['Targeted_Size']):,} / {len(results):,}")
    col_r2.metric("Campaign Cost", f"${sel_row['Cost']:.2f}")
    col_r3.metric("Estimated Revenue", f"${sel_row['Revenue']:.2f}")
    
    roi_delta = sel_row['ROI'] - max_roi
    col_r4.metric("Net ROI", f"${sel_row['ROI']:.2f}", 
                    delta=f"{roi_delta:.2f} vs Peak" if abs(roi_delta) > 0.01 else "Optimal Peak",
                    delta_color="inverse" if roi_delta < 0 else "normal")
    
    # Plotting ROI Curve
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df_sim['Threshold'], df_sim['ROI'], color='#2ca02c', linewidth=2.5, label='Estimated ROI')
    ax.fill_between(df_sim['Threshold'], df_sim['ROI'], color='#2ca02c', alpha=0.1)
    
    # Highlight optimal peak and current selection
    ax.scatter([optimal_threshold], [max_roi], color='#d62728', s=100, zorder=5, label=f'Optimal Peak ({optimal_threshold:.2f})')
    ax.scatter([threshold], [sel_row['ROI']], color='#1f77b4', s=100, marker='X', zorder=5, label=f'Your Selection ({threshold:.2f})')
    
    ax.set_title("Net Campaign ROI vs. Causal Targeting Threshold", fontsize=11, fontweight='bold')
    ax.set_xlabel("Uplift Score Threshold")
    ax.set_ylabel("Net ROI / Profit ($)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    
    # STRATEGY COMPARISON
   
    st.divider()
    st.header("📊 Causal Strategy Comparison")
    
    # Strategy A: Target All Users
    all_cost = len(results) * cost_per_treatment
    all_revenue = results['Uplift_Score'].sum() * revenue_per_success
    all_roi = all_revenue - all_cost
    
    # Strategy B: Random targeting (Target a random 50%)
    rand_cost = (len(results) / 2) * cost_per_treatment
    rand_revenue = (results['Uplift_Score'].sum() / 2) * revenue_per_success
    rand_roi = rand_revenue - rand_cost
    
    df_strategies = pd.DataFrame({
        'Targeting Strategy': ['Target All', 'Random (50%)', 'Target None', 'Uplift Optimized (Selection)'],
        'Net ROI ($)': [all_roi, rand_roi, 0.0, sel_row['ROI']],
        'Color': ['#aec7e8', '#ffbb78', '#c7c7c7', '#2ca02c']
    })
    
    col_table, col_bar = st.columns([1, 1])
    
    with col_table:
        st.write("##### Financial Performance Table:")
        st.dataframe(df_strategies[['Targeting Strategy', 'Net ROI ($)']], use_container_width=True)
        st.markdown("""
        **Core Lesson:**
        By targeting only users who have a high probability of being influenced positively, 
        we save massive campaign costs and avoid bothering sleeping dogs, leading to significantly higher net profits!
        """)
        
    with col_bar:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df_strategies, x='Targeting Strategy', y='Net ROI ($)', palette=df_strategies['Color'].tolist(), ax=ax)
        ax.set_title("Strategy Net Profit Comparison", fontsize=11, fontweight='bold')
        ax.set_ylabel("Net ROI ($)")
        plt.xticks(rotation=15)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    st.info("👉 Strategy simulations complete! Proceed to **Reports** in the sidebar to generate the final summary.")
else:
    st.warning("⚠️ Please calculate uplift scores on the **Uplift Analysis** page first.")
