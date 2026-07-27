import visualizations
import plotly.io as pio

# Set default renderer to None to avoid browser opening, we just want to save
pio.renderers.default = None

print("Generating Abstract Visualizations...")

# 1. ICIND Diagram
fig_icind = visualizations.create_icind_rewiring_diagram(disruption_score=0.056)
fig_icind.write_html("abstract_icind_rewiring.html")
print("Saved abstract_icind_rewiring.html")

# 2. Survival Risk (Hazard Ratio)
fig_survival = visualizations.create_survival_risk_plot(hazard_ratio=2.95)
fig_survival.write_html("abstract_survival_risk.html")
print("Saved abstract_survival_risk.html")

# 3. Attention Features
# Using data from NOVEL_PIPELINE_REPORT.json (Step 716)
top_features = [
    {"feature": "Macrophage_Blockade", "attention_weight": 0.166},
    {"feature": "Quantiseq_Macrophage_M2", "attention_weight": 0.150},
    {"feature": "effector_suppressor_balance", "attention_weight": 0.126},
    {"feature": "Quantiseq_T_cell_CD8+", "attention_weight": 0.115},
    {"feature": "Quantiseq_T_cell_regulatory_(Tregs)", "attention_weight": 0.097}
]
fig_attention = visualizations.create_attention_features_chart(top_features)
fig_attention.write_html("abstract_attention_drivers.html")
print("Saved abstract_attention_drivers.html")

print("Done.")
