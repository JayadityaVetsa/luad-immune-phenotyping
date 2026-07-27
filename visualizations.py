import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd

def create_icind_rewiring_diagram(disruption_score=0.056):
    """
    Creates a simplified, aesthetic network diagram comparing Normal vs Tumor 
    interactions, highlighting the Macrophage-Treg rewiring.
    """
    # Create the graph
    fig = go.Figure()

    # Nodes: CD8+, M2, Treg, Tumor
    nodes = {
        'CD8+ T-Cell': {'x': 0, 'y': 1, 'color': '#2ecc71'},  # Green (Good)
        'Tumor': {'x': 0.5, 'y': 0.5, 'color': '#e74c3c'},    # Red (Bad)
        'M2 Macrophage': {'x': 1, 'y': 1, 'color': '#f1c40f'}, # Yellow (Suppressor)
        'Treg': {'x': 0.5, 'y': 1.5, 'color': '#9b59b6'}      # Purple (Suppressor)
    }

    # Add Nodes
    for name, data in nodes.items():
        fig.add_trace(go.Scatter(
            x=[data['x']], y=[data['y']],
            mode='markers+text',
            marker=dict(size=40, color=data['color'], line=dict(color='white', width=2)),
            text=[name], textposition="bottom center",
            hoverinfo='text',
            name=name, showlegend=False
        ))

    # --- NORMAL STATE (Faint Background) ---
    # In normal state, CD8 attacks Tumor (if present), and suppression is low.
    # We represent this as a "Ghost" connection or just implied.
    
    # --- TUMOR STATE (The "Rewiring") ---
    # Highlight 1: Macrophage-Treg Crosstalk (The suppression mechanism)
    # Highlight 2: Blockade of CD8
    
    # Edge: M2 <-> Treg (Strong Crosstalk)
    fig.add_trace(go.Scatter(
        x=[nodes['M2 Macrophage']['x'], nodes['Treg']['x']],
        y=[nodes['M2 Macrophage']['y'], nodes['Treg']['y']],
        mode='lines',
        line=dict(width=8, color='#e67e22'), # Orange, thick
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Edge: Treg -> CD8 (Suppression)
    fig.add_trace(go.Scatter(
        x=[nodes['Treg']['x'], nodes['CD8+ T-Cell']['x']],
        y=[nodes['Treg']['y'], nodes['CD8+ T-Cell']['y']],
        mode='lines',
        line=dict(width=4, color='#95a5a6', dash='dash'), # Gray, dashed
        hoverinfo='skip',
        showlegend=False
    ))

    # Annotation for the Crosstalk
    fig.add_annotation(
        x=0.75, y=1.25,
        text="<b>Critical Rewiring</b><br>Macrophage-Treg Crosstalk",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        ax=40, ay=-40,
        bgcolor="rgba(255, 255, 255, 0.8)", opacity=0.9
    )

    # Disruption Score Display
    fig.add_annotation(
        x=0.5, y=0,
        text=f"<b>Network Disruption Score: {disruption_score}</b><br>(0.0 = Normal Baseline)",
        showarrow=False,
        font=dict(size=14, color="#c0392b"),
        bgcolor="#fadbd8", bordercolor="#c0392b", borderwidth=1, borderpad=10
    )

    fig.update_layout(
        title="<b>ICIND Analysis: Immune Network Rewiring</b><br><i>Visualizing the mechanism of suppression</i>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.8]),
        plot_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

def create_survival_risk_plot(hazard_ratio=2.95):
    """
    Creates a gauge-style plot or bar chart to visualize the Hazard Ratio risk.
    """
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = hazard_ratio,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "<b>Mortality Risk (Hazard Ratio)</b><br>Immune-Desert vs Inflamed"},
        delta = {'reference': 1.0, 'increasing': {'color': "red"}, "suffix": "x Risk"},
        gauge = {
            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#c0392b"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': '#2ecc71'},
                {'range': [1, 2], 'color': '#f1c40f'},
                {'range': [2, 5], 'color': '#e74c3c'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': hazard_ratio}}))
    
    fig.update_layout(height=400)
    return fig

def create_attention_features_chart(top_features):
    """
    Creates a 'cool' radial bar chart or stylized bar chart for attention weights.
    top_features: list of dicts {'feature': name, 'attention_weight': val}
    """
    df = pd.DataFrame(top_features).sort_values('attention_weight', ascending=True)
    
    # Simplify names for display
    df['Label'] = df['feature'].str.replace('Quantiseq_', '').str.replace('Epidish_BPRNACan_', '').str.replace('_', ' ')
    
    fig = px.bar(
        df, x='attention_weight', y='Label', orientation='h',
        title="<b>Deep Learning Attention Weights</b><br><i>What drives the prediction?</i>",
        color='attention_weight',
        color_continuous_scale='Bluered',
        labels={'attention_weight': 'Attention Importance', 'Label': 'Feature'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        height=500
    )
    
    # Add annotation for the top feature if it's Macrophage Blockade
    top_row = df.iloc[-1]
    msg = f"<b>{top_row['Label']}</b><br>Top Driver"
    
    fig.add_annotation(
        x=top_row['attention_weight'], y=top_row['Label'],
        text=msg,
        showarrow=True, arrowhead=2, ax=-60, ay=0,
        bgcolor="white", opacity=0.8
    )

    return fig
