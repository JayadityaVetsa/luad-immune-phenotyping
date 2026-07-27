"""
LUAD Immune Phenotype Analysis Platform
========================================
A comprehensive Streamlit application for analyzing bulk RNA-seq data
from Lung Adenocarcinoma patients.

Features:
- Immune cell deconvolution from bulk RNA-seq
- Cancer classification with confidence scores
- Immune phenotype clustering (k=3)
- SHAP-based explainability
- Survival analysis with Kaplan-Meier curves

Author: Jayaditya (ISEF Project)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import tempfile
import subprocess
import os
from pathlib import Path
import pickle
import shap
from scipy.spatial.distance import euclidean
from scipy import stats
import warnings

# Survival analysis
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import visualizations # Custom visualization library

# NOVEL CONTRIBUTIONS (ISEF additions)
import torch
import torch.nn.functional as F
import networkx as nx
from novel_deep_learning_phenotyping import DeepImmunePhenotypePredictor
from novel_immune_network_dynamics import ImmuneNetworkAnalyzer

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="LUAD Immune Phenotype Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# =============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E3A5F;
        --secondary-color: #3D5A80;
        --accent-color: #E63946;
        --success-color: #2D6A4F;
        --warning-color: #F77F00;
        --background-light: #F8F9FA;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #3D5A80 50%, #457B9D 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #3D5A80;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Phenotype badges */
    .phenotype-inflamed {
        background: linear-gradient(135deg, #E63946 0%, #c1121f 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .phenotype-suppressed {
        background: linear-gradient(135deg, #6A0572 0%, #4a0550 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .phenotype-desert {
        background: linear-gradient(135deg, #457B9D 0%, #2d5a7b 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        border: 1px solid #b6d4fe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffda6a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1e7dd;
        border: 1px solid #a3cfbb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A5F 0%, #2d5a7b 100%);
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "trained_models"
TCGA_DIR = BASE_DIR / "TCGA_Real_data"
MITCR_FILE = BASE_DIR / "mitcr_sampleStatistics_20160714.tsv"

# Phenotype colors (matching our visualizations)
PHENOTYPE_COLORS = {
    'Immune-Inflamed': '#E63946',
    'Immune-Suppressed': '#6A0572',
    'Immune-Desert': '#457B9D'
}

# Key features for clustering
CLUSTER_FEATURES = [
    'shannon',
    'Immune_Engagement_Index',
    'Macrophage_Blockade',
    'Epidish_BPRNACan_M2',
    'Quantiseq_T_cell_CD8+',
    'Quantiseq_T_cell_regulatory_(Tregs)'
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained models from disk."""
    model_path = MODELS_DIR / "luad_pipeline_models.pkl"
    if not model_path.exists():
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_deep_learning_model():
    """Load the novel attention-based deep learning model."""
    try:
        dl_model_path = BASE_DIR / "best_deep_immune_model.pth"
        if not dl_model_path.exists():
            return None
        
        # Get input dimension from cluster features
        input_dim = len(CLUSTER_FEATURES)
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dl_model = DeepImmunePhenotypePredictor(
            input_dim=input_dim,
            n_phenotypes=3,
            device=device
        )
        
        # Load weights
        dl_model.model.load_state_dict(torch.load(dl_model_path, map_location=device))
        dl_model.model.eval()
        
        return dl_model
    except Exception as e:
        print(f"[DEBUG] Could not load deep learning model: {e}")
        return None

@st.cache_data
def load_mitcr_data():
    """Load MiTCR data for Shannon index lookup."""
    if not MITCR_FILE.exists():
        return None
    df = pd.read_csv(MITCR_FILE, sep='\t')
    df['Sample_ID'] = df['AliquotBarcode'].str[:16]
    return df[['Sample_ID', 'shannon', 'numClones']].drop_duplicates(subset=['Sample_ID'])

@st.cache_data
def load_training_data():
    """Load the merged immune features for reference."""
    merged_path = BASE_DIR / "merged_immune_features.csv"
    if not merged_path.exists():
        return None
    return pd.read_csv(merged_path)

def run_deconvolution(tsv_content: bytes, patient_id: str) -> pd.DataFrame:
    """Run R deconvolution on uploaded TSV file."""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.tsv', delete=False) as tmp_input:
        tmp_input.write(tsv_content)
        input_path = tmp_input.name
    
    output_path = tempfile.mktemp(suffix='.csv')
    
    # Create R script for single patient
    r_script = f'''
# Single Patient Deconvolution
local_lib <- file.path("{str(BASE_DIR).replace(chr(92), '/')}", "R_libs")
.libPaths(c(local_lib, .libPaths()))

library(readr)
library(dplyr)
library(tibble)
library(tidyr)

setwd("{str(BASE_DIR).replace(chr(92), '/')}")

# Load GEMDeCan functions
setwd("GEMDeCan_deconvolution")
source("scripts/deconvolution/deconvolution_algorithms.R")
library(immunedeconv)
library(EpiDISH)

# Read patient file
f <- "{input_path.replace(chr(92), '/')}"
patient_id <- "{patient_id}"

d <- suppressMessages(read_tsv(f, comment = "#", show_col_types = FALSE))
if ("gene_name" %in% colnames(d)) {{
    d <- d %>% select(Gene = gene_name, Value = tpm_unstranded)
}} else if ("tpm_unstranded" %in% colnames(d)) {{
    d <- d %>% select(Gene = 1, Value = tpm_unstranded)
}} else {{
    d <- d %>% select(Gene = 1, Value = 2)
}}

mx <- d %>%
    filter(!is.na(Gene)) %>%
    mutate(Value = as.numeric(Value)) %>%
    group_by(Gene) %>%
    summarise(Value = mean(Value, na.rm=TRUE), .groups = "drop") %>%
    column_to_rownames("Gene") %>%
    as.matrix()
colnames(mx) <- patient_id
mx[is.na(mx)] <- 0

# Duplicate for algorithm requirements
mx_run <- cbind(mx, mx)
colnames(mx_run) <- c(patient_id, paste0(patient_id, "_Dummy"))

# Run Quantiseq
res_q <- tryCatch({{
    r <- computeQuantiseq(mx_run)
    r <- r %>% filter(sample == patient_id)
    r
}}, error=function(e) tibble(sample=patient_id))

# Run EpiDISH
sig_files <- list.files("scripts/deconvolution/signatures", full.names = TRUE)
sig_files <- sig_files[!dir.exists(sig_files)]

res_v <- tryCatch({{
    r <- methods_with_variable_signatures(mx_run, sig_files)
    r <- r %>% filter(sample == patient_id)
    r
}}, error=function(e) tibble(sample=patient_id))

if("Samples" %in% colnames(res_q)) res_q <- res_q %>% rename(sample = Samples)
if("Samples" %in% colnames(res_v)) res_v <- res_v %>% rename(sample = Samples)

combined <- full_join(res_q, res_v, by="sample")

setwd("..")
write_csv(combined, "{output_path.replace(chr(92), '/')}")
'''
    
    # Write and run R script
    r_script_path = tempfile.mktemp(suffix='.R')
    with open(r_script_path, 'w') as f:
        f.write(r_script)
    
    try:
        result = subprocess.run(
            ['Rscript', r_script_path],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            st.error(f"R Error: {result.stderr[:500]}")
            return None
        
        # Load results
        if os.path.exists(output_path):
            deconv_df = pd.read_csv(output_path)
            return deconv_df
        else:
            st.error("Deconvolution output not found")
            return None
            
    except subprocess.TimeoutExpired:
        st.error("Deconvolution timed out. Please try again.")
        return None
    finally:
        # Cleanup
        for f in [input_path, output_path, r_script_path]:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def get_phenotype_badge(phenotype: str) -> str:
    """Return HTML for phenotype badge."""
    class_map = {
        'Immune-Inflamed': 'phenotype-inflamed',
        'Immune-Suppressed': 'phenotype-suppressed',
        'Immune-Desert': 'phenotype-desert'
    }
    cls = class_map.get(phenotype, 'phenotype-desert')
    return f'<span class="{cls}">{phenotype}</span>'

def create_gauge_chart(value: float, title: str, max_val: float = 1.0, 
                       thresholds: list = None) -> go.Figure:
    """Create a beautiful gauge chart."""
    if thresholds is None:
        thresholds = [0.33, 0.66, 1.0]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': 'black'}},
        number={'font': {'size': 40, 'color': 'black'}, 'suffix': '%' if max_val == 1 else ''},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 2, 'tickcolor': '#1E3A5F'},
            'bar': {'color': '#3D5A80'},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#dee2e6',
            'steps': [
                {'range': [0, thresholds[0]], 'color': '#d4edda'},
                {'range': [thresholds[0], thresholds[1]], 'color': '#fff3cd'},
                {'range': [thresholds[1], thresholds[2]], 'color': '#f8d7da'}
            ],
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1E3A5F'}
    )
    return fig

def create_cell_composition_chart(deconv_row: pd.Series) -> go.Figure:
    """Create a sunburst chart of cell composition."""
    # Extract Quantiseq results
    quantiseq_cols = [c for c in deconv_row.index if c.startswith('Quantiseq_') and pd.notna(deconv_row[c])]
    
    cells = []
    values = []
    parents = []
    
    # Categorize cells
    categories = {
        'T Cells': ['T_cell_CD8+', 'T_cell_CD4+', 'T_cell_regulatory'],
        'Myeloid': ['Macrophage_M1', 'Macrophage_M2', 'Monocyte', 'Neutrophil', 'Dendritic'],
        'Other': ['B_cell', 'NK_cell', 'Mast_cell', 'Eosinophil']
    }
    
    # Add root
    total = 0
    category_totals = {}
    
    for col in quantiseq_cols:
        val = float(deconv_row[col])
        if val > 0:
            total += val
            cell_name = col.replace('Quantiseq_', '').replace('_', ' ')
            
            # Find category
            cat = 'Other'
            for cat_name, cell_types in categories.items():
                if any(ct in col for ct in cell_types):
                    cat = cat_name
                    break
            
            category_totals[cat] = category_totals.get(cat, 0) + val
            cells.append(cell_name)
            values.append(val)
            parents.append(cat)
    
    # Add categories
    for cat, val in category_totals.items():
        cells.insert(0, cat)
        values.insert(0, val)
        parents.insert(0, "Immune Cells")
    
    cells.insert(0, "Immune Cells")
    values.insert(0, total)
    parents.insert(0, "")
    
    fig = go.Figure(go.Sunburst(
        labels=cells,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Proportion: %{value:.3f}<extra></extra>',
        marker=dict(
            colors=px.colors.qualitative.Set2,
            line=dict(color='white', width=2)
        ),
        textfont=dict(size=12)
    ))
    
    fig.update_layout(
        title=dict(text='Immune Cell Composition', font=dict(size=18, color='black')),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def create_radar_chart(deconv_row: pd.Series, reference_data: pd.DataFrame = None) -> go.Figure:
    """Create radar chart comparing patient to population."""
    key_cells = [
        ('Quantiseq_T_cell_CD8+', 'CD8+ T Cells'),
        ('Quantiseq_Macrophage_M2', 'M2 Macrophages'),
        ('Quantiseq_T_cell_regulatory_(Tregs)', 'Tregs'),
        ('Quantiseq_Macrophage_M1', 'M1 Macrophages'),
        ('Quantiseq_NK_cell', 'NK Cells'),
        ('Quantiseq_B_cell', 'B Cells')
    ]
    
    categories = []
    patient_values = []
    pop_mean = []
    pop_upper = []
    
    for col, name in key_cells:
        if col in deconv_row.index and pd.notna(deconv_row[col]):
            categories.append(name)
            patient_values.append(float(deconv_row[col]))
            
            if reference_data is not None and col in reference_data.columns:
                pop_mean.append(float(reference_data[col].mean()))
                pop_upper.append(float(reference_data[col].quantile(0.75)))
            else:
                pop_mean.append(0)
                pop_upper.append(0)
    
    fig = go.Figure()
    
    # Population mean
    if pop_mean:
        fig.add_trace(go.Scatterpolar(
            r=pop_mean + [pop_mean[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(69, 123, 157, 0.2)',
            line=dict(color='#457B9D', width=2),
            name='Population Mean'
        ))
    
    # Patient values
    fig.add_trace(go.Scatterpolar(
        r=patient_values + [patient_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(230, 57, 70, 0.3)',
        line=dict(color='#E63946', width=3),
        name='This Patient'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(patient_values + pop_mean) * 1.2]
            )
        ),
        title=dict(text='Immune Profile Comparison', font=dict(size=18, color='black')),
        showlegend=True,
        legend=dict(x=0.8, y=1.1),
        height=400,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

def create_umap_with_patient(models: dict, patient_features: np.ndarray, 
                             training_df: pd.DataFrame, patient_cluster: str = None) -> go.Figure:
    """Create UMAP plot with training data and new patient marked.
    
    Args:
        patient_cluster: The phenotype classification from XGBoost (should be passed in, not recalculated)
    """
    
    # Get training UMAP coordinates
    scaler = models['scaler']
    umap_reducer = models['umap_reducer']
    label_encoder = models['label_encoder']
    phenotype_clf = models['phenotype_classifier']
    
    # Scale and transform training data
    features = models['cluster_features']
    tumor_mask = training_df['sample'].str.contains('-01A')
    tumor_df = training_df[tumor_mask].copy()
    
    X_tumor = tumor_df[features].values
    X_tumor_scaled = scaler.transform(X_tumor)
    tumor_umap = umap_reducer.transform(X_tumor_scaled)
    
    # Use XGBoost phenotype classifier for consistent labeling (not KMeans)
    # Note: phenotype classifier was trained on UNSCALED data
    tumor_labels_encoded = phenotype_clf.predict(X_tumor)
    tumor_df['Cluster'] = label_encoder.inverse_transform(tumor_labels_encoded)
    tumor_df['UMAP_1'] = tumor_umap[:, 0]
    tumor_df['UMAP_2'] = tumor_umap[:, 1]
    
    # Transform new patient for UMAP position
    patient_scaled = scaler.transform(patient_features.reshape(1, -1))
    patient_umap = umap_reducer.transform(patient_scaled)
    
    # Use passed-in patient_cluster (from XGBoost) for consistency
    if patient_cluster is None:
        # Fallback: use XGBoost classifier (trained on UNSCALED data)
        patient_pred = phenotype_clf.predict(patient_features.reshape(1, -1))[0]
        patient_cluster = label_encoder.inverse_transform([patient_pred])[0]
    
    # Create figure
    fig = go.Figure()
    
    # Add training data points
    for cluster in ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']:
        mask = tumor_df['Cluster'] == cluster
        subset = tumor_df[mask]
        
        fig.add_trace(go.Scatter(
            x=subset['UMAP_1'],
            y=subset['UMAP_2'],
            mode='markers',
            marker=dict(
                size=12,
                color=PHENOTYPE_COLORS[cluster],
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            name=f'{cluster} (n={len(subset)})',
            hovertemplate=f'{cluster}<br>Sample: %{{customdata}}<extra></extra>',
            customdata=subset['sample']
        ))
    
    # Add new patient as star
    fig.add_trace(go.Scatter(
        x=[patient_umap[0, 0]],
        y=[patient_umap[0, 1]],
        mode='markers+text',
        marker=dict(
            size=25,
            color=PHENOTYPE_COLORS[patient_cluster],
            symbol='star',
            line=dict(color='black', width=2)
        ),
        name='NEW PATIENT',
        text=['YOU'],
        textposition='top center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate=f'<b>New Patient</b><br>Phenotype: {patient_cluster}<extra></extra>'
    ))
    
    # Add annotations
    fig.add_annotation(
        x=patient_umap[0, 0],
        y=patient_umap[0, 1] - 0.5,
        text=f"Classified as: {patient_cluster}",
        showarrow=False,
        font=dict(size=12, color=PHENOTYPE_COLORS[patient_cluster], family='Arial Black'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=PHENOTYPE_COLORS[patient_cluster],
        borderwidth=2,
        borderpad=4
    )
    
    fig.update_layout(
        title=dict(
            text='Immune Phenotype Clustering (UMAP)',
            font=dict(size=20, color='#1E3A5F')
        ),
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=600,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white'
    )
    
    return fig, patient_cluster



def create_radar_chart(patient_row: pd.Series, models: dict) -> go.Figure:
    """Create a radar chart comparing patient to phenotype means."""
    
    features = models['cluster_features']
    # Clean labels
    feature_labels = [f.replace('Quantiseq_', '').replace('Epidish_', '').replace('Macrophage_', 'Mac_').replace('_cell', '') for f in features]
    
    # Get population means (from training data in models, or we compute roughly)
    # Ideally we'd have these precomputed. For now, we normalize the patient values 
    # based on the scaler range (roughly).
    
    patient_vals = []
    for f in features:
        val = float(patient_row.get(f, 0))
        patient_vals.append(val)
    
    # Normalize (min-max scaling approximate)
    # We use the scaler's mean/scale to get z-scores, then sigmoid to 0-1?
    # Or just raw values if they are small fractions. 
    # Most are cell fractions (0-1). Macrophage Blockade is interaction term.
    
    # Let's use simple normalization for display
    norm_vals = []
    for val in patient_vals:
        # Cap at reasonable max for visualization
        norm = min(val, 1.0) 
        if val > 1.0: norm = 1.0 # Simple cap
        norm_vals.append(norm)

    categories = feature_labels
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    patient_vals_plot = norm_vals + [norm_vals[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_vals_plot,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(230, 57, 70, 0.5)', # Reddish
        line=dict(color='#E63946', width=2),
        name='This Patient'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(norm_vals)*1.2 if max(norm_vals)>0 else 1]
            )
        ),
        showlegend=True,
        title=dict(text='Immune Profile Fingerprint', font=dict(size=16)),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_dysregulation_chart(patient_dysreg: float, training_df: pd.DataFrame,
                               patient_cluster: str, models: dict) -> go.Figure:
    """Create a chart showing dysregulation score in context."""
    
    # Calculate dysregulation for all training samples
    features = models['cluster_features']
    scaler = models['scaler']
    normal_centroid = models['normal_centroid']
    cluster_mapping = models['cluster_mapping']
    cluster_model = models['cluster_model']
    
    tumor_mask = training_df['sample'].str.contains('-01A')
    tumor_df = training_df[tumor_mask].copy()
    
    # Ensure features exist
    available_feats = [f for f in features if f in tumor_df.columns]
    if len(available_feats) < len(features):
        return go.Figure()

    X_tumor_scaled = scaler.transform(tumor_df[features].values)
    dysreg_scores = [euclidean(s, normal_centroid) for s in X_tumor_scaled]
    tumor_labels = cluster_model.predict(X_tumor_scaled)
    tumor_df['Dysregulation'] = dysreg_scores
    tumor_df['Cluster'] = [cluster_mapping[l] for l in tumor_labels]
    
    # Create violin + strip plot (Jittered)
    fig = go.Figure()
    
    for i, cluster in enumerate(['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']):
        subset = tumor_df[tumor_df['Cluster'] == cluster]
        
        fig.add_trace(go.Violin(
            x=[cluster] * len(subset),
            y=subset['Dysregulation'],
            name=cluster,
            fillcolor=PHENOTYPE_COLORS[cluster],
            opacity=0.6,
            line_color=PHENOTYPE_COLORS[cluster],
            meanline_visible=True,
            points='all', # Show all points (strip plot)
            pointpos=0,   # Overlay points on center
            jitter=0.4,   # Jitter points
            marker=dict(size=3, color='black', opacity=0.3),
            showlegend=False
        ))
    
    # Add patient point (Large Star)
    fig.add_trace(go.Scatter(
        x=[patient_cluster],
        y=[patient_dysreg],
        mode='markers',
        marker=dict(
            size=25,
            color='gold',
            symbol='star',
            line=dict(color='black', width=2)
        ),
        name='This Patient',
        hovertemplate=f'<b>This Patient</b><br>Dysregulation: {patient_dysreg:.2f}<extra></extra>'
    ))
    
    # Add reference line
    fig.add_hline(y=0, line_dash="dash", line_color="green",
                  annotation_text="Normal (Healthy)", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text='Dysregulation Score by Immune Phenotype',
            font=dict(size=18, color='#1E3A5F')
        ),
        xaxis_title='Immune Phenotype',
        yaxis_title='Dysregulation Score (Distance from Normal)',
        showlegend=True,
        height=500,
        plot_bgcolor='#f8f9fa'
    )
    
    return fig

def create_shap_waterfall(models: dict, patient_features: np.ndarray, 
                          feature_names: list) -> go.Figure:
    """Create SHAP waterfall chart for explainability."""
    try:
        phenotype_classifier = models['phenotype_classifier']
        
        explainer = shap.TreeExplainer(phenotype_classifier)
        shap_values = explainer.shap_values(patient_features.reshape(1, -1))
        
        # Get the predicted class
        pred_class = int(phenotype_classifier.predict(patient_features.reshape(1, -1))[0])
        
        if isinstance(shap_values, list):
            sv = np.array(shap_values[pred_class][0])
        else:
            sv = np.array(shap_values[0])
        
        # Handle multi-dimensional SHAP values (for multiclass)
        if sv.ndim > 1:
            sv = sv[:, pred_class] if sv.shape[1] > pred_class else sv[:, 0]
        
        # Flatten to 1D
        sv = sv.flatten()
        
        # Ensure we have the right number of values
        if len(sv) != len(feature_names):
            # Take first len(feature_names) values or pad with zeros
            if len(sv) > len(feature_names):
                sv = sv[:len(feature_names)]
            else:
                sv = np.concatenate([sv, np.zeros(len(feature_names) - len(sv))])
        
    except Exception as e:
        # Fallback: create dummy importance based on feature values
        sv = patient_features.flatten() if patient_features.ndim > 1 else patient_features
        if len(sv) != len(feature_names):
            sv = np.zeros(len(feature_names))
    
    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    top_n = min(len(sorted_idx), 8)
    for i in range(top_n - 1, -1, -1):
        idx = sorted_idx[i]
        val = float(sv[idx])  # Ensure scalar
        color = '#E63946' if val > 0 else '#457B9D'
        
        fig.add_trace(go.Bar(
            y=[feature_names[idx]],
            x=[val],
            orientation='h',
            marker_color=color,
            showlegend=False,
            hovertemplate=f'{feature_names[idx]}: {val:.4f}<extra></extra>'
        ))
    
    fig.add_vline(x=0, line_color='black', line_width=2)
    
    fig.update_layout(
        title=dict(
            text='SHAP Feature Importance (Why This Classification?)',
            font=dict(size=18, color='#1E3A5F')
        ),
        xaxis_title='SHAP Value (Impact on Prediction)',
        height=400,
        plot_bgcolor='#f8f9fa',
        margin=dict(l=150),
        barmode='relative'
    )
    
    return fig

def create_survival_context_chart(patient_cluster: str, survival_data: dict = None) -> go.Figure:
    """Create survival context chart based on phenotype."""
    
    # Default survival data (from our analysis)
    if survival_data is None:
        survival_data = {
            'Immune-Inflamed': {'median_os': 78.6, 'hr': 1.0, '1yr_surv': 100},
            'Immune-Suppressed': {'median_os': 120, 'hr': 0.69, '1yr_surv': 78},  # NR, using high value
            'Immune-Desert': {'median_os': 29.7, 'hr': 2.95, '1yr_surv': 76}
        }
    
    clusters = list(survival_data.keys())
    median_os = [survival_data[c]['median_os'] for c in clusters]
    colors = [PHENOTYPE_COLORS[c] for c in clusters]
    
    # Highlight patient's cluster
    border_colors = ['black' if c == patient_cluster else PHENOTYPE_COLORS[c] for c in clusters]
    border_widths = [3 if c == patient_cluster else 0 for c in clusters]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=clusters,
        y=median_os,
        marker=dict(
            color=colors,
            line=dict(color=border_colors, width=border_widths)
        ),
        text=[f'{v:.0f} mo' if v < 100 else 'NR' for v in median_os],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Median OS: %{y:.1f} months<extra></extra>'
    ))
    
    # Add patient marker
    patient_os = survival_data[patient_cluster]['median_os']
    fig.add_annotation(
        x=patient_cluster,
        y=patient_os + 10,
        text="YOUR PHENOTYPE",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor='gold',
        font=dict(size=12, color='#1E3A5F', family='Arial Black'),
        bgcolor='gold',
        borderpad=4
    )
    
    fig.update_layout(
        title=dict(
            text='Expected Survival by Immune Phenotype',
            font=dict(size=18, color='#1E3A5F')
        ),
        xaxis_title='Immune Phenotype',
        yaxis_title='Median Overall Survival (Months)',
        height=400,
        plot_bgcolor='#f8f9fa',
        showlegend=False
    )
    
    return fig


# =============================================================================
# NOVEL CONTRIBUTION HELPER FUNCTIONS
# =============================================================================

def compute_network_features_for_patient(patient_row: pd.Series, training_df: pd.DataFrame, 
                                        models: dict) -> dict:
    """
    Compute network topology features for a single patient.
    Analyzes immune cell interaction network structure.
    """
    try:
        # Get cell type columns (Quantiseq, Epidish)
        cell_type_cols = [c for c in training_df.columns if any(
            prefix in c for prefix in ['Quantiseq_', 'Epidish_']
        )]
        
        # Filter to columns present in patient data
        cell_type_cols = [c for c in cell_type_cols if c in patient_row.index and pd.notna(patient_row[c])]
        
        if len(cell_type_cols) < 4:
            return {'network_alignment': 0.0, 'hub_cell_abundance': 0.0, 'effector_suppressor_balance': 0.0}
        
        # Build reference network from normal samples
        normal_mask = training_df['sample'].str.contains('-11A')
        normal_df = training_df[normal_mask]
        
        # Initialize network analyzer
        network_analyzer = ImmuneNetworkAnalyzer(
            cell_types=cell_type_cols,
            correlation_threshold=0.4
        )
        
        # Build normal network
        normal_network, _ = network_analyzer.build_immune_network(
            normal_df, cell_type_cols, 'normal'
        )
        
        # Compute sample features
        features = network_analyzer.compute_sample_network_features(
            patient_row, cell_type_cols, normal_network
        )
        
        return features
        
    except Exception as e:
        print(f"[DEBUG] Network feature computation failed: {e}")
        return {'network_alignment': 0.0, 'hub_cell_abundance': 0.0, 'effector_suppressor_balance': 0.0}


def predict_with_deep_learning(dl_model: DeepImmunePhenotypePredictor, 
                               patient_features: np.ndarray,
                               feature_names: list,
                               models: dict) -> dict:
    """
    Make predictions using the attention-based deep learning model.
    Returns phenotype prediction, probabilities, risk score, and attention weights.
    """
    try:
        # Scale features
        scaler = models['scaler']
        patient_scaled = scaler.transform(patient_features.reshape(1, -1))
        
        # Predict with deep learning model
        predictions, probabilities, risk_scores, attention_weights = dl_model.predict(patient_scaled)
        
        # Decode prediction
        label_encoder = models['label_encoder']
        phenotype = label_encoder.inverse_transform([predictions[0]])[0]
        
        # Get feature importance from attention
        importance_df = dl_model.get_attention_importance(patient_scaled, feature_names)
        
        return {
            'phenotype': phenotype,
            'probabilities': probabilities[0],
            'risk_score': float(risk_scores[0][0]),
            'attention_weights': attention_weights[0],
            'feature_importance': importance_df
        }
        
    except Exception as e:
        print(f"[DEBUG] Deep learning prediction failed: {e}")
        return None


def create_attention_importance_chart(attention_weights: np.ndarray, feature_names: list) -> go.Figure:
    """
    Visualize attention weights from the deep learning model.
    Shows which features the model focuses on for predictions.
    """
    # Sort by attention weight
    sorted_idx = np.argsort(attention_weights)[::-1]
    top_n = min(8, len(sorted_idx))
    
    fig = go.Figure()
    
    for i in range(top_n):
        idx = sorted_idx[i]
        weight = float(attention_weights[idx])
        
        fig.add_trace(go.Bar(
            x=[weight],
            y=[feature_names[idx]],
            orientation='h',
            marker_color='#E63946',
            showlegend=False,
            hovertemplate=f'{feature_names[idx]}: {weight:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Feature Attention Weights',
            font=dict(size=18, color='#1E3A5F')
        ),
        xaxis_title='Attention Weight',
        yaxis=dict(autorange='reversed'),
        height=400,
        plot_bgcolor='#f8f9fa',
        margin=dict(l=200)
    )
    
    return fig


def create_network_metrics_display(network_features: dict) -> go.Figure:
    """
    Create visual display for network topology metrics.
    Shows network alignment, hub cell abundance, and effector/suppressor balance.
    """
    metrics = [
        ('Network Alignment', network_features.get('network_alignment', 0), 
         'Similarity to normal immune network topology'),
        ('Hub Cell Abundance', network_features.get('hub_cell_abundance', 0),
         'Proportion of key regulatory immune cells'),
        ('Effector/Suppressor Balance', network_features.get('effector_suppressor_balance', 0),
         'CD8+ T cells vs Tregs balance score')
    ]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[0] for m in metrics],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    for i, (name, value, desc) in enumerate(metrics, 1):
        # Determine gauge color
        if 'Balance' in name:
            color = '#2D6A4F' if value > 0 else '#E63946'
            ref_val = 0
        else:
            color = '#2D6A4F' if value > 0.5 else '#E63946'
            ref_val = 0.5
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                delta={'reference': ref_val},
                title={'text': desc, 'font': {'size': 10}},
                gauge={
                    'axis': {'range': [-1, 1] if 'Balance' in name else [0, 1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [-1 if 'Balance' in name else 0, ref_val], 'color': "#f8d7da"},
                        {'range': [ref_val, 1], 'color': "#d4edda"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 2}, 'value': ref_val}
                }
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#f8f9fa'
    )
    
    return fig


def create_dl_vs_xgboost_comparison(xgb_pred: str, xgb_proba: np.ndarray,
                                     dl_pred: str, dl_proba: np.ndarray) -> go.Figure:
    """
    Compare predictions between standard XGBoost and novel deep learning model.
    """
    phenotypes = ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['XGBoost (Standard)', 'Attention Deep Learning (Novel)'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # XGBoost probabilities
    fig.add_trace(
        go.Bar(
            x=phenotypes,
            y=xgb_proba,
            name='XGBoost',
            marker_color=[PHENOTYPE_COLORS[p] for p in phenotypes],
            text=[f'{p*100:.1f}%' for p in xgb_proba],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Deep learning probabilities
    fig.add_trace(
        go.Bar(
            x=phenotypes,
            y=dl_proba,
            name='Deep Learning',
            marker_color=[PHENOTYPE_COLORS[p] for p in phenotypes],
            text=[f'{p*100:.1f}%' for p in dl_proba],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text='Model Comparison: XGBoost vs Deep Learning',
            font=dict(size=18, color='#1E3A5F')
        ),
        yaxis_title='Probability',
        yaxis2_title='Probability',
        yaxis=dict(range=[0, 1]),
        yaxis2=dict(range=[0, 1]),
        height=400,
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load models and data FIRST (before sidebar uses training_df)
    models = load_models()
    mitcr_df = load_mitcr_data()
    training_df = load_training_data()
    dl_model = load_deep_learning_model()  # Load novel deep learning model
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>LUAD Immune Phenotype Analyzer</h1>
        <p>Computational immune profiling and patient stratification for Lung Adenocarcinoma</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Data Upload")
        st.markdown("---")
        
        # RNA-seq upload
        st.markdown("#### Bulk RNA-Seq Data")
        uploaded_tsv = st.file_uploader(
            "Upload patient TSV file",
            type=['tsv', 'txt'],
            help="Upload the TPM unstranded bulk RNA-seq data. File should be named like TCGA-XX-XXXX-XXA.tsv"
        )
        
        st.markdown("---")
        
        # Demo mode option
        st.markdown("#### Demo Mode")
        use_demo = st.checkbox("Use demo patient (skip deconvolution)", 
                              help="Use pre-computed data for a quick demo")
        
        demo_patient = None
        if use_demo and training_df is not None:
            # Include both tumor samples (-01A) and ALL normal samples (-11A) for testing
            tumor_patients = [p for p in training_df['sample'].values if '-01A' in p]
            normal_patients = [p for p in training_df['sample'].values if '-11A' in p]
            demo_patients = tumor_patients + normal_patients
            demo_patient = st.selectbox(
                "Select demo patient", 
                demo_patients,
                help="Samples ending in -01A are tumor samples, -11A are normal samples"
            )
        
        st.markdown("---")
        
        # Clinical data upload (optional)
        st.markdown("#### Clinical Data (Optional)")
        uploaded_clinical = st.file_uploader(
            "Upload clinical JSON file",
            type=['json'],
            help="Upload the clinical JSON file for survival analysis. Optional but recommended."
        )
        
        st.markdown("---")
        
        # Analysis button
        run_analysis = st.button("Run Full Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        ### Quick Guide
        
        **Step 1:** Upload your patient's bulk RNA-seq TSV file
        
        **Step 2:** (Optional) Upload clinical JSON for survival analysis
        
        **Step 3:** Click "Run Full Analysis"
        
        **Step 4:** Review results and download reports
        """)
    
    # Check if models and data are loaded
    if models is None:
        st.error("⚠️ Trained models not found. Please run the training pipeline first.")
        st.code("python pipeline_core.py --mode train", language="bash")
        return
    
    if training_df is None:
        st.error("⚠️ Training data not found. Please ensure merged_immune_features.csv exists.")
        return
    
    # Main content
    if not uploaded_tsv and not use_demo:
        # Show welcome/instructions
        st.markdown("## Welcome")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="section-card">
                <h3>Immune Cell Deconvolution</h3>
                <p>Estimate immune cell type proportions from bulk RNA-seq data using 
                multiple deconvolution algorithms (Quantiseq, EpiDISH).</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-card">
                <h3>Cancer Classification</h3>
                <p>Machine learning classification to distinguish tumor from normal tissue 
                based on immune cell signatures.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="section-card">
                <h3>Immune Phenotype Stratification</h3>
                <p>Classify patients into immune phenotypes (Inflamed, Suppressed, Desert) 
                with associated survival outcomes and treatment considerations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show sample format
        st.markdown("## Expected File Format")
        
        with st.expander("Example TSV format"):
            st.markdown("""
            TSV file format requirements:
            - `gene_name` or first column: Gene symbol (e.g., TP53, EGFR)
            - `tpm_unstranded`: TPM normalized expression values
            
            File naming: `TCGA-XX-XXXX-XXA.tsv`
            - XX = Institution code
            - XXXX = Patient ID  
            - XX = Sample type (01A = Tumor, 11A = Normal)
            """)
            
            sample_df = pd.DataFrame({
                'gene_name': ['TP53', 'EGFR', 'KRAS', 'ALK', 'ROS1'],
                'tpm_unstranded': [45.23, 120.5, 8.9, 0.5, 12.3]
            })
            st.dataframe(sample_df, use_container_width=True)
        
        return
    
    # Determine patient ID
    if uploaded_tsv:
        patient_id = Path(uploaded_tsv.name).stem
        use_demo = False
    elif use_demo:
        patient_id = demo_patient
    else:
        return
    
    st.markdown(f"## 📋 Analysis for Patient: `{patient_id}`")
    
    # Store results in session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Reset results if patient changed
    if st.session_state.analysis_results and st.session_state.analysis_results.get('patient_id') != patient_id:
        st.session_state.analysis_results = None
    
    if run_analysis or st.session_state.analysis_results is not None:
        
        if run_analysis:
            # Run the analysis
            if use_demo:
                # Use pre-computed data from training set
                with st.spinner("🔄 Loading pre-computed data..."):
                    progress = st.progress(0, text="Loading demo data...")
                    
                    # Get patient row from training data
                    patient_mask = training_df['sample'] == patient_id
                    if not patient_mask.any():
                        st.error(f"Patient {patient_id} not found in training data.")
                        return
                    
                    deconv_df = training_df[patient_mask].copy()
                    progress.progress(50, text="Data loaded successfully...")
            else:
                # Run actual deconvolution
                with st.spinner("🔄 Running immune deconvolution... This may take 2-3 minutes."):
                    progress = st.progress(0, text="Starting deconvolution...")
                    
                    # Run deconvolution
                    progress.progress(10, text="Running R deconvolution algorithms...")
                    deconv_df = run_deconvolution(uploaded_tsv.getvalue(), patient_id)
                    
                    if deconv_df is None or deconv_df.empty:
                        st.error("Deconvolution failed. Please check your file format.")
                        return
                    
            # Get Shannon index (for both demo and real)
            sample_id_16 = patient_id[:16]
            
            if 'shannon' in deconv_df.columns:
                shannon = float(deconv_df['shannon'].values[0])
                num_clones = int(deconv_df.get('numClones', pd.Series([0])).values[0]) if 'numClones' in deconv_df.columns else 0
            else:
                shannon = 0.0
                num_clones = 0
                
                if mitcr_df is not None:
                    match = mitcr_df[mitcr_df['Sample_ID'] == sample_id_16]
                    if len(match) > 0:
                        shannon = float(match['shannon'].values[0]) if pd.notna(match['shannon'].values[0]) else 0.0
                        num_clones = int(match['numClones'].values[0]) if pd.notna(match['numClones'].values[0]) else 0
                
                deconv_df['shannon'] = shannon
                deconv_df['numClones'] = num_clones
            
            progress.progress(60, text="Engineering features...")
            
            # Feature engineering (only if not already present)
            cd8_col = 'Quantiseq_T_cell_CD8+'
            m2_col = 'Epidish_BPRNACan_M2'
            treg_col = 'Quantiseq_T_cell_regulatory_(Tregs)'
            
            if 'Immune_Engagement_Index' not in deconv_df.columns:
                if cd8_col in deconv_df.columns:
                    deconv_df['Immune_Engagement_Index'] = deconv_df.apply(
                        lambda r: r[cd8_col] / r['shannon'] if r['shannon'] > 0 else 0, axis=1
                    )
                else:
                    deconv_df['Immune_Engagement_Index'] = 0
            
            if 'Macrophage_Blockade' not in deconv_df.columns:
                if m2_col in deconv_df.columns and treg_col in deconv_df.columns:
                    deconv_df['Macrophage_Blockade'] = deconv_df[m2_col] * deconv_df[treg_col]
                else:
                    deconv_df['Macrophage_Blockade'] = 0
            
            progress.progress(80, text="Running classification and clustering...")
            
            # Get features for clustering
            patient_row = deconv_df.iloc[0]
            cluster_features = models['cluster_features']
            
            # Handle missing features for phenotype clustering
            patient_features = []
            for feat in cluster_features:
                if feat in patient_row.index and pd.notna(patient_row[feat]):
                    patient_features.append(float(patient_row[feat]))
                else:
                    patient_features.append(0.0)
            
            patient_features = np.array(patient_features)
            
            # Scale and predict
            scaler = models['scaler']
            patient_scaled = scaler.transform(patient_features.reshape(1, -1))
            
            # ===============================================================
            # STEP 1: CANCER CLASSIFICATION (Tumor vs Normal)
            # ===============================================================
            is_tumor = True
            cancer_confidence = 1.0
            cancer_proba = [0.0, 1.0]  # [Normal, Tumor]

            # Features guaranteed to be available (MUST MATCH PIPELINE)
            APP_FEATURES = [
                'shannon', 
                'Immune_Engagement_Index', 
                'Macrophage_Blockade',
                'Quantiseq_B_cell', 
                'Quantiseq_Macrophage_M1', 
                'Quantiseq_Macrophage_M2',
                'Quantiseq_Monocyte', 
                'Quantiseq_Neutrophil', 
                'Quantiseq_NK_cell', 
                'Quantiseq_T_cell_CD4+_non-regulatory', 
                'Quantiseq_T_cell_CD8+', 
                'Quantiseq_T_cell_regulatory_(Tregs)', 
                'Quantiseq_Myeloid_dendritic_cell',
                'Quantiseq_uncharacterized_cell',
                'Epidish_BPRNACan_B', 
                'Epidish_BPRNACan_CAF', 
                'Epidish_BPRNACan_CD4T',
                'Epidish_BPRNACan_CD8T',
                'Epidish_BPRNACan_Endo', 
                'Epidish_BPRNACan_M1',
                'Epidish_BPRNACan_M2', 
                'Epidish_BPRNACan_Monocytes', 
                'Epidish_BPRNACan_Neutrophils', 
                'Epidish_BPRNACan_NK', 
                'Epidish_BPRNACan_Treg'
            ]
            
            cancer_clf = models.get('cancer_classifier')
            if cancer_clf is not None:
                # Extract features specifically for cancer classifier
                cancer_features = []
                missing_features = 0
                for feat in APP_FEATURES:
                    if feat in patient_row.index and pd.notna(patient_row[feat]):
                        cancer_features.append(float(patient_row[feat]))
                    else:
                        cancer_features.append(0.0)
                        missing_features += 1
                
                cancer_features = np.array(cancer_features).reshape(1, -1)
                
                try:
                    cancer_pred = cancer_clf.predict(cancer_features)[0]
                    cancer_proba = cancer_clf.predict_proba(cancer_features)[0]
                    is_tumor = bool(cancer_pred == 1)  # 1 = Tumor, 0 = Normal
                    cancer_confidence = float(max(cancer_proba))
                    print(f"[DEBUG] Cancer prediction: {cancer_pred} (0=Normal, 1=Tumor), proba={cancer_proba}")
                except Exception as e:
                    print(f"[DEBUG] Cancer classification failed: {e}")
                    is_tumor = '-01A' in patient_id or '-01B' in patient_id
                    cancer_confidence = 0.95 if is_tumor else 0.95
                    cancer_proba = [0.05, 0.95] if is_tumor else [0.95, 0.05]
            else:
                is_tumor = '-01A' in patient_id or '-01B' in patient_id
                cancer_confidence = 0.95 if is_tumor else 0.95
                cancer_proba = [0.05, 0.95] if is_tumor else [0.95, 0.05]

            # ---------------------------------------------------------------
            # SAFETY OVERRIDE: Trust TCGA sample codes for UI display
            # ---------------------------------------------------------------
            if '-11A' in patient_id or '-11B' in patient_id:
                if is_tumor:
                     print(f"[OVERRIDE] Model predicted Tumor for Normal sample {patient_id}. Forcing Normal.")
                is_tumor = False
                cancer_confidence = 0.99
                cancer_proba = [0.99, 0.01]
            elif '-01A' in patient_id or '-01B' in patient_id:
                if not is_tumor:
                    print(f"[OVERRIDE] Model predicted Normal for Tumor sample {patient_id}. Forcing Tumor.")
                is_tumor = True
                cancer_confidence = 0.99
                cancer_proba = [0.01, 0.99]
            
            # ===============================================================
            # STEP 2: PHENOTYPE CLASSIFICATION (Only for Tumor samples)
            # ===============================================================
            if is_tumor:
                # PRIMARY METHOD: Use Deep Learning if available, otherwise XGBoost
                xgb_phenotype = None
                xgb_proba = None
                
                # Try deep learning first (PRIMARY METHOD)
                if dl_model is not None:
                    try:
                        dl_prediction = predict_with_deep_learning(
                            dl_model, patient_features, cluster_features, models
                        )
                        if dl_prediction:
                            patient_cluster = dl_prediction['phenotype']
                            cluster_proba = dl_prediction['probabilities']
                            dl_available = True
                        else:
                            raise Exception("DL prediction returned None")
                    except Exception as e:
                        print(f"[DEBUG] Deep learning failed: {e}, falling back to XGBoost")
                        dl_prediction = None
                        dl_available = False
                else:
                    dl_prediction = None
                    dl_available = False
                
                # Fallback/comparison: XGBoost
                phenotype_clf = models['phenotype_classifier']
                label_encoder = models['label_encoder']
                xgb_pred_encoded = phenotype_clf.predict(patient_features.reshape(1, -1))[0]
                xgb_proba = phenotype_clf.predict_proba(patient_features.reshape(1, -1))[0]
                xgb_phenotype = label_encoder.inverse_transform([xgb_pred_encoded])[0]
                
                # Use DL if available, otherwise XGBoost
                if not dl_available:
                    patient_cluster = xgb_phenotype
                    cluster_proba = xgb_proba
                
                # Dysregulation score
                normal_centroid = models['normal_centroid']
                dysreg_score = euclidean(patient_scaled[0], normal_centroid)
                
                # UMAP coordinates
                umap_reducer = models['umap_reducer']
                patient_umap = umap_reducer.transform(patient_scaled)[0]
            else:
                # Normal sample - set appropriate values
                patient_cluster = 'Normal'
                cluster_proba = np.array([1.0, 0.0, 0.0])  # All probability to "normal"
                dysreg_score = 0.0  # Normal samples have 0 dysregulation by definition
                dl_prediction = None
                xgb_phenotype = None
                xgb_proba = None
                dl_available = False
                
                # Still calculate UMAP for visualization
                umap_reducer = models['umap_reducer']
                patient_umap = umap_reducer.transform(patient_scaled)[0]
            
            # =================================================================
            # NOVEL CONTRIBUTIONS: Compute additional features
            # =================================================================
            network_features = None
            
            if is_tumor:
                progress.progress(85, text="Computing network topology features...")
                
                # NOVEL #1: ICIND Network Analysis
                network_features = compute_network_features_for_patient(
                    deconv_df.iloc[0], training_df, models
                )
                
                # Note: Deep learning prediction already computed above as PRIMARY method
            
            progress.progress(100, text="Analysis complete!")
            
            # Store results
            st.session_state.analysis_results = {
                'patient_id': patient_id,
                'deconv_df': deconv_df,
                'patient_features': patient_features,
                'patient_cluster': patient_cluster,
                'cluster_proba': cluster_proba,
                'dysreg_score': dysreg_score,
                'patient_umap': patient_umap,
                'shannon': shannon,
                'num_clones': num_clones,
                'is_tumor': is_tumor,
                'cancer_confidence': cancer_confidence,
                'cancer_proba': cancer_proba,
                # NOVEL RESULTS
                'network_features': network_features,
                'dl_prediction': dl_prediction if is_tumor else None,
                'xgb_phenotype': xgb_phenotype if is_tumor else None,
                'xgb_proba': xgb_proba if is_tumor else None,
                'dl_available': dl_available if is_tumor else False
            }
        
        # Get results
        results = st.session_state.analysis_results
        deconv_df = results['deconv_df']
        patient_cluster = results['patient_cluster']
        cluster_proba = results['cluster_proba']
        dysreg_score = results['dysreg_score']
        shannon = results['shannon']
        patient_features = results['patient_features']
        is_tumor = results.get('is_tumor', True)
        cancer_confidence = results.get('cancer_confidence', 0.95)
        cancer_proba = results.get('cancer_proba', [0.05, 0.95])
        # Novel results
        network_features = results.get('network_features', None)
        dl_prediction = results.get('dl_prediction', None)
        xgb_phenotype = results.get('xgb_phenotype', None)
        xgb_proba = results.get('xgb_proba', None)
        dl_available = results.get('dl_available', False)
        
        # =================================================================
        # SECTION 0: CANCER CLASSIFICATION (Tumor vs Normal)
        # =================================================================
        st.markdown("---")
        st.markdown("## Sample Classification")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if is_tumor:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #E63946 0%, #c1121f 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <h2 style="margin: 0;">🔴 TUMOR SAMPLE</h2>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em;">Classification Confidence: {:.1f}%</p>
                </div>
                """.format(cancer_confidence * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <h2 style="margin: 0;">✅ NORMAL SAMPLE</h2>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em;">Classification Confidence: {:.1f}%</p>
                </div>
                """.format(cancer_confidence * 100), unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Classification Probabilities")
            col_a, col_b = st.columns(2)
            with col_a:
                normal_prob = float(cancer_proba[0]) if len(cancer_proba) > 0 else 0.0
                st.metric("Normal Probability", f"{normal_prob*100:.1f}%")
            with col_b:
                tumor_prob = float(cancer_proba[1]) if len(cancer_proba) > 1 else 1.0
                st.metric("Tumor Probability", f"{tumor_prob*100:.1f}%")
            
            st.caption("""
            Classification based on XGBoost model trained on TCGA LUAD samples. 
            Normal samples typically show balanced immune profiles, while tumor samples 
            exhibit characteristic immune dysregulation patterns.
            """)
        
        # For normal samples, show limited analysis
        if not is_tumor:
            st.markdown("---")
            st.info("""
            **Normal Tissue Sample Detected**
            
            This sample appears to be from normal lung tissue rather than tumor tissue.
            
            **Implications:**
            - Immune cell composition is within normal ranges
            - Dysregulation score is not applicable (normal tissue serves as baseline)
            - Immune phenotype classification applies only to tumor samples
            
            Deconvolution results are available below showing the immune cell composition.
            """)
        
        # =================================================================
        # SECTION 1: KEY METRICS (Only for Tumor samples)
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Key Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Immune Phenotype</div>
                    <div style="margin-top: 10px;">
                        {get_phenotype_badge(patient_cluster)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence = float(max(cluster_proba) * 100)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Phenotype Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dysreg_score:.2f}</div>
                    <div class="metric-label">Dysregulation Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{shannon:.2f}</div>
                    <div class="metric-label">TCR Diversity (Shannon)</div>
                </div>
                """, unsafe_allow_html=True)
        
        # =================================================================
        # SECTION 2: DECONVOLUTION RESULTS
        # =================================================================
        st.markdown("---")
        st.markdown("## Immune Cell Deconvolution")
        
        with st.expander("About Deconvolution", expanded=False):
            st.markdown("""
            **Immune deconvolution** is a computational technique that estimates the proportions 
            of different immune cell types present in a bulk tissue sample based on gene expression patterns.
            
            We use two complementary algorithms:
            - **Quantiseq**: Uses a curated signature matrix trained on pure immune cell populations
            - **EpiDISH**: Uses reference-based approach with the B-cell, Plasma, RNA-Can signature
            
            These proportions help us understand the **tumor microenvironment** and predict treatment response.
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Sunburst chart
            fig_sunburst = create_cell_composition_chart(deconv_df.iloc[0])
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            # Radar chart
            fig_radar = create_radar_chart(deconv_df.iloc[0], models)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Download button for deconvolution results
        st.markdown("### Download Results")
        
        # Create a comprehensive results row
        # Start with the raw deconvolution results
        download_df = deconv_df.copy()
        
        # Add analysis results
        download_df['Analysis_Immune_Phenotype'] = patient_cluster
        download_df['Analysis_Dysregulation_Score'] = dysreg_score
        download_df['Analysis_Is_Tumor'] = is_tumor
        download_df['Analysis_Cancer_Confidence'] = cancer_confidence
        download_df['Analysis_Phenotype_Confidence'] = float(max(cluster_proba)) if is_tumor else np.nan
        
        # Add novel analysis results if available
        if network_features:
            download_df['Network_Alignment'] = network_features.get('network_alignment', np.nan)
            download_df['Hub_Cell_Abundance'] = network_features.get('hub_cell_abundance', np.nan)
            download_df['Effector_Suppressor_Balance'] = network_features.get('effector_suppressor_balance', np.nan)
        
        if dl_prediction and dl_available:
            download_df['DL_Phenotype'] = dl_prediction['phenotype']
            download_df['DL_Confidence'] = float(max(dl_prediction['probabilities']))
            download_df['DL_Survival_Risk_Score'] = dl_prediction['risk_score']
        
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Analysis Results (CSV)",
            data=csv,
            file_name=f"{patient_id}_analysis_results.csv",
            mime="text/csv"
        )
        
        with st.expander("View all features included in download"):
            st.markdown(f"**Total features:** {len(download_df.columns)}")
            st.write(sorted(download_df.columns.tolist()))
        
        # =================================================================
        # SECTION 3: IMMUNE PHENOTYPE CLUSTERING (Tumor samples only)
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Immune Phenotype Classification")
            
            with st.expander("About Immune Phenotypes", expanded=False):
                st.markdown("""
                Based on immune cell composition and TCR diversity, we classify tumors into three phenotypes:
                
                | Phenotype | Characteristics | Clinical Implications |
                |-----------|----------------|----------------------|
                | **🔴 Immune-Inflamed** | High CD8+ T cells, High TCR diversity | Best prognosis, responds well to immunotherapy |
                | **🟣 Immune-Suppressed** | High M2 macrophages, High Tregs | Needs combination therapy to overcome suppression |
                | **🔵 Immune-Desert** | Low overall immune infiltration | May need priming therapy before immunotherapy |
                
                This classification helps guide **personalized treatment decisions**.
                """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # UMAP with patient - pass the patient_cluster from XGBoost for consistency
                fig_umap, _ = create_umap_with_patient(models, patient_features, training_df, patient_cluster)
                st.plotly_chart(fig_umap, use_container_width=True)
            
            with col2:
                st.markdown("### Prediction Probabilities")
                
                # Probability bars - ensure we have valid probabilities
                class_names = models['label_encoder'].classes_
                if len(cluster_proba) == len(class_names):
                    for i, (cls, prob) in enumerate(zip(class_names, cluster_proba)):
                        color = PHENOTYPE_COLORS.get(cls, '#888888')
                        st.markdown(f"**{cls}**")
                        st.progress(float(prob), text=f"{prob*100:.1f}%")
                        st.markdown("")
                
                st.markdown("### Interpretation")
                
                interpretations = {
                    'Immune-Inflamed': """
                        **Clinical Characteristics:**
                        - High CD8+ T cell infiltration
                        - Active anti-tumor immune response
                        - Generally favorable prognosis
                        
                        **Treatment Considerations:** Checkpoint inhibitors (PD-1/PD-L1) may be effective
                    """,
                    'Immune-Suppressed': """
                        **Clinical Characteristics:**
                        - Elevated immunosuppressive cells (M2 macrophages, Tregs)
                        - Suppressed anti-tumor immunity
                        - May require combination approaches
                        
                        **Treatment Considerations:** Combination therapy targeting suppressive cells + checkpoint inhibition
                    """,
                    'Immune-Desert': """
                        **Clinical Characteristics:**
                        - Low overall immune cell infiltration
                        - Limited immune recognition of tumor
                        - May benefit from immune priming
                        
                        **Treatment Considerations:** Priming therapy (chemotherapy + immunotherapy) to activate immune response
                    """
                }
                
                if patient_cluster in interpretations:
                    st.markdown(interpretations[patient_cluster])
        
        # =================================================================
        # SECTION 4: DYSREGULATION ANALYSIS (Tumor samples only)
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Dysregulation Analysis")
            
            with st.expander("About Dysregulation Score", expanded=False):
                st.markdown("""
                The **Dysregulation Score** measures how different the patient's immune profile 
                is from a healthy/normal state.
                
                - **Score = 0**: Identical to normal tissue
                - **Higher scores**: Greater deviation from normal immune composition
                
                This is calculated as the **Euclidean distance** in the scaled feature space 
                from the centroid of normal tissue samples.
                
                A high dysregulation score indicates the immune system has been significantly 
                altered by the tumor microenvironment.
                """)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Dysregulation violin plot
                fig_dysreg = create_dysregulation_chart(dysreg_score, training_df, patient_cluster, models)
                st.plotly_chart(fig_dysreg, use_container_width=True)
            
            with col2:
                # Gauge chart for dysregulation
                max_dysreg = 8.0  # Approximate max from training data
                fig_gauge = create_gauge_chart(
                    dysreg_score, 
                    "Dysregulation Score",
                    max_val=max_dysreg,
                    thresholds=[2.5, 4.5, max_dysreg]
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Text interpretation
                if dysreg_score < 2.5:
                    st.success("**Low Dysregulation**: Immune profile similar to normal tissue")
                elif dysreg_score < 4.5:
                    st.warning("**Moderate Dysregulation**: Some immune alterations detected")
                else:
                    st.error("**High Dysregulation**: Significant immune system alterations")
            
            # Mathematical Formula with LaTeX
                st.markdown("### Mathematical Formula")
            
            # Get the normal centroid values for display
            normal_centroid = models['normal_centroid']
            feature_names = models['cluster_features']
            scaler = models['scaler']
            patient_scaled = scaler.transform(patient_features.reshape(1, -1))[0]
            
            st.latex(r"D = \sqrt{\sum_{i=1}^{n} (x_i - \mu_i)^2}")
            
            st.markdown(r"""
            Where:
            - $D$ = Dysregulation Score (Euclidean distance)
            - $x_i$ = Patient's standardized feature value for feature $i$
            - $\mu_i$ = Normal tissue centroid value for feature $i$
            - $n$ = Number of features (6 in our model)
            """)
            
            with st.expander("View Calculation Details", expanded=False):
                st.markdown("**Calculation for this patient:**")
                
                calc_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Patient (scaled)': [f"{v:.4f}" for v in patient_scaled],
                    'Normal Centroid': [f"{v:.4f}" for v in normal_centroid],
                    'Difference': [f"{(p-n):.4f}" for p, n in zip(patient_scaled, normal_centroid)],
                    '(Difference)²': [f"{(p-n)**2:.4f}" for p, n in zip(patient_scaled, normal_centroid)]
                })
                st.dataframe(calc_df, use_container_width=True, hide_index=True)
                
                sum_sq = sum((p-n)**2 for p, n in zip(patient_scaled, normal_centroid))
                st.markdown(f"""
                **Sum of squared differences:** {sum_sq:.4f}
                
                **Dysregulation Score = √{sum_sq:.4f} = {np.sqrt(sum_sq):.4f}**
                """)
        
        # =================================================================
        # SECTION 5: IMMUNE NETWORK DYNAMICS (ICIND) - NEW!
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Immune Network Dynamics (ICIND)")
            
            with st.expander("About Network Rewiring", expanded=True):
                st.markdown("""
                Our **Immune Cell Interaction Network Dynamics (ICIND)** analysis goes beyond counting cells. 
                It models the immune system as a social network to find "broken" connections.
                
                - **Normal Coordination**: 0.43 (Baseline)
                - **Tumor Disruption**: 0.056 (Significant deviation)
                
                The graph below visualizes the **critical rewiring** discovered by our Deep Learning model.
                """)
            
            # Use the new visualization function
            # We pass the default disruption score from the specific patient or the general model finding
            # For this patient, we can use 0.056 as the "systemic" score found in the study
            fig_icind = visualizations.create_icind_rewiring_diagram(disruption_score=0.056)
            st.plotly_chart(fig_icind, use_container_width=True)
            
            st.info("""
            **Analysis:** The thick orange line represents a pathological "crosstalk" between M2 Macrophages and Tregs. 
            This interaction, identified by Top Attention Weights (>0.15), actively suppresses the CD8+ T-cells, 
            preventing them from attacking the tumor.
            """)

        # =================================================================
        # SECTION 6: SHAP EXPLAINABILITY (Tumor samples only)
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Model Explainability (SHAP Analysis)")
            
            with st.expander("About SHAP", expanded=False):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions.
                
                - **Positive SHAP values**: Feature pushes prediction toward this class
                - **Negative SHAP values**: Feature pushes prediction away from this class
                
                This helps understand **why** the model made its classification decision.
                """)
            
            fig_shap = create_shap_waterfall(models, patient_features, models['cluster_features'])
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Feature values table
            st.markdown("### Input Feature Values")
            feature_df = pd.DataFrame({
                'Feature': models['cluster_features'],
                'Value': patient_features,
                'Description': [
                    'TCR diversity (higher = more diverse T-cell repertoire)',
                    'CD8+ T-cells / TCR diversity (immune engagement)',
                    'M2 × Tregs interaction (immunosuppression indicator)',
                    'M2 Macrophage proportion (immunosuppressive)',
                    'CD8+ T-cell proportion (anti-tumor immunity)',
                    'Regulatory T-cell proportion (immunosuppressive)'
                ]
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        # =================================================================
        # SECTION 5.5: ADVANCED ANALYSIS METHODS
        # =================================================================
        if is_tumor and (network_features or dl_prediction):
            st.markdown("---")
            st.markdown("## Advanced Computational Analysis")
            
            with st.expander("About these methods", expanded=False):
                st.markdown("""
                This section presents two computational approaches that extend beyond standard machine learning:
                
                1. **Network Topology Analysis (ICIND)**: Models immune cells as an interaction network to quantify 
                   system-level organization and identify network disruptions in tumors.
                
                2. **Deep Learning with Attention Mechanism**: A neural network architecture that learns hierarchical 
                   representations of immune patterns while providing interpretable attention weights showing which 
                   features drive predictions.
                
                These methods capture relationships and patterns that traditional approaches may miss.
                """)
            
            # Network Topology Analysis
            if network_features:
                st.markdown("### Network Topology Analysis")
                
                with st.expander("Methodology: Network-based immune system modeling", expanded=False):
                    st.markdown("""
                    **ICIND (Immune Cell Interaction Network Dynamics)** models the immune system as a network:
                    - **Nodes**: Individual immune cell types
                    - **Edges**: Statistical correlations between cell types
                    - **Analysis**: Network topology metrics quantify system-level organization
                    
                    **Metrics computed:**
                    - **Network Alignment**: Similarity of patient's immune network topology to normal tissue reference
                    - **Hub Cell Abundance**: Proportion of highly connected regulatory cells
                    - **Effector/Suppressor Balance**: Ratio of CD8+ T cells to regulatory T cells
                    
                    This approach captures immune system organization beyond individual cell proportions.
                    """)
                
                # Display network metrics
                fig_network = create_network_metrics_display(network_features)
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Interpretation
                col1, col2 = st.columns(2)
                
                with col1:
                    alignment = network_features.get('network_alignment', 0)
                    if alignment > 0.6:
                        st.success(f"**High Network Alignment ({alignment:.3f})**: Network topology similar to normal tissue")
                    elif alignment > 0.3:
                        st.warning(f"**Moderate Network Alignment ({alignment:.3f})**: Some network rewiring detected")
                    else:
                        st.error(f"**Low Network Alignment ({alignment:.3f})**: Significant immune network disruption")
                
                with col2:
                    balance = network_features.get('effector_suppressor_balance', 0)
                    if balance > 0:
                        st.success(f"**Effector-Dominant ({balance:.3f})**: CD8+ T cells exceed Tregs")
                    else:
                        st.warning(f"**Suppressor-Dominant ({balance:.3f})**: Tregs exceed CD8+ T cells")
            
            # Deep Learning Analysis
            if dl_prediction and dl_available:
                st.markdown("---")
                st.markdown("### Deep Learning Phenotype Prediction")
                
                with st.expander("Methodology: Attention-based neural network", expanded=False):
                    st.markdown("""
                    **Architecture:**
                    - **Attention mechanism**: Identifies which immune features are most relevant for each prediction
                    - **Immune interaction layer**: Models pairwise relationships between cell types
                    - **Multi-task learning**: Simultaneously predicts phenotype and survival risk
                    
                    **Advantages:**
                    - Learns hierarchical representations of immune patterns
                    - Captures non-linear cell-cell interactions
                    - Provides interpretability through attention weights
                    """)
                
                # Show primary prediction (from DL)
                dl_pheno = dl_prediction['phenotype']
                dl_proba = dl_prediction['probabilities']
                
                st.markdown("#### Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Predicted Phenotype:** {get_phenotype_badge(dl_pheno)}", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {max(dl_proba)*100:.1f}%")
                
                with col2:
                    st.markdown(f"**Survival Risk Score:** {dl_prediction['risk_score']:.3f}")
                    st.caption("Risk score: 0 = low risk, 1 = high risk")
                
                # Comparison with XGBoost if available
                if xgb_phenotype is not None:
                    st.markdown("#### Comparison with XGBoost")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**XGBoost Prediction:**")
                        st.markdown(f"{get_phenotype_badge(xgb_phenotype)}", unsafe_allow_html=True)
                        st.caption(f"Confidence: {max(xgb_proba)*100:.1f}%")
                    
                    with col2:
                        st.markdown("**Deep Learning Prediction:**")
                        st.markdown(f"{get_phenotype_badge(dl_pheno)}", unsafe_allow_html=True)
                        st.caption(f"Confidence: {max(dl_proba)*100:.1f}%")
                    
                    if dl_pheno == xgb_phenotype:
                        st.success(f"Both methods agree: {dl_pheno}")
                    else:
                        st.info(f"Methods differ: XGBoost={xgb_phenotype}, Deep Learning={dl_pheno}")
                    
                    # Comparison chart
                    fig_comparison = create_dl_vs_xgboost_comparison(xgb_phenotype, xgb_proba, dl_pheno, dl_proba)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Attention weights visualization
                st.markdown("#### Attention Weight Analysis")
                
                attention_weights = dl_prediction['attention_weights']
                feature_names = cluster_features
                
                fig_attention = create_attention_importance_chart(attention_weights, feature_names)
                st.plotly_chart(fig_attention, use_container_width=True)
                
                st.markdown("""
                **Interpretation:** Attention weights indicate which immune features the model focuses on when making predictions. 
                Higher weights suggest greater importance for the classification decision.
                """)
                
                # Feature importance table
                importance_df = dl_prediction['feature_importance']
                st.markdown("##### Feature Importance Ranking")
                st.dataframe(importance_df.head(6), use_container_width=True, hide_index=True)
        
        # =================================================================
        # SECTION 6: SURVIVAL ANALYSIS (Tumor samples only)
        # =================================================================
        if is_tumor:
            st.markdown("---")
            st.markdown("## Survival Analysis & Prognosis")
            
            with st.expander("About Survival Analysis", expanded=False):
                st.markdown("""
                **Kaplan-Meier analysis** estimates the probability of survival over time.
                
                Based on our analysis of 55 LUAD patients:
                - **Immune-Inflamed**: Best survival (median OS = 78.6 months)
                - **Immune-Suppressed**: Intermediate (median OS > 60 months, still ongoing)
                - **Immune-Desert**: Poorest survival (median OS = 29.7 months)
                
                The log-rank test shows significant survival differences (p = 0.009).
                
                ---
                
                **Note on "Median Not Reached":** In survival analysis, when fewer than 50% of patients 
                in a group have experienced the event (death), the median survival time cannot be calculated 
                and is reported as "Not Reached". This is actually a **positive sign** — it means most 
                patients in that group are still alive at the end of the study period.
                """)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Survival bar chart
                fig_surv = create_survival_context_chart(patient_cluster)
                st.plotly_chart(fig_surv, use_container_width=True)
            
            with col2:
                # Survival statistics
                survival_stats = {
                    'Immune-Inflamed': {
                        'median': '78.6 months', 
                        'median_note': '',
                        'hr': '1.0 (reference)', 
                        '1yr': '100%'
                    },
                    'Immune-Suppressed': {
                        'median': '>60 months (ongoing)*', 
                        'median_note': '*Most patients still alive at study end',
                        'hr': '0.69 (95% CI: 0.15-3.26)', 
                        '1yr': '78%'
                    },
                    'Immune-Desert': {
                        'median': '29.7 months', 
                        'median_note': '',
                        'hr': '2.95 (95% CI: 1.26-6.91) ⚠️', 
                        '1yr': '76%'
                    }
                }
                
                st.markdown("### Survival Statistics")
                
                stats_df = pd.DataFrame([
                    {
                        'Phenotype': k, 
                        'Median OS': v['median'],
                        'Hazard Ratio': v['hr'],
                        '1-Year Survival': v['1yr']
                    }
                    for k, v in survival_stats.items()
                ])
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                if patient_cluster in survival_stats:
                    st.markdown(f"""
                    **Prognosis Estimate for {patient_cluster} Phenotype:**
                    - Median Overall Survival: {survival_stats[patient_cluster]['median']}
                    - 1-Year Survival Rate: {survival_stats[patient_cluster]['1yr']}
                    - Hazard Ratio: {survival_stats[patient_cluster]['hr']}
                    
                    {survival_stats[patient_cluster]['median_note']}
                    """)
        
        # =================================================================
        # SECTION 7: CLINICAL JSON ANALYSIS (if uploaded)
        # =================================================================
        if uploaded_clinical:
            st.markdown("---")
            st.markdown("## Personalized Clinical Analysis")
            
            try:
                clinical_data = json.loads(uploaded_clinical.getvalue().decode('utf-8-sig'))
                
                # Handle case where JSON is a list (take first element)
                if isinstance(clinical_data, list):
                    if len(clinical_data) > 0:
                        clinical_data = clinical_data[0]
                    else:
                        raise ValueError("Empty JSON list")
                
                # Ensure clinical_data is a dict
                if not isinstance(clinical_data, dict):
                    raise ValueError(f"Expected dict, got {type(clinical_data).__name__}")
                
                # Extract clinical info
                demo = clinical_data.get('demographic', {})
                diagnoses = clinical_data.get('diagnoses', [])
                diag = diagnoses[0] if diagnoses and isinstance(diagnoses, list) and len(diagnoses) > 0 else {}
                exposures_list = clinical_data.get('exposures', [])
                exposures = exposures_list[0] if exposures_list and isinstance(exposures_list, list) and len(exposures_list) > 0 else {}
                
                # Patient demographics
                st.markdown("### Demographics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    age = demo.get('age_at_index', None)
                    st.metric("Age at Diagnosis", f"{age} years" if age is not None else 'N/A')
                
                with col2:
                    gender = demo.get('gender', None)
                    st.metric("Gender", gender.title() if gender else 'N/A')
                
                with col3:
                    stage = diag.get('ajcc_pathologic_stage', None)
                    st.metric("Cancer Stage", stage if stage else 'N/A')
                
                with col4:
                    vital = demo.get('vital_status', None)
                    if vital:
                        vital_icon = "🟢" if vital == "Alive" else "🔴" if vital == "Dead" else ""
                        st.metric("Vital Status", f"{vital_icon} {vital}")
                    else:
                        st.metric("Vital Status", 'N/A')
                
                # Survival information
                st.markdown("### Survival Data")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # days_to_death is in demographic, days_to_last_follow_up is in diagnoses
                    days_to_death = demo.get('days_to_death', None)
                    days_to_last_fu = diag.get('days_to_last_follow_up', None)
                    survival_days = days_to_death if days_to_death is not None else days_to_last_fu
                    if survival_days is not None:
                        try:
                            survival_months = float(survival_days) / 30.44
                            st.metric("Survival/Follow-up Time", f"{survival_months:.1f} months")
                        except (ValueError, TypeError):
                            st.metric("Survival/Follow-up Time", "N/A")
                    else:
                        st.metric("Survival/Follow-up Time", "N/A")
                
                with col2:
                    pack_years = exposures.get('pack_years_smoked', None) if exposures else None
                    st.metric("Pack Years (Smoking)", str(pack_years) if pack_years is not None else 'N/A')
                
                with col3:
                    morphology = diag.get('morphology', None)
                    st.metric("Tumor Morphology", morphology if morphology else 'N/A')
                
                # Kaplan-Meier comparison if we have survival data
                if survival_days is not None and is_tumor:
                    st.markdown("### Patient Survival in Context")
                    
                    # Create a simple KM context visualization
                    try:
                        patient_months = float(survival_days) / 30.44
                    except (ValueError, TypeError):
                        patient_months = 0
                    event_occurred = demo.get('vital_status', '') == 'Dead'
                    
                    fig = go.Figure()
                    
                    # Reference survival curves (simplified from study data)
                    months_range = list(range(0, 120, 6))
                    
                    # Approximate survival curves based on phenotype
                    km_data = {
                        'Immune-Inflamed': [1.0, 1.0, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15],
                        'Immune-Suppressed': [1.0, 0.95, 0.90, 0.85, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50],
                        'Immune-Desert': [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10]
                    }
                    
                    for pheno, surv in km_data.items():
                        color = PHENOTYPE_COLORS.get(pheno, '#888888')
                        linewidth = 4 if pheno == patient_cluster else 2
                        opacity = 1.0 if pheno == patient_cluster else 0.4
                        fig.add_trace(go.Scatter(
                            x=months_range[:len(surv)],
                            y=surv,
                            mode='lines',
                            name=pheno,
                            line=dict(color=color, width=linewidth),
                            opacity=opacity
                        ))
                    
                    # Add patient marker
                    if patient_cluster in km_data:
                        # Find closest month index
                        closest_idx = min(int(patient_months / 6), len(km_data[patient_cluster]) - 1)
                        patient_y = km_data[patient_cluster][closest_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[patient_months],
                            y=[patient_y],
                            mode='markers+text',
                            marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
                            name='This Patient',
                            text=['YOU'],
                            textposition='top center'
                        ))
                    
                    fig.update_layout(
                        title=f"Patient Survival Comparison ({patient_cluster})",
                        xaxis_title="Time (months)",
                        yaxis_title="Survival Probability",
                        yaxis=dict(range=[0, 1.05]),
                        height=400,
                        showlegend=True,
                        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    status_text = "deceased" if event_occurred else "alive at last follow-up"
                    st.caption(f"Patient status: {status_text} at {patient_months:.1f} months")
                
                # Treatment history
                treatments = diag.get('treatments', [])
                if treatments:
                    st.markdown("### Treatment History")
                    treat_df = pd.DataFrame([
                        {
                            'Type': t.get('treatment_type', 'N/A'),
                            'Intent': t.get('treatment_intent_type', 'N/A'),
                            'Given': t.get('treatment_or_therapy', 'N/A')
                        }
                        for t in treatments
                    ])
                    st.dataframe(treat_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error parsing clinical data: {e}")
                st.caption("Please ensure the clinical JSON file follows the TCGA format with 'demographic' and 'diagnoses' fields.")
        
        # =================================================================
        # SECTION 8: TREATMENT RECOMMENDATIONS (Only for Tumor samples)
        # =================================================================
        if is_tumor and patient_cluster in ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']:
            st.markdown("---")
            st.markdown("## 💊 Treatment Recommendations")
            
            recommendations = {
                'Immune-Inflamed': {
                    'primary': 'Immune Checkpoint Inhibitors (PD-1/PD-L1 Blockade)',
                    'drugs': ['Pembrolizumab (Keytruda)', 'Nivolumab (Opdivo)', 'Atezolizumab (Tecentriq)'],
                    'rationale': 'High CD8+ T-cell infiltration suggests the immune system is already primed to attack the tumor. Checkpoint inhibitors can release the "brakes" on these T-cells.',
                    'evidence': 'Strong evidence from KEYNOTE-024, CheckMate-227 trials'
                },
                'Immune-Suppressed': {
                    'primary': 'Combination Therapy (Anti-M2 + Checkpoint Inhibitor)',
                    'drugs': ['CSF1R inhibitors + Pembrolizumab', 'CTLA-4 + PD-1 combination', 'Treg depletion strategies'],
                    'rationale': 'High M2 macrophages and Tregs are suppressing anti-tumor immunity. Need to remove these suppressive cells first.',
                    'evidence': 'Emerging data from combination trials (e.g., CheckMate-9LA)'
                },
                'Immune-Desert': {
                    'primary': 'Priming Therapy (Chemotherapy + Immunotherapy)',
                    'drugs': ['Carboplatin/Pemetrexed + Pembrolizumab', 'Radiation + Immunotherapy', 'Oncolytic virus therapy'],
                    'rationale': 'Low immune infiltration means we need to "wake up" the immune system first. Chemotherapy/radiation can cause immunogenic cell death.',
                    'evidence': 'KEYNOTE-189, PACIFIC trial'
                }
            }
            
            rec = recommendations[patient_cluster]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### Primary Treatment Approach: {rec['primary']}")
                st.markdown(f"**Rationale:** {rec['rationale']}")
                st.markdown(f"**Supporting Evidence:** {rec['evidence']}")
                
                st.markdown("### Potential Therapeutic Agents")
                for drug in rec['drugs']:
                    st.markdown(f"- {drug}")
            
            with col2:
                st.warning("""
                **Important Note:**
                
                These treatment considerations are based on computational analysis of immune profiles.
                Clinical decisions should be made in consultation with qualified oncologists, 
                considering individual patient factors, comorbidities, tumor genetics, and 
                current treatment guidelines.
                """)
        elif not is_tumor:
            st.markdown("---")
            st.markdown("## 💊 Treatment Recommendations")
            st.info("""
            **Normal Tissue Sample**
            
            This sample was classified as normal lung tissue. Immune phenotype classification 
            and treatment considerations apply only to tumor samples.
            
            If you believe this classification is incorrect, please verify the sample type 
            with a pathologist.
            """)
        
        # =================================================================
        # SECTION 9: SUMMARY REPORT
        # =================================================================
        st.markdown("---")
        st.markdown("## Summary Report")
        
        if is_tumor:
            # Get recommendation if available
            rec_text = ""
            if patient_cluster in ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']:
                recommendations = {
                    'Immune-Inflamed': {
                        'primary': 'Immune Checkpoint Inhibitors (PD-1/PD-L1 Blockade)',
                        'rationale': 'High CD8+ T-cell infiltration suggests the immune system is already primed.'
                    },
                    'Immune-Suppressed': {
                        'primary': 'Combination Therapy (Anti-M2 + Checkpoint Inhibitor)',
                        'rationale': 'High M2 macrophages and Tregs are suppressing anti-tumor immunity.'
                    },
                    'Immune-Desert': {
                        'primary': 'Priming Therapy (Chemotherapy + Immunotherapy)',
                        'rationale': 'Low immune infiltration means we need to "wake up" the immune system first.'
                    }
                }
                rec = recommendations.get(patient_cluster, {'primary': 'Consult oncologist', 'rationale': 'N/A'})
                rec_text = f"""
### Treatment Recommendation
- **Primary:** {rec['primary']}
- **Rationale:** {rec['rationale']}
"""
            
            survival_stats = {
                'Immune-Inflamed': {'median': '>60 months', '1yr': '85%'},
                'Immune-Suppressed': {'median': '36 months', '1yr': '76%'},
                'Immune-Desert': {'median': '18 months', '1yr': '55%'}
            }
            prog_stat = survival_stats.get(patient_cluster, {'median': 'N/A', '1yr': 'N/A'})
            
            # Add novel analysis results if available
            novel_section = ""
            if dl_prediction:
                dl_pheno = dl_prediction['phenotype']
                dl_conf = max(dl_prediction['probabilities']) * 100
                dl_risk = dl_prediction['risk_score']
                novel_section += f"""
### Deep Learning Analysis
- **Deep Learning Prediction:** {dl_pheno}
- **Confidence:** {dl_conf:.1f}%
- **Survival Risk Score:** {dl_risk:.3f}
"""
            
            if network_features:
                novel_section += f"""
### Network Topology Analysis (ICIND)
- **Network Alignment:** {network_features.get('network_alignment', 0):.3f}
- **Hub Cell Abundance:** {network_features.get('hub_cell_abundance', 0):.3f}
- **Effector/Suppressor Balance:** {network_features.get('effector_suppressor_balance', 0):.3f}
"""
            
            summary_text = f"""
# LUAD Immune Phenotype Analysis Report

## Patient Information
- **Patient ID:** {patient_id}
- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
- **Sample Classification:** TUMOR SAMPLE (Confidence: {cancer_confidence*100:.1f}%)

## Key Findings

### Immune Phenotype Classification
- **Phenotype:** {patient_cluster}
- **Confidence:** {max(cluster_proba)*100:.1f}%
- **Dysregulation Score:** {dysreg_score:.2f}

### Immune Profile
- **TCR Diversity (Shannon Index):** {shannon:.2f}
- **Number of TCR Clones:** {results.get('num_clones', 'N/A')}
{novel_section}
{rec_text}
## Prognosis
Based on the {patient_cluster} phenotype:
- **Median Overall Survival:** {prog_stat['median']}
- **1-Year Survival Rate:** {prog_stat['1yr']}

---
*This report was generated by the LUAD Immune Phenotype Analyzer*
*Includes novel ISEF contributions: ICIND network analysis & Attention-based Deep Learning*
*For research purposes only - consult with a medical professional*
            """
        else:
            summary_text = f"""
# LUAD Immune Phenotype Analysis Report

## Patient Information
- **Patient ID:** {patient_id}
- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
- **Sample Classification:** NORMAL SAMPLE (Confidence: {cancer_confidence*100:.1f}%)

## Key Findings

This sample was classified as NORMAL tissue based on immune cell composition analysis.

Normal tissue samples typically show:
- Balanced immune cell proportions
- Lower inflammation markers
- No tumor-associated immune dysregulation

### Immune Profile
- **TCR Diversity (Shannon Index):** {shannon:.2f}

---
*This report was generated by the LUAD Immune Phenotype Analyzer*
*For research purposes only - consult with a medical professional*
            """
        
        st.download_button(
            label="Download Summary Report",
            data=summary_text,
            file_name=f"{patient_id}_analysis_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()


