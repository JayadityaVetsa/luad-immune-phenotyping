"""
LUAD Immune Deconvolution & Classification Pipeline
====================================================
A production-ready pipeline for:
1. Immune cell deconvolution from bulk RNA-seq
2. Cancer classification using immune signatures
3. Patient stratification into immune phenotypes
4. Therapeutic recommendations

Author: Jayaditya (ISEF Project)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

# ML & Stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, silhouette_samples
)
from scipy.stats import f_oneway, ttest_ind, bootstrap
from scipy.spatial.distance import euclidean
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Dimensionality Reduction
import umap

# SHAP for explainability
import shap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Central configuration for the pipeline."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    TCGA_DIR = BASE_DIR / "TCGA_Real_data"
    CLINICAL_DIR = BASE_DIR / "patient_clinical_data"
    MODELS_DIR = BASE_DIR / "trained_models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Key immune features for clustering (biologically relevant)
    CLUSTER_FEATURES = [
        # 'shannon',                      # Removed to avoid artifact (imputed as 0 for new samples)
        # 'Immune_Engagement_Index',      # Removed as it depends on shannon
        'Macrophage_Blockade',
        'Epidish_BPRNACan_M2',
        'Quantiseq_T_cell_CD8+',
        'Quantiseq_T_cell_regulatory_(Tregs)',
        'Quantiseq_Macrophage_M2',        # Added robust cell types
        'Quantiseq_B_cell'                # Added robust cell types
    ]
    
    # Features guaranteed to be available in the App
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
    
    # Extended features for classification
    CLASSIFICATION_EXCLUDE = ['sample', 'Sample_ID', 'Label']
    
    # Cluster names with biological meaning
    CLUSTER_NAMES = {
        'inflamed': 'Immune-Inflamed',
        'suppressed': 'Immune-Suppressed', 
        'desert': 'Immune-Desert'
    }
    
    # Treatment recommendations
    TREATMENT_MAP = {
        'Immune-Inflamed': {
            'primary': 'Checkpoint Inhibitor (PD-1/PD-L1)',
            'rationale': 'High CD8+ infiltration indicates immunotherapy responsiveness',
            'alternatives': ['Pembrolizumab', 'Nivolumab']
        },
        'Immune-Suppressed': {
            'primary': 'Combination Therapy (Anti-M2 + Checkpoint)',
            'rationale': 'M2 macrophage depletion may restore anti-tumor immunity',
            'alternatives': ['CSF1R inhibitors + PD-1', 'CTLA-4 + PD-1']
        },
        'Immune-Desert': {
            'primary': 'Priming Therapy (Chemotherapy + Immunotherapy)',
            'rationale': 'Low immune infiltration requires immune activation first',
            'alternatives': ['Carboplatin/Pemetrexed + Pembrolizumab', 'Radiation + Immunotherapy']
        }
    }
    
    # Visualization settings
    PALETTE = {
        'Immune-Inflamed': '#E63946',    # Vibrant Red
        'Immune-Suppressed': '#6A0572',   # Deep Purple
        'Immune-Desert': '#457B9D'        # Steel Blue
    }
    
    # Random state for reproducibility
    RANDOM_STATE = 42
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        (cls.RESULTS_DIR / "plots").mkdir(exist_ok=True)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
    def load_merged_features(self, filepath: str = None) -> pd.DataFrame:
        """Load the merged immune features dataset."""
        if filepath is None:
            filepath = self.config.BASE_DIR / "merged_immune_features.csv"
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    def load_mitcr_data(self, filepath: str = None) -> pd.DataFrame:
        """Load TCR statistics for Shannon index lookup."""
        if filepath is None:
            filepath = self.config.BASE_DIR / "mitcr_sampleStatistics_20160714.tsv"
        
        df = pd.read_csv(filepath, sep='\t')
        # Create sample ID column (first 16 chars of AliquotBarcode)
        df['Sample_ID'] = df['AliquotBarcode'].str[:16]
        return df[['Sample_ID', 'shannon', 'numClones']].drop_duplicates(subset=['Sample_ID'])
    
    def split_tumor_normal(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into tumor (-01A) and normal (-11A) samples."""
        tumor_df = df[df['sample'].str.contains('-01A')].copy()
        normal_df = df[df['sample'].str.contains('-11A')].copy()
        
        print(f"Tumor samples: {len(tumor_df)}, Normal samples: {len(normal_df)}")
        return tumor_df, normal_df
    
    def create_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary label: 1=Tumor, 0=Normal."""
        df = df.copy()
        df['Label'] = df['sample'].apply(
            lambda x: 1 if '-01A' in str(x) else (0 if '-11A' in str(x) else np.nan)
        )
        df = df.dropna(subset=['Label'])
        df['Label'] = df['Label'].astype(int)
        return df
    
    def get_numeric_features(self, df: pd.DataFrame, 
                            exclude_cols: list = None) -> pd.DataFrame:
        """Extract numeric features, excluding specified columns."""
        if exclude_cols is None:
            exclude_cols = self.config.CLASSIFICATION_EXCLUDE
        
        # Also exclude cancer-related columns to avoid leakage
        cancer_keywords = ['cancer', 'malignant', 'tumor']
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if any(kw in col.lower() for kw in cancer_keywords):
                continue
            if df[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
        
        return df[feature_cols]


# =============================================================================
# MODEL TRAINING & PERSISTENCE
# =============================================================================

class ModelTrainer:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.config.ensure_dirs()
        
        # Models to be trained
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.cancer_classifier = None
        self.phenotype_classifier = None
        self.umap_reducer = None
        self.label_encoder = LabelEncoder()
        
        # Fitted statistics
        self.normal_centroid = None
        self.cluster_mapping = None
        
    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the scaler."""
        return self.scaler.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)
    
    def train_clustering(self, tumor_df: pd.DataFrame, normal_df: pd.DataFrame,
                        n_clusters: int = 3) -> pd.DataFrame:
        """
        Train k-means clustering on tumor samples with forced taxonomy.
        Returns tumor_df with cluster assignments.
        """
        features = self.config.CLUSTER_FEATURES
        
        # Check for missing features
        missing = [f for f in features if f not in tumor_df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Prepare data
        X_tumor = tumor_df[features].values
        X_normal = normal_df[features].values
        
        # Fit scaler on combined data (for proper normalization)
        X_combined = np.vstack([X_tumor, X_normal])
        self.scaler.fit(X_combined)
        
        X_tumor_scaled = self.scaler.transform(X_tumor)
        X_normal_scaled = self.scaler.transform(X_normal)
        
        # Calculate normal centroid for dysregulation score
        self.normal_centroid = np.mean(X_normal_scaled, axis=0)
        
        # Train KMeans with k=3
        self.cluster_model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.config.RANDOM_STATE,
            n_init=20,
            max_iter=500
        )
        cluster_labels = self.cluster_model.fit_predict(X_tumor_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_tumor_scaled, cluster_labels)
        print(f"Silhouette Score (k={n_clusters}): {sil_score:.4f}")
        
        # Determine cluster naming based on biological characteristics
        self.cluster_mapping = self._determine_cluster_names(
            tumor_df, features, cluster_labels
        )
        
        # Assign results to dataframe
        tumor_df = tumor_df.copy()
        tumor_df['Cluster_ID'] = cluster_labels
        tumor_df['Clinical_Cluster'] = [self.cluster_mapping[l] for l in cluster_labels]
        
        # Calculate dysregulation score
        tumor_df['Dysregulation_Score'] = [
            euclidean(s, self.normal_centroid) for s in X_tumor_scaled
        ]
        
        # Fit UMAP for visualization
        self.umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=self.config.RANDOM_STATE,
            n_neighbors=50,  # Highly increased for global structure
            min_dist=0.5,    # Maximized for cluster separation
            metric='euclidean'
        )
        embedding = self.umap_reducer.fit_transform(X_tumor_scaled)
        tumor_df['UMAP_1'] = embedding[:, 0]
        tumor_df['UMAP_2'] = embedding[:, 1]
        
        return tumor_df
    
    def _determine_cluster_names(self, tumor_df: pd.DataFrame, 
                                 features: list, labels: np.ndarray) -> Dict[int, str]:
        """
        Determine cluster names based on biological signatures.
        - Inflamed: High CD8+, High Shannon (diverse TCR)
        - Suppressed: High M2, High Tregs
        - Desert: Low overall immune infiltration
        """
        cluster_stats = []
        
        for i in range(3):
            mask = labels == i
            if np.sum(mask) == 0:
                cluster_stats.append({'id': i, 'inflamed': -999, 'suppressed': -999, 'desert': 999})
                continue
            
            subset = tumor_df.iloc[mask]
            
            # Inflamed score: CD8+ T cells + Shannon diversity
            inflamed_score = (
                subset['Quantiseq_T_cell_CD8+'].mean() + 
                subset['shannon'].mean() / 5  # Normalize Shannon
            )
            
            # Suppressed score: M2 macrophages + Tregs
            suppressed_score = (
                subset['Epidish_BPRNACan_M2'].mean() + 
                subset['Quantiseq_T_cell_regulatory_(Tregs)'].mean()
            )
            
            # Desert score: Overall immune activity (inverse)
            desert_score = subset[features].mean().mean()
            
            cluster_stats.append({
                'id': i,
                'inflamed': inflamed_score,
                'suppressed': suppressed_score,
                'desert': desert_score,
                'n_samples': np.sum(mask)
            })
        
        # Assign names greedily
        mapping = {}
        assigned = set()
        
        # 1. Most suppressed → Immune-Suppressed
        sorted_supp = sorted(cluster_stats, key=lambda x: x['suppressed'], reverse=True)
        mapping[sorted_supp[0]['id']] = 'Immune-Suppressed'
        assigned.add(sorted_supp[0]['id'])
        
        # 2. Most inflamed (not assigned) → Immune-Inflamed
        sorted_inf = sorted(cluster_stats, key=lambda x: x['inflamed'], reverse=True)
        for c in sorted_inf:
            if c['id'] not in assigned:
                mapping[c['id']] = 'Immune-Inflamed'
                assigned.add(c['id'])
                break
        
        # 3. Remaining → Immune-Desert
        for i in range(3):
            if i not in assigned:
                mapping[i] = 'Immune-Desert'
        
        print(f"Cluster Mapping: {mapping}")
        for stat in cluster_stats:
            name = mapping[stat['id']]
            print(f"  {name}: n={stat.get('n_samples', 0)}, "
                  f"inflamed={stat['inflamed']:.3f}, "
                  f"suppressed={stat['suppressed']:.3f}")
        
        return mapping
    
    def train_cancer_classifier(self, X: pd.DataFrame, y: pd.Series,
                               n_folds: int = 5) -> Dict[str, Any]:
        """
        Train XGBoost classifier for cancer vs normal prediction.
        Returns performance metrics with confidence intervals.
        """
        # Filter to APP_FEATURES for consistency with Streamlit
        app_feats = [f for f in self.config.APP_FEATURES if f in X.columns]
        if len(app_feats) > 0:
            print(f"Training Cancer Classifier on {len(app_feats)} App-Compatible features")
            X = X[app_feats]
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, 
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        self.cancer_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.config.RANDOM_STATE
        )
        self.cancer_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.cancer_classifier.predict(X_test)
        y_pred_proba = self.cancer_classifier.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                            random_state=self.config.RANDOM_STATE)
        cv_scores = cross_val_score(self.cancer_classifier, X, y, cv=cv, scoring='accuracy')
        
        # Bootstrap confidence interval for accuracy
        accuracy_ci = self._bootstrap_accuracy_ci(y_test.values, y_pred)
        
        results = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'accuracy_95ci': accuracy_ci,
            'classification_report': classification_report(
                y_test, y_pred, target_names=['Normal', 'Tumor'], output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_names': X.columns.tolist()
        }
        
        print(f"\n{'='*50}")
        print("CANCER CLASSIFICATION RESULTS")
        print(f"{'='*50}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"95% CI: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
        print(f"Cross-Validation: {results['cv_mean']:.4f} ± {results['cv_std']*2:.4f}")
        print(f"{'='*50}\n")
        
        return results
    
    def train_phenotype_classifier(self, tumor_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train classifier to predict immune phenotype (cluster) from features.
        This enables phenotype prediction for new patients.
        """
        features = self.config.CLUSTER_FEATURES
        X = tumor_df[features]
        y = tumor_df['Clinical_Cluster']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, stratify=y_encoded,
            random_state=self.config.RANDOM_STATE
        )
        
        # Train
        self.phenotype_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=self.config.RANDOM_STATE
        )
        self.phenotype_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.phenotype_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.phenotype_classifier, X, y_encoded, cv=5, scoring='accuracy'
        )
        
        results = {
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classes': self.label_encoder.classes_.tolist()
        }
        
        print(f"Phenotype Classifier Accuracy: {accuracy:.4f}")
        print(f"CV: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
        
        return results
    
    def _bootstrap_accuracy_ci(self, y_true: np.ndarray, y_pred: np.ndarray,
                               n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for accuracy."""
        n = len(y_true)
        accuracies = []
        
        rng = np.random.RandomState(self.config.RANDOM_STATE)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            acc = accuracy_score(y_true[idx], y_pred[idx])
            accuracies.append(acc)
        
        lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
        upper = np.percentile(accuracies, (1 + ci) / 2 * 100)
        
        return (lower, upper)
    
    def save_models(self, prefix: str = "luad_pipeline", cancer_feature_names: list = None):
        """Save all trained models to disk."""
        models_dict = {
            'scaler': self.scaler,
            'cluster_model': self.cluster_model,
            'cancer_classifier': self.cancer_classifier,
            'phenotype_classifier': self.phenotype_classifier,
            'umap_reducer': self.umap_reducer,
            'label_encoder': self.label_encoder,
            'normal_centroid': self.normal_centroid,
            'cluster_mapping': self.cluster_mapping,
            'cluster_features': self.config.CLUSTER_FEATURES,
            'cancer_feature_names': cancer_feature_names or []
        }
        
        filepath = self.config.MODELS_DIR / f"{prefix}_models.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(models_dict, f)
        
        print(f"Models saved to: {filepath}")
        return filepath
    
    def load_models(self, prefix: str = "luad_pipeline") -> bool:
        """Load trained models from disk."""
        filepath = self.config.MODELS_DIR / f"{prefix}_models.pkl"
        
        if not filepath.exists():
            print(f"No saved models found at: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            models_dict = pickle.load(f)
        
        self.scaler = models_dict['scaler']
        self.cluster_model = models_dict['cluster_model']
        self.cancer_classifier = models_dict['cancer_classifier']
        self.phenotype_classifier = models_dict['phenotype_classifier']
        self.umap_reducer = models_dict['umap_reducer']
        self.label_encoder = models_dict['label_encoder']
        self.normal_centroid = models_dict['normal_centroid']
        self.cluster_mapping = models_dict['cluster_mapping']
        
        print(f"Models loaded from: {filepath}")
        return True


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

class StatisticalAnalyzer:
    """Handles statistical tests and analysis."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
    
    def cluster_anova(self, tumor_df: pd.DataFrame, 
                     metric: str = 'Dysregulation_Score') -> Dict[str, Any]:
        """Perform ANOVA across clusters for a given metric."""
        clusters = tumor_df['Clinical_Cluster'].unique()
        groups = [tumor_df[tumor_df['Clinical_Cluster'] == c][metric].values 
                  for c in clusters]
        
        f_stat, p_value = f_oneway(*groups)
        
        # Pairwise t-tests
        pairwise = {}
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i < j:
                    g1 = tumor_df[tumor_df['Clinical_Cluster'] == c1][metric]
                    g2 = tumor_df[tumor_df['Clinical_Cluster'] == c2][metric]
                    t_stat, p = ttest_ind(g1, g2)
                    pairwise[f"{c1} vs {c2}"] = {'t_stat': t_stat, 'p_value': p}
        
        return {
            'metric': metric,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'pairwise_tests': pairwise
        }
    
    def feature_importance_analysis(self, model, feature_names: list,
                                   X: pd.DataFrame) -> pd.DataFrame:
        """Get SHAP-based feature importance."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, take absolute mean
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Creates publication-quality visualizations."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib style for beautiful plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.dpi': 150,
            'figure.figsize': (12, 8),
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.titleweight': 'bold',
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def plot_immune_phenotype_clusters(self, tumor_df: pd.DataFrame,
                                       save_path: str = None,
                                       show_centroids: bool = True,
                                       show_silhouette: bool = True) -> plt.Figure:
        """
        Create a beautiful, publication-ready cluster visualization.
        
        Features:
        - UMAP scatter with cluster colors
        - Cluster centroids marked
        - Silhouette score annotation
        - Sample count per cluster
        - Clean, modern aesthetic
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Main scatter plot
        ax_main = fig.add_axes([0.1, 0.25, 0.55, 0.65])
        
        # Color palette
        palette = self.config.PALETTE
        
        # Plot each cluster
        for cluster_name in ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']:
            mask = tumor_df['Clinical_Cluster'] == cluster_name
            subset = tumor_df[mask]
            
            ax_main.scatter(
                subset['UMAP_1'], 
                subset['UMAP_2'],
                c=palette[cluster_name],
                s=120,
                alpha=0.75,
                edgecolors='white',
                linewidth=0.5,
                label=f"{cluster_name} (n={len(subset)})",
                zorder=2
            )
            
            # Plot centroid
            if show_centroids and len(subset) > 0:
                centroid_x = subset['UMAP_1'].mean()
                centroid_y = subset['UMAP_2'].mean()
                ax_main.scatter(
                    centroid_x, centroid_y,
                    c=palette[cluster_name],
                    s=400,
                    marker='*',
                    edgecolors='black',
                    linewidth=2,
                    zorder=3
                )
        
        # Styling
        ax_main.set_xlabel('UMAP Dimension 1', fontweight='bold')
        ax_main.set_ylabel('UMAP Dimension 2', fontweight='bold')
        ax_main.set_title(
            'LUAD Patient Stratification by Immune Phenotype\n'
            '(k=3 Forced Taxonomy Based on Tumor Microenvironment)',
            fontsize=14, fontweight='bold', pad=15
        )
        
        # Legend
        legend = ax_main.legend(
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            title='Immune Phenotype',
            title_fontsize=11
        )
        legend.get_title().set_fontweight('bold')
        
        # Add grid with low alpha
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.set_axisbelow(True)
        
        # ======= Right Panel: Cluster Characteristics =======
        ax_right = fig.add_axes([0.72, 0.35, 0.25, 0.45])
        
        # Calculate mean feature values per cluster
        features = ['Quantiseq_T_cell_CD8+', 'Epidish_BPRNACan_M2', 
                   'Quantiseq_T_cell_regulatory_(Tregs)', 'Quantiseq_B_cell']
        feature_labels = ['CD8+ T Cells', 'M2 Macrophages', 'Tregs', 'B Cells']
        
        cluster_order = ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']
        
        # Normalize for visualization
        data_matrix = []
        for cluster in cluster_order:
            mask = tumor_df['Clinical_Cluster'] == cluster
            means = []
            for feat in features:
                val = tumor_df.loc[mask, feat].mean()
                means.append(val)
            data_matrix.append(means)
        
        data_matrix = np.array(data_matrix)
        # Min-max normalize each feature
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            if col.max() > col.min():
                data_matrix[:, j] = (col - col.min()) / (col.max() - col.min())
        
        # Heatmap
        im = ax_right.imshow(data_matrix, aspect='auto', cmap='RdYlBu_r')
        
        ax_right.set_xticks(range(len(feature_labels)))
        ax_right.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=9)
        ax_right.set_yticks(range(len(cluster_order)))
        ax_right.set_yticklabels([c.replace('Immune-', '') for c in cluster_order], fontsize=10)
        ax_right.set_title('Immune Signature\n(Normalized)', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_right, shrink=0.8, pad=0.02)
        cbar.set_label('Relative Level', fontsize=9)
        
        # ======= Bottom Panel: Dysregulation Distribution =======
        ax_bottom = fig.add_axes([0.1, 0.05, 0.55, 0.15])
        
        for i, cluster in enumerate(cluster_order):
            mask = tumor_df['Clinical_Cluster'] == cluster
            data = tumor_df.loc[mask, 'Dysregulation_Score']
            
            violin_parts = ax_bottom.violinplot(
                [data], positions=[i], showmeans=True, showmedians=False
            )
            
            for pc in violin_parts['bodies']:
                pc.set_facecolor(palette[cluster])
                pc.set_alpha(0.7)
            
            for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
                if key in violin_parts:
                    violin_parts[key].set_color('black')
            
            # Add jittered points (strip plot equivalent)
            jitter_x = np.random.normal(i, 0.04, size=len(data))
            ax_bottom.scatter(jitter_x, data, s=10, color='black', alpha=0.3, zorder=3)
        
        ax_bottom.set_xticks(range(3))
        ax_bottom.set_xticklabels([c.replace('Immune-', '') for c in cluster_order])
        ax_bottom.set_ylabel('Dysregulation\nScore', fontsize=10)
        ax_bottom.set_title('Distance from Normal Immune State', fontsize=11, fontweight='bold')
        ax_bottom.grid(True, alpha=0.3, axis='y')
        
        # ======= Statistics Annotation =======
        ax_stats = fig.add_axes([0.72, 0.05, 0.25, 0.25])
        ax_stats.axis('off')
        
        # Calculate stats
        n_total = len(tumor_df)
        stats_text = "STATISTICAL SUMMARY\n" + "─" * 25 + "\n\n"
        
        for cluster in cluster_order:
            n = (tumor_df['Clinical_Cluster'] == cluster).sum()
            pct = n / n_total * 100
            mean_dys = tumor_df[tumor_df['Clinical_Cluster'] == cluster]['Dysregulation_Score'].mean()
            stats_text += f"• {cluster.replace('Immune-', '')}\n"
            stats_text += f"   n={n} ({pct:.1f}%), Dysreg={mean_dys:.2f}\n\n"
        
        ax_stats.text(
            0.05, 0.95, stats_text,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='gray', alpha=0.8)
        )
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             labels: list = ['Normal', 'Tumor'],
                             save_path: str = None) -> plt.Figure:
        """Create a beautiful confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 20, 'weight': 'bold'},
            ax=ax, cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Cancer Classification Performance', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               save_path: str = None) -> plt.Figure:
        """Plot top feature importances."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(
            range(len(top_features)), 
            top_features['importance'].values,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
        ax.set_title('Top Predictive Features (Cancer Classification)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    def plot_dysregulation_distribution(self, tumor_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Create a dedicated, detailed violin plot for dysregulation scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_order = ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']
        
        # Filter for known clusters
        plot_df = tumor_df[tumor_df['Clinical_Cluster'].isin(cluster_order)].copy()
        
        sns.violinplot(
            x='Clinical_Cluster', 
            y='Dysregulation_Score',
            data=plot_df,
            order=cluster_order,
            palette=self.config.PALETTE,
            alpha=0.5,
            inner=None,
            ax=ax
        )
        
        # Add jittered points 
        sns.stripplot(
            x='Clinical_Cluster',
            y='Dysregulation_Score',
            data=plot_df,
            order=cluster_order,
            color='black',
            alpha=0.4,
            size=3,
            jitter=0.2,
            ax=ax
        )
        
        ax.set_title('Dysregulation Score Distribution by Immune Phenotype', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dysregulation Score (Distance from Normal)', fontsize=12)
        ax.set_xlabel('Immune Phenotype', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add stats
        means = plot_df.groupby('Clinical_Cluster')['Dysregulation_Score'].mean()
        for i, cluster in enumerate(cluster_order):
            if cluster in means:
                 ax.text(i, means[cluster], f"Mean: {means[cluster]:.2f}", 
                        ha='center', va='bottom', fontweight='bold', color='black')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            
        return fig

    def plot_phenotype_radar_chart(self, tumor_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Create a radar chart comparing the mean profiles of each phenotype."""
        
        # Features to plot (normalized)
        features = self.config.CLUSTER_FEATURES
        # Clean labels
        feature_labels = [f.replace('Quantiseq_', '').replace('Epidish_', '').replace('Macrophage_', 'Mac_').replace('_cell', '') for f in features]
        
        # Calculate means per cluster
        means_df = tumor_df.groupby('Clinical_Cluster')[features].mean()
        
        # Normalize to 0-1 for radar chart
        normalized_means = (means_df - means_df.min()) / (means_df.max() - means_df.min())
        
        # Prepare plot
        categories = feature_labels
        N = len(categories)
        
        # Angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for cluster in ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']:
            if cluster not in normalized_means.index: continue
            
            values = normalized_means.loc[cluster].values.flatten().tolist()
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster, color=self.config.PALETTE[cluster])
            ax.fill(angles, values, alpha=0.1, color=self.config.PALETTE[cluster])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        
        ax.set_title("Comparative Immune Profiles (Normalized)", size=15, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            
        return fig


# =============================================================================
# SINGLE PATIENT INFERENCE
# =============================================================================

class PatientPredictor:
    """
    Handles single-patient inference through the full pipeline.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.trainer = ModelTrainer(config)
        self.data_processor = DataProcessor(config)
        
        # Load pre-trained models
        self.models_loaded = self.trainer.load_models()
        
        # Load MiTCR data for Shannon lookup
        if self.models_loaded:
            self.mitcr_df = self.data_processor.load_mitcr_data()
    
    def predict_from_deconvolution(self, deconv_row: pd.Series) -> Dict[str, Any]:
        """
        Predict cancer status and phenotype from a single deconvolution result.
        
        Args:
            deconv_row: Series containing deconvolution results + shannon
        
        Returns:
            Dictionary with predictions and confidence
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Train first.")
        
        # Get cluster features
        features = self.config.CLUSTER_FEATURES
        X = deconv_row[features].values.reshape(1, -1)
        
        # Scale
        X_scaled = self.trainer.scaler.transform(X)
        
        # Predict phenotype
        phenotype_id = self.trainer.phenotype_classifier.predict(X_scaled)[0]
        phenotype_proba = self.trainer.phenotype_classifier.predict_proba(X_scaled)[0]
        phenotype_name = self.trainer.label_encoder.inverse_transform([phenotype_id])[0]
        
        # Calculate dysregulation score
        dysreg_score = euclidean(X_scaled[0], self.trainer.normal_centroid)
        
        # Get UMAP position
        umap_pos = self.trainer.umap_reducer.transform(X_scaled)[0]
        
        # Get treatment recommendation
        treatment = self.config.TREATMENT_MAP.get(phenotype_name, {})
        
        return {
            'sample_id': deconv_row.get('sample', 'Unknown'),
            'immune_phenotype': phenotype_name,
            'phenotype_confidence': float(phenotype_proba.max()),
            'phenotype_probabilities': {
                self.trainer.label_encoder.inverse_transform([i])[0]: float(p) 
                for i, p in enumerate(phenotype_proba)
            },
            'dysregulation_score': float(dysreg_score),
            'umap_coordinates': {'x': float(umap_pos[0]), 'y': float(umap_pos[1])},
            'treatment_recommendation': treatment,
            'cell_proportions': {
                'CD8+ T Cells': float(deconv_row['Quantiseq_T_cell_CD8+']),
                'M2 Macrophages': float(deconv_row['Epidish_BPRNACan_M2']),
                'Tregs': float(deconv_row['Quantiseq_T_cell_regulatory_(Tregs)']),
                'TCR Diversity (Shannon)': float(deconv_row['shannon'])
            }
        }
    
    def run_deconvolution_for_patient(self, tsv_path: str) -> pd.DataFrame:
        """
        Run R deconvolution script for a single patient.
        
        Args:
            tsv_path: Path to patient's bulk RNA-seq TSV file
        
        Returns:
            DataFrame with deconvolution results
        """
        # Create a temporary script that processes just this file
        temp_script = self.config.BASE_DIR / "temp_single_patient_deconv.R"
        
        r_code = f'''
# Single Patient Deconvolution Script
local_lib <- file.path(getwd(), "R_libs")
.libPaths(c(local_lib, .libPaths()))

library(readr)
library(dplyr)
library(tibble)
library(tidyr)

# Load GEMDeCan functions
setwd("GEMDeCan_deconvolution")
source("scripts/deconvolution/deconvolution_algorithms.R")
library(immunedeconv)
library(EpiDISH)
library(DeconRNASeq)

# Read patient file
f <- "{tsv_path.replace(chr(92), '/')}"
patient_id <- tools::file_path_sans_ext(basename(f))

d <- suppressMessages(read_tsv(f, comment = "#", show_col_types = FALSE))
if ("gene_name" %in% colnames(d)) {{
    d <- d %>% select(Gene = gene_name, Value = tpm_unstranded)
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
write_csv(combined, "temp_patient_deconv_result.csv")
'''
        
        with open(temp_script, 'w') as f:
            f.write(r_code)
        
        # Run R script
        print(f"Running deconvolution for: {tsv_path}")
        result = subprocess.run(
            ['Rscript', str(temp_script)],
            capture_output=True,
            text=True,
            cwd=str(self.config.BASE_DIR)
        )
        
        if result.returncode != 0:
            print(f"R Error: {result.stderr}")
            raise RuntimeError(f"Deconvolution failed: {result.stderr}")
        
        # Load result
        result_path = self.config.BASE_DIR / "temp_patient_deconv_result.csv"
        deconv_df = pd.read_csv(result_path)
        
        # Cleanup
        temp_script.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        
        return deconv_df
    
    def predict_new_patient(self, tsv_path: str) -> Dict[str, Any]:
        """
        Full pipeline: TSV → Deconvolution → Feature Engineering → Prediction
        
        Args:
            tsv_path: Path to patient's bulk RNA-seq TSV file
        
        Returns:
            Complete prediction results
        """
        tsv_path = Path(tsv_path)
        patient_id = tsv_path.stem
        
        print(f"\n{'='*60}")
        print(f"PROCESSING NEW PATIENT: {patient_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Run deconvolution
        print("[1/4] Running immune deconvolution...")
        deconv_df = self.run_deconvolution_for_patient(str(tsv_path))
        
        # Step 2: Get Shannon index
        print("[2/4] Fetching TCR diversity metrics...")
        sample_id_16 = patient_id[:16]
        shannon_match = self.mitcr_df[self.mitcr_df['Sample_ID'] == sample_id_16]
        
        if len(shannon_match) > 0:
            shannon = shannon_match['shannon'].values[0]
            if pd.isna(shannon):
                shannon = 0.0
        else:
            print(f"  Warning: No TCR data found for {sample_id_16}, using default")
            shannon = 0.0
        
        deconv_df['shannon'] = shannon
        
        # Step 3: Feature engineering
        print("[3/4] Engineering immune features...")
        cd8_col = 'Quantiseq_T_cell_CD8+'
        m2_col = 'Epidish_BPRNACan_M2' if 'Epidish_BPRNACan_M2' in deconv_df.columns else None
        treg_col = 'Quantiseq_T_cell_regulatory_(Tregs)'
        
        if cd8_col in deconv_df.columns:
            deconv_df['Immune_Engagement_Index'] = deconv_df.apply(
                lambda r: r[cd8_col] / r['shannon'] if r['shannon'] > 0 else 0, axis=1
            )
        else:
            deconv_df['Immune_Engagement_Index'] = 0
        
        if m2_col and treg_col in deconv_df.columns:
            deconv_df['Macrophage_Blockade'] = deconv_df[m2_col] * deconv_df[treg_col]
        else:
            deconv_df['Macrophage_Blockade'] = 0
        
        # Step 4: Predict
        print("[4/4] Predicting immune phenotype...")
        result = self.predict_from_deconvolution(deconv_df.iloc[0])
        
        # Add raw deconvolution data
        result['raw_deconvolution'] = deconv_df.iloc[0].to_dict()
        
        print(f"\n{'='*60}")
        print("PREDICTION COMPLETE")
        print(f"{'='*60}")
        print(f"Immune Phenotype: {result['immune_phenotype']}")
        print(f"Confidence: {result['phenotype_confidence']*100:.1f}%")
        print(f"Dysregulation Score: {result['dysregulation_score']:.3f}")
        print(f"Treatment: {result['treatment_recommendation'].get('primary', 'N/A')}")
        print(f"{'='*60}\n")
        
        return result


# =============================================================================
# MAIN PIPELINE ORCHESTRATION
# =============================================================================

def train_full_pipeline(save_models: bool = True) -> Dict[str, Any]:
    """
    Train the complete pipeline from scratch.
    
    Returns:
        Dictionary with all training results and tumor DataFrame
    """
    config = PipelineConfig()
    config.ensure_dirs()
    
    print("\n" + "="*70)
    print("LUAD IMMUNE PHENOTYPE CLASSIFICATION PIPELINE - TRAINING")
    print("="*70 + "\n")
    
    # Initialize components
    processor = DataProcessor(config)
    trainer = ModelTrainer(config)
    stats = StatisticalAnalyzer(config)
    viz = Visualizer(config)
    
    # Load data
    print("[1/6] Loading merged immune features...")
    df = processor.load_merged_features()
    
    # Split tumor/normal
    print("[2/6] Splitting tumor and normal samples...")
    tumor_df, normal_df = processor.split_tumor_normal(df)
    
    # Train clustering
    print("[3/6] Training immune phenotype clustering (k=3)...")
    tumor_df = trainer.train_clustering(tumor_df, normal_df, n_clusters=3)
    
    # Train phenotype classifier
    print("[4/6] Training phenotype classifier...")
    phenotype_results = trainer.train_phenotype_classifier(tumor_df)
    
    # Train cancer classifier
    print("[5/6] Training cancer classifier...")
    df_labeled = processor.create_label(df)
    X = processor.get_numeric_features(df_labeled)
    y = df_labeled['Label']
    cancer_results = trainer.train_cancer_classifier(X, y)
    
    # Statistical analysis
    print("[6/6] Performing statistical analysis...")
    anova_results = stats.cluster_anova(tumor_df, 'Dysregulation_Score')
    print(f"\nANOVA (Dysregulation across clusters):")
    print(f"  F-statistic: {anova_results['f_statistic']:.3f}")
    print(f"  p-value: {anova_results['p_value']:.2e}")
    print(f"  Significant: {anova_results['significant']}")
    
    for comparison, result in anova_results['pairwise_tests'].items():
        print(f"  {comparison}: p={result['p_value']:.4f}")
    
    # Save models
    if save_models:
        # Include cancer classifier feature names for inference
        cancer_feature_names = cancer_results.get('feature_names', X.columns.tolist())
        trainer.save_models(cancer_feature_names=cancer_feature_names)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Main cluster plot
    cluster_plot_path = config.RESULTS_DIR / "plots" / "immune_phenotype_clusters.png"
    viz.plot_immune_phenotype_clusters(tumor_df, save_path=str(cluster_plot_path))
    
    # Confusion matrix
    cm = np.array(cancer_results['confusion_matrix'])
    cm_path = config.RESULTS_DIR / "plots" / "cancer_classification_cm.png"
    viz.plot_confusion_matrix(cm, save_path=str(cm_path))
    
    # Feature importance
    importance_df = stats.feature_importance_analysis(
        trainer.cancer_classifier, 
        cancer_results['feature_names'],
        X
    )
    fi_path = config.RESULTS_DIR / "plots" / "feature_importance.png"
    viz.plot_feature_importance(importance_df, save_path=str(fi_path))
    
    # Save results summary
    results_summary = {
        'cancer_classification': {
            'test_accuracy': cancer_results['test_accuracy'],
            'cv_mean': cancer_results['cv_mean'],
            'cv_std': cancer_results['cv_std'],
            'cv_scores': cancer_results['cv_scores'],
            'accuracy_95ci': list(cancer_results['accuracy_95ci']),
        },
        'phenotype_classification': phenotype_results,
        'cluster_anova': {
            'f_statistic': float(anova_results['f_statistic']),
            'p_value': float(anova_results['p_value']),
            'significant': bool(anova_results['significant'])
        },
        'cluster_distribution': {k: int(v) for k, v in tumor_df['Clinical_Cluster'].value_counts().to_dict().items()}
    }
    
    results_path = config.RESULTS_DIR / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return {
        'tumor_df': tumor_df,
        'normal_df': normal_df,
        'trainer': trainer,
        'results': results_summary
    }


def predict_patient(tsv_path: str) -> Dict[str, Any]:
    """
    Convenience function to predict immune phenotype for a new patient.
    
    Args:
        tsv_path: Path to patient's bulk RNA-seq TSV file
    
    Returns:
        Prediction results
    """
    predictor = PatientPredictor()
    return predictor.predict_new_patient(tsv_path)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LUAD Immune Phenotype Classification Pipeline'
    )
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict'],
        default='train',
        help='Pipeline mode: train or predict'
    )
    parser.add_argument(
        '--patient-file',
        type=str,
        help='Path to patient TSV file (for predict mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        results = train_full_pipeline()
        plt.show()
    
    elif args.mode == 'predict':
        if not args.patient_file:
            print("Error: --patient-file required for predict mode")
        else:
            result = predict_patient(args.patient_file)
            print(json.dumps(result, indent=2, default=str))

