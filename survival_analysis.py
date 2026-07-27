"""
Survival Analysis Module for LUAD Immune Phenotype Study
=========================================================
Integrates clinical survival data with immune phenotype clustering
to validate clinical relevance of the discovered clusters.

Key Analyses:
1. Kaplan-Meier survival curves by immune phenotype
2. Log-rank tests for statistical significance
3. Cox Proportional Hazards regression
4. Forest plots for hazard ratios
5. Risk stratification validation

Author: Jayaditya (ISEF Project)
"""

import pandas as pd
import numpy as np
import json
import glob
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import warnings

# Survival Analysis
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Stats
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class SurvivalConfig:
    """Configuration for survival analysis."""
    
    BASE_DIR = Path(__file__).parent
    CLINICAL_DIR = BASE_DIR / "patient_clinical_data"
    RESULTS_DIR = BASE_DIR / "results"
    PLOTS_DIR = RESULTS_DIR / "plots"
    
    # Immune phenotype colors (matching pipeline_core.py)
    PALETTE = {
        'Immune-Inflamed': '#E63946',    # Vibrant Red
        'Immune-Suppressed': '#6A0572',   # Deep Purple
        'Immune-Desert': '#457B9D'        # Steel Blue
    }
    
    # Linestyles for different phenotypes
    LINESTYLES = {
        'Immune-Inflamed': '-',
        'Immune-Suppressed': '--',
        'Immune-Desert': ':'
    }
    
    @classmethod
    def ensure_dirs(cls):
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# CLINICAL DATA PARSING
# =============================================================================

class ClinicalDataParser:
    """
    Parses TCGA clinical JSON files and extracts survival-relevant data.
    """
    
    def __init__(self, config: SurvivalConfig = None):
        self.config = config or SurvivalConfig()
    
    def parse_single_patient(self, json_path: str) -> Dict[str, Any]:
        """
        Parse a single clinical JSON file.
        
        Extracts:
        - Survival time (days)
        - Event status (1=death, 0=censored)
        - Age at diagnosis
        - Cancer stage
        - Smoking history
        - Gender
        """
        # Handle UTF-8 BOM encoding
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Extract sample ID from filename
        sample_id = Path(json_path).stem.replace('_clinical', '')
        
        # Demographics
        demo = data.get('demographic', {})
        vital_status = demo.get('vital_status', 'Unknown')
        age = demo.get('age_at_index', None)
        gender = demo.get('gender', 'Unknown')
        days_to_death = demo.get('days_to_death', None)
        
        # Find primary diagnosis (diagnosis_is_primary_disease = true)
        diagnoses = data.get('diagnoses', [])
        primary_diag = None
        for diag in diagnoses:
            if diag.get('diagnosis_is_primary_disease', False):
                primary_diag = diag
                break
        
        # If no explicit primary, use first diagnosis
        if primary_diag is None and diagnoses:
            primary_diag = diagnoses[0]
        
        # Extract from primary diagnosis
        days_to_follow_up = None
        stage = None
        if primary_diag:
            days_to_follow_up = primary_diag.get('days_to_last_follow_up', None)
            stage = primary_diag.get('ajcc_pathologic_stage', None)
        
        # Exposures (smoking)
        exposures = data.get('exposures', [])
        pack_years = None
        for exp in exposures:
            if exp.get('exposure_type') == 'Tobacco':
                pack_years = exp.get('pack_years_smoked', None)
                break
        
        # Calculate survival time and event
        event = 0
        survival_time = None
        
        if vital_status == 'Dead':
            event = 1
            survival_time = days_to_death
        elif vital_status == 'Alive':
            event = 0
            survival_time = days_to_follow_up
        
        # Convert survival time to months for easier interpretation
        survival_months = survival_time / 30.44 if survival_time is not None else None
        
        return {
            'sample': sample_id,
            'vital_status': vital_status,
            'event': event,
            'survival_days': survival_time,
            'survival_months': survival_months,
            'age_at_diagnosis': age,
            'gender': gender,
            'stage': stage,
            'pack_years_smoked': pack_years
        }
    
    def parse_all_patients(self) -> pd.DataFrame:
        """Parse all clinical JSON files in the clinical directory."""
        json_files = glob.glob(str(self.config.CLINICAL_DIR / "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.config.CLINICAL_DIR}")
        
        records = []
        for jf in json_files:
            try:
                record = self.parse_single_patient(jf)
                records.append(record)
            except Exception as e:
                print(f"Warning: Failed to parse {jf}: {e}")
        
        df = pd.DataFrame(records)
        
        print(f"Parsed {len(df)} clinical records")
        print(f"  - Alive: {(df['event'] == 0).sum()}")
        print(f"  - Dead: {(df['event'] == 1).sum()}")
        print(f"  - Missing survival time: {df['survival_days'].isna().sum()}")
        
        return df


# =============================================================================
# SURVIVAL ANALYSIS
# =============================================================================

class SurvivalAnalyzer:
    """
    Performs comprehensive survival analysis stratified by immune phenotype.
    """
    
    def __init__(self, config: SurvivalConfig = None):
        self.config = config or SurvivalConfig()
        self.config.ensure_dirs()
        self.parser = ClinicalDataParser(config)
    
    def prepare_survival_data(self, tumor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge immune phenotype data with clinical survival data.
        
        Args:
            tumor_df: DataFrame with immune features and Clinical_Cluster column
        
        Returns:
            Merged DataFrame ready for survival analysis
        """
        # Parse clinical data
        clinical_df = self.parser.parse_all_patients()
        
        # Merge on sample ID
        merged = pd.merge(
            tumor_df,
            clinical_df,
            on='sample',
            how='inner'
        )
        
        # Filter to patients with valid survival data
        valid_mask = (
            merged['survival_days'].notna() & 
            (merged['survival_days'] > 0)
        )
        merged = merged[valid_mask].copy()
        
        print(f"\nMerged dataset: {len(merged)} patients with valid survival data")
        print(f"Cluster distribution:")
        for cluster in merged['Clinical_Cluster'].unique():
            n = (merged['Clinical_Cluster'] == cluster).sum()
            events = merged[merged['Clinical_Cluster'] == cluster]['event'].sum()
            print(f"  {cluster}: n={n}, events={events}")
        
        return merged
    
    def kaplan_meier_analysis(self, surv_df: pd.DataFrame,
                             time_col: str = 'survival_months',
                             event_col: str = 'event',
                             group_col: str = 'Clinical_Cluster') -> Dict[str, Any]:
        """
        Perform Kaplan-Meier survival analysis.
        
        Returns:
            Dictionary with KM estimators, median survival, and statistics
        """
        results = {
            'km_estimators': {},
            'median_survival': {},
            'survival_at_timepoints': {},
            'n_per_group': {},
            'events_per_group': {}
        }
        
        groups = surv_df[group_col].unique()
        
        for group in groups:
            mask = surv_df[group_col] == group
            subset = surv_df[mask]
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                subset[time_col],
                subset[event_col],
                label=group
            )
            
            results['km_estimators'][group] = kmf
            results['median_survival'][group] = kmf.median_survival_time_
            results['n_per_group'][group] = len(subset)
            results['events_per_group'][group] = subset[event_col].sum()
            
            # Survival probability at specific timepoints (12, 24, 36 months)
            timepoints = [12, 24, 36, 60]
            results['survival_at_timepoints'][group] = {}
            for t in timepoints:
                try:
                    # Get survival probability at timepoint
                    idx = kmf.survival_function_.index.get_indexer([t], method='nearest')[0]
                    surv_prob = kmf.survival_function_.iloc[idx].values[0]
                    
                    # Get confidence interval
                    ci_low = kmf.confidence_interval_survival_function_.iloc[idx, 0]
                    ci_high = kmf.confidence_interval_survival_function_.iloc[idx, 1]
                    
                    results['survival_at_timepoints'][group][f'{t}m'] = {
                        'probability': surv_prob,
                        'ci_low': ci_low,
                        'ci_high': ci_high
                    }
                except:
                    pass
        
        return results
    
    def log_rank_tests(self, surv_df: pd.DataFrame,
                       time_col: str = 'survival_months',
                       event_col: str = 'event',
                       group_col: str = 'Clinical_Cluster') -> Dict[str, Any]:
        """
        Perform log-rank tests for survival differences.
        
        Returns:
            Overall multivariate log-rank test and pairwise comparisons
        """
        groups = surv_df[group_col].unique()
        
        # Overall multivariate log-rank test
        multi_result = multivariate_logrank_test(
            surv_df[time_col],
            surv_df[group_col],
            surv_df[event_col]
        )
        
        results = {
            'overall': {
                'test_statistic': multi_result.test_statistic,
                'p_value': multi_result.p_value,
                'significant': multi_result.p_value < 0.05
            },
            'pairwise': {}
        }
        
        # Pairwise log-rank tests
        groups_list = list(groups)
        for i, g1 in enumerate(groups_list):
            for g2 in groups_list[i+1:]:
                mask1 = surv_df[group_col] == g1
                mask2 = surv_df[group_col] == g2
                
                result = logrank_test(
                    surv_df.loc[mask1, time_col],
                    surv_df.loc[mask2, time_col],
                    surv_df.loc[mask1, event_col],
                    surv_df.loc[mask2, event_col]
                )
                
                comparison = f"{g1} vs {g2}"
                results['pairwise'][comparison] = {
                    'test_statistic': result.test_statistic,
                    'p_value': result.p_value,
                    'significant': result.p_value < 0.05
                }
        
        return results
    
    def cox_regression(self, surv_df: pd.DataFrame,
                       time_col: str = 'survival_months',
                       event_col: str = 'event',
                       group_col: str = 'Clinical_Cluster',
                       covariates: List[str] = None) -> Dict[str, Any]:
        """
        Perform Cox Proportional Hazards regression.
        
        Args:
            covariates: Additional covariates to include (e.g., 'age_at_diagnosis', 'stage')
        
        Returns:
            Cox model results with hazard ratios and confidence intervals
        """
        # Prepare data - create dummy variables for cluster
        df_cox = surv_df[[time_col, event_col, group_col]].copy()
        
        # Use Immune-Inflamed as reference (best prognosis typically)
        df_cox = pd.get_dummies(df_cox, columns=[group_col], drop_first=False)
        
        # Rename columns for clarity
        cluster_cols = [c for c in df_cox.columns if 'Clinical_Cluster' in c]
        rename_map = {c: c.replace('Clinical_Cluster_', '') for c in cluster_cols}
        df_cox = df_cox.rename(columns=rename_map)
        
        # Add covariates if specified
        if covariates:
            for cov in covariates:
                if cov in surv_df.columns:
                    df_cox[cov] = surv_df[cov].values
        
        # Drop rows with missing values
        df_cox = df_cox.dropna()
        
        # Drop reference category (Immune-Inflamed)
        if 'Immune-Inflamed' in df_cox.columns:
            df_cox = df_cox.drop(columns=['Immune-Inflamed'])
        
        # Fit Cox model
        cph = CoxPHFitter()
        try:
            cph.fit(df_cox, duration_col=time_col, event_col=event_col)
        except Exception as e:
            print(f"Cox regression warning: {e}")
            return None
        
        # Extract results
        summary = cph.summary
        
        results = {
            'model': cph,
            'summary_df': summary,
            'hazard_ratios': {},
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'aic': cph.AIC_partial_
        }
        
        for var in summary.index:
            hr = summary.loc[var, 'exp(coef)']
            hr_lower = summary.loc[var, 'exp(coef) lower 95%']
            hr_upper = summary.loc[var, 'exp(coef) upper 95%']
            p_val = summary.loc[var, 'p']
            
            results['hazard_ratios'][var] = {
                'HR': hr,
                'HR_lower_95': hr_lower,
                'HR_upper_95': hr_upper,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        
        return results
    
    def multivariate_cox(self, surv_df: pd.DataFrame,
                         time_col: str = 'survival_months',
                         event_col: str = 'event') -> Dict[str, Any]:
        """
        Fit multivariate Cox model with immune phenotype, age, and dysregulation.
        """
        # Prepare variables
        df_mv = surv_df[[time_col, event_col, 'Clinical_Cluster', 
                         'Dysregulation_Score']].copy()
        
        # Add age if available
        if 'age_at_diagnosis' in surv_df.columns:
            df_mv['Age'] = surv_df['age_at_diagnosis'].values
        
        # Create dummy variables for cluster (Inflamed as reference)
        df_mv = pd.get_dummies(df_mv, columns=['Clinical_Cluster'], drop_first=False)
        
        # Rename and drop reference
        rename_map = {c: c.replace('Clinical_Cluster_', '') for c in df_mv.columns 
                      if 'Clinical_Cluster' in c}
        df_mv = df_mv.rename(columns=rename_map)
        
        if 'Immune-Inflamed' in df_mv.columns:
            df_mv = df_mv.drop(columns=['Immune-Inflamed'])
        
        df_mv = df_mv.dropna()
        
        if len(df_mv) < 10:
            print("Warning: Too few samples for multivariate analysis")
            return None
        
        cph = CoxPHFitter()
        cph.fit(df_mv, duration_col=time_col, event_col=event_col)
        
        return {
            'model': cph,
            'summary_df': cph.summary,
            'concordance_index': cph.concordance_index_,
            'aic': cph.AIC_partial_
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class SurvivalVisualizer:
    """Creates publication-quality survival analysis visualizations."""
    
    def __init__(self, config: SurvivalConfig = None):
        self.config = config or SurvivalConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.dpi': 150,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
        })
    
    def plot_kaplan_meier(self, km_results: Dict, logrank_results: Dict,
                         save_path: str = None,
                         title: str = "Overall Survival by Immune Phenotype",
                         show_ci: bool = True,
                         show_censor: bool = True,
                         show_at_risk: bool = True) -> plt.Figure:
        """
        Create a publication-quality Kaplan-Meier survival plot.
        
        Features:
        - Survival curves with confidence intervals
        - Log-rank p-value annotation
        - At-risk table
        - Median survival lines
        - Professional styling
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Main KM plot
        if show_at_risk:
            ax_main = fig.add_axes([0.12, 0.30, 0.80, 0.60])
        else:
            ax_main = fig.add_axes([0.12, 0.12, 0.80, 0.78])
        
        # Plot order (for consistent appearance)
        cluster_order = ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']
        
        # Track survival functions for at-risk table
        km_estimators = km_results['km_estimators']
        
        for cluster in cluster_order:
            if cluster not in km_estimators:
                continue
            
            kmf = km_estimators[cluster]
            color = self.config.PALETTE[cluster]
            ls = self.config.LINESTYLES[cluster]
            
            n = km_results['n_per_group'][cluster]
            events = km_results['events_per_group'][cluster]
            median = km_results['median_survival'].get(cluster, np.nan)
            
            # Create label with key info
            median_str = f"{median:.1f}" if not np.isnan(median) else "NR"
            label = f"{cluster}\n(n={n}, events={events}, median={median_str}mo)"
            
            # Plot survival curve
            ax_main.step(
                kmf.survival_function_.index,
                kmf.survival_function_.values.flatten(),
                where='post',
                color=color,
                linewidth=2.5,
                linestyle=ls,
                label=label
            )
            
            # Confidence interval
            if show_ci:
                ci = kmf.confidence_interval_survival_function_
                ax_main.fill_between(
                    ci.index,
                    ci.iloc[:, 0],
                    ci.iloc[:, 1],
                    alpha=0.2,
                    color=color,
                    step='post'
                )
            
            # Censoring marks
            if show_censor:
                censored_times = kmf.event_table[kmf.event_table['censored'] > 0].index
                for t in censored_times:
                    # Find survival probability at censoring time
                    idx = kmf.survival_function_.index.get_indexer([t], method='pad')[0]
                    if idx >= 0:
                        surv = kmf.survival_function_.iloc[idx].values[0]
                        ax_main.plot(t, surv, '|', color=color, markersize=8, 
                                    markeredgewidth=1.5)
        
        # Styling
        ax_main.set_xlim(0, None)
        ax_main.set_ylim(0, 1.05)
        ax_main.set_xlabel('Time (Months)', fontweight='bold', fontsize=12)
        ax_main.set_ylabel('Survival Probability', fontweight='bold', fontsize=12)
        ax_main.set_title(title, fontweight='bold', fontsize=14, pad=15)
        
        # Add horizontal lines at key survival probabilities
        for y in [0.5, 0.75]:
            ax_main.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Legend
        legend = ax_main.legend(
            loc='lower left',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fontsize=9
        )
        
        # Add log-rank p-value
        p_value = logrank_results['overall']['p_value']
        p_text = f"Log-rank p = {p_value:.4f}" if p_value >= 0.0001 else f"Log-rank p < 0.0001"
        
        significance = ""
        if p_value < 0.001:
            significance = " ***"
        elif p_value < 0.01:
            significance = " **"
        elif p_value < 0.05:
            significance = " *"
        
        ax_main.text(
            0.98, 0.98, 
            p_text + significance,
            transform=ax_main.transAxes,
            fontsize=11,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
        )
        
        # At-risk table
        if show_at_risk:
            ax_risk = fig.add_axes([0.12, 0.08, 0.80, 0.18])
            ax_risk.axis('off')
            
            # Define timepoints for at-risk table
            max_time = max(kmf.survival_function_.index.max() 
                          for kmf in km_estimators.values())
            timepoints = [0, 12, 24, 36, 48, 60]
            timepoints = [t for t in timepoints if t <= max_time]
            
            # Create at-risk table
            table_data = []
            for cluster in cluster_order:
                if cluster not in km_estimators:
                    continue
                kmf = km_estimators[cluster]
                row = [cluster.replace('Immune-', '')]
                for t in timepoints:
                    # Number at risk at time t
                    at_risk = (kmf.event_table.index <= t).sum()
                    remaining = kmf.event_table.loc[kmf.event_table.index <= t, 
                                                    'at_risk'].iloc[-1] if at_risk > 0 else 0
                    row.append(str(int(remaining)))
                table_data.append(row)
            
            # Create table
            col_labels = [''] + [f'{t}mo' for t in timepoints]
            table = ax_risk.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
                colWidths=[0.15] + [0.12] * len(timepoints)
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor('#E6E6E6')
                table[(0, i)].set_text_props(fontweight='bold')
            
            for row_idx, cluster in enumerate(cluster_order):
                if cluster in km_estimators:
                    color = self.config.PALETTE[cluster]
                    table[(row_idx + 1, 0)].set_facecolor(color)
                    table[(row_idx + 1, 0)].set_text_props(color='white', fontweight='bold')
            
            ax_risk.set_title('Number at Risk', fontsize=10, fontweight='bold', 
                             loc='left', pad=-5)
        
        ax_main.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_forest(self, cox_results: Dict,
                   save_path: str = None,
                   title: str = "Hazard Ratios for Mortality Risk") -> plt.Figure:
        """
        Create a forest plot showing hazard ratios with confidence intervals.
        """
        if cox_results is None:
            print("No Cox results to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        hr_data = cox_results['hazard_ratios']
        variables = list(hr_data.keys())
        n_vars = len(variables)
        
        y_positions = list(range(n_vars))
        
        for i, var in enumerate(variables):
            data = hr_data[var]
            hr = data['HR']
            hr_lower = data['HR_lower_95']
            hr_upper = data['HR_upper_95']
            p_val = data['p_value']
            
            # Color based on significance
            color = '#E63946' if data['significant'] else '#457B9D'
            
            # Plot point estimate
            ax.plot(hr, i, 'o', color=color, markersize=12, zorder=3)
            
            # Plot CI line
            ax.hlines(y=i, xmin=hr_lower, xmax=hr_upper, color=color, 
                     linewidth=2.5, zorder=2)
            
            # Add caps
            ax.plot([hr_lower, hr_lower], [i-0.1, i+0.1], color=color, linewidth=2)
            ax.plot([hr_upper, hr_upper], [i-0.1, i+0.1], color=color, linewidth=2)
            
            # Add HR value and p-value text
            text = f"HR={hr:.2f} ({hr_lower:.2f}-{hr_upper:.2f})"
            p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            
            ax.text(ax.get_xlim()[1] * 0.95, i, f"{text}\n{p_text}", 
                   va='center', ha='left', fontsize=9)
        
        # Reference line at HR=1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, 
                  label='No effect (HR=1)', zorder=1)
        
        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels([v.replace('Immune-', '').replace('_', ' ') 
                           for v in variables], fontsize=11)
        ax.set_xlabel('Hazard Ratio (95% CI)', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
        
        # Set x-axis to log scale for better visualization
        ax.set_xscale('log')
        ax.set_xlim(0.1, 20)
        
        # Add interpretive labels
        ax.text(0.15, -0.7, '← Lower Risk', fontsize=10, ha='center', 
               transform=ax.get_xaxis_transform(), color='green')
        ax.text(6, -0.7, 'Higher Risk →', fontsize=10, ha='center',
               transform=ax.get_xaxis_transform(), color='red')
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946',
                   markersize=10, label='Significant (p<0.05)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#457B9D',
                   markersize=10, label='Not Significant'),
            Line2D([0], [0], color='black', linestyle='--', label='No Effect (HR=1)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # Add concordance index annotation
        c_idx = cox_results['concordance_index']
        ax.text(0.02, 0.02, f"Concordance Index: {c_idx:.3f}",
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_survival_summary(self, surv_df: pd.DataFrame,
                              km_results: Dict,
                              logrank_results: Dict,
                              cox_results: Dict,
                              save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive survival analysis summary figure.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # ====== Panel A: Kaplan-Meier Curves ======
        ax_km = fig.add_axes([0.05, 0.52, 0.55, 0.42])
        
        cluster_order = ['Immune-Inflamed', 'Immune-Suppressed', 'Immune-Desert']
        km_estimators = km_results['km_estimators']
        
        for cluster in cluster_order:
            if cluster not in km_estimators:
                continue
            
            kmf = km_estimators[cluster]
            color = self.config.PALETTE[cluster]
            
            n = km_results['n_per_group'][cluster]
            events = km_results['events_per_group'][cluster]
            
            ax_km.step(
                kmf.survival_function_.index,
                kmf.survival_function_.values.flatten(),
                where='post',
                color=color,
                linewidth=2.5,
                label=f"{cluster} (n={n}, events={events})"
            )
            
            ci = kmf.confidence_interval_survival_function_
            ax_km.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                              alpha=0.2, color=color, step='post')
        
        ax_km.set_xlim(0, None)
        ax_km.set_ylim(0, 1.05)
        ax_km.set_xlabel('Time (Months)', fontweight='bold')
        ax_km.set_ylabel('Survival Probability', fontweight='bold')
        ax_km.set_title('A. Kaplan-Meier Survival Curves', fontweight='bold', 
                       fontsize=13, loc='left')
        ax_km.legend(loc='lower left', fontsize=9)
        ax_km.grid(True, alpha=0.3)
        
        p_val = logrank_results['overall']['p_value']
        p_text = f"Log-rank p = {p_val:.4f}" if p_val >= 0.0001 else "Log-rank p < 0.0001"
        ax_km.text(0.98, 0.98, p_text, transform=ax_km.transAxes,
                  fontweight='bold', va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # ====== Panel B: Survival Statistics Table ======
        ax_table = fig.add_axes([0.65, 0.55, 0.32, 0.38])
        ax_table.axis('off')
        
        # Create statistics table
        table_data = [
            ['Phenotype', 'N', 'Events', 'Median OS\n(months)', '1-yr Surv\n(95% CI)']
        ]
        
        for cluster in cluster_order:
            if cluster not in km_estimators:
                continue
            
            n = km_results['n_per_group'][cluster]
            events = km_results['events_per_group'][cluster]
            median = km_results['median_survival'].get(cluster, np.nan)
            median_str = f"{median:.1f}" if not np.isnan(median) else "NR"
            
            # 12-month survival
            surv_12m = km_results['survival_at_timepoints'].get(cluster, {}).get('12m', {})
            if surv_12m:
                surv_str = f"{surv_12m['probability']*100:.0f}%\n({surv_12m['ci_low']*100:.0f}-{surv_12m['ci_high']*100:.0f}%)"
            else:
                surv_str = "N/A"
            
            table_data.append([
                cluster.replace('Immune-', ''),
                str(n),
                str(events),
                median_str,
                surv_str
            ])
        
        table = ax_table.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.12, 0.15, 0.22, 0.26]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 2.0)
        
        # Style header
        for j in range(5):
            table[(0, j)].set_facecolor('#2C3E50')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Color phenotype cells
        for i, cluster in enumerate(cluster_order):
            if cluster in km_estimators:
                table[(i+1, 0)].set_facecolor(self.config.PALETTE[cluster])
                table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
        
        ax_table.set_title('B. Survival Statistics by Immune Phenotype', 
                          fontweight='bold', fontsize=13, loc='left', pad=10)
        
        # ====== Panel C: Dysregulation vs Survival ======
        ax_scatter = fig.add_axes([0.05, 0.08, 0.40, 0.38])
        
        for cluster in cluster_order:
            mask = surv_df['Clinical_Cluster'] == cluster
            subset = surv_df[mask]
            
            ax_scatter.scatter(
                subset['Dysregulation_Score'],
                subset['survival_months'],
                c=self.config.PALETTE[cluster],
                s=80,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5,
                label=cluster
            )
            
            # Mark events (deaths) with X
            events_mask = subset['event'] == 1
            ax_scatter.scatter(
                subset.loc[events_mask, 'Dysregulation_Score'],
                subset.loc[events_mask, 'survival_months'],
                marker='x',
                c='black',
                s=30,
                linewidths=1.5
            )
        
        ax_scatter.set_xlabel('Dysregulation Score', fontweight='bold')
        ax_scatter.set_ylabel('Survival Time (Months)', fontweight='bold')
        ax_scatter.set_title('C. Dysregulation Score vs Survival', 
                            fontweight='bold', fontsize=13, loc='left')
        ax_scatter.legend(loc='upper right', fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
        
        # Add correlation annotation
        r, p = stats.spearmanr(surv_df['Dysregulation_Score'], surv_df['survival_months'])
        ax_scatter.text(0.02, 0.02, f"Spearman ρ = {r:.3f}\np = {p:.4f}",
                       transform=ax_scatter.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ====== Panel D: Cox Regression Summary ======
        ax_cox = fig.add_axes([0.52, 0.08, 0.45, 0.38])
        
        if cox_results:
            hr_data = cox_results['hazard_ratios']
            variables = list(hr_data.keys())
            n_vars = len(variables)
            
            for i, var in enumerate(variables):
                data = hr_data[var]
                hr = data['HR']
                hr_lower = data['HR_lower_95']
                hr_upper = data['HR_upper_95']
                
                color = '#E63946' if data['significant'] else '#457B9D'
                
                ax_cox.plot(hr, i, 'o', color=color, markersize=12, zorder=3)
                ax_cox.hlines(y=i, xmin=hr_lower, xmax=hr_upper, 
                             color=color, linewidth=2.5, zorder=2)
                ax_cox.plot([hr_lower, hr_lower], [i-0.1, i+0.1], color=color, linewidth=2)
                ax_cox.plot([hr_upper, hr_upper], [i-0.1, i+0.1], color=color, linewidth=2)
            
            ax_cox.axvline(x=1, color='black', linestyle='--', linewidth=1.5, zorder=1)
            ax_cox.set_yticks(range(n_vars))
            ax_cox.set_yticklabels([v.replace('Immune-', '').replace('_', ' ') 
                                   for v in variables])
            ax_cox.set_xscale('log')
            ax_cox.set_xlim(0.1, 20)
            ax_cox.set_xlabel('Hazard Ratio (95% CI)', fontweight='bold')
            
            c_idx = cox_results['concordance_index']
            ax_cox.text(0.02, 0.98, f"C-index: {c_idx:.3f}",
                       transform=ax_cox.transAxes, fontsize=10, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax_cox.text(0.5, 0.5, "Insufficient data for\nCox regression",
                       ha='center', va='center', fontsize=12, transform=ax_cox.transAxes)
            ax_cox.axis('off')
        
        ax_cox.set_title('D. Cox Proportional Hazards Model', 
                        fontweight='bold', fontsize=13, loc='left')
        ax_cox.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
        
        return fig


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_survival_analysis(tumor_df: pd.DataFrame = None,
                          save_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete survival analysis pipeline.
    
    Args:
        tumor_df: DataFrame with immune features and Clinical_Cluster column.
                  If None, will load from trained pipeline.
    
    Returns:
        Dictionary with all analysis results
    """
    config = SurvivalConfig()
    config.ensure_dirs()
    
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS - LUAD IMMUNE PHENOTYPE STUDY")
    print("="*70 + "\n")
    
    # Load tumor data if not provided
    if tumor_df is None:
        from pipeline_core import train_full_pipeline
        print("Loading pipeline data...")
        pipeline_result = train_full_pipeline(save_models=False)
        tumor_df = pipeline_result['tumor_df']
    
    # Initialize analyzer
    analyzer = SurvivalAnalyzer(config)
    viz = SurvivalVisualizer(config)
    
    # Prepare survival data
    print("[1/5] Preparing survival data...")
    surv_df = analyzer.prepare_survival_data(tumor_df)
    
    if len(surv_df) < 10:
        print("Warning: Too few patients with survival data for robust analysis")
        return {'error': 'Insufficient survival data'}
    
    # Kaplan-Meier analysis
    print("\n[2/5] Running Kaplan-Meier analysis...")
    km_results = analyzer.kaplan_meier_analysis(surv_df)
    
    print("\nMedian Survival by Phenotype:")
    for cluster, median in km_results['median_survival'].items():
        if np.isnan(median):
            print(f"  {cluster}: Not Reached")
        else:
            print(f"  {cluster}: {median:.1f} months")
    
    # Log-rank tests
    print("\n[3/5] Performing log-rank tests...")
    logrank_results = analyzer.log_rank_tests(surv_df)
    
    print(f"\nOverall Log-Rank Test:")
    print(f"  Chi-square: {logrank_results['overall']['test_statistic']:.3f}")
    print(f"  p-value: {logrank_results['overall']['p_value']:.4f}")
    print(f"  Significant: {logrank_results['overall']['significant']}")
    
    print("\nPairwise Comparisons:")
    for comparison, result in logrank_results['pairwise'].items():
        sig = "*" if result['significant'] else ""
        print(f"  {comparison}: p = {result['p_value']:.4f} {sig}")
    
    # Cox regression
    print("\n[4/5] Fitting Cox Proportional Hazards model...")
    cox_results = analyzer.cox_regression(surv_df)
    
    if cox_results:
        print("\nHazard Ratios (Reference: Immune-Inflamed):")
        for var, data in cox_results['hazard_ratios'].items():
            sig = "*" if data['significant'] else ""
            print(f"  {var}: HR = {data['HR']:.2f} "
                  f"({data['HR_lower_95']:.2f}-{data['HR_upper_95']:.2f}), "
                  f"p = {data['p_value']:.4f} {sig}")
        print(f"\nConcordance Index: {cox_results['concordance_index']:.3f}")
    
    # Multivariate Cox with dysregulation score
    print("\n[5/5] Fitting multivariate Cox model...")
    mv_cox_results = analyzer.multivariate_cox(surv_df)
    
    if mv_cox_results:
        print("\nMultivariate Model Summary:")
        print(mv_cox_results['summary_df'])
    
    # Generate visualizations
    if save_plots:
        print("\nGenerating survival plots...")
        
        # Main KM plot
        km_path = config.PLOTS_DIR / "survival_kaplan_meier.png"
        viz.plot_kaplan_meier(km_results, logrank_results, save_path=str(km_path))
        
        # Forest plot
        if cox_results:
            forest_path = config.PLOTS_DIR / "survival_forest_plot.png"
            viz.plot_forest(cox_results, save_path=str(forest_path))
        
        # Comprehensive summary
        summary_path = config.PLOTS_DIR / "survival_analysis_summary.png"
        viz.plot_survival_summary(surv_df, km_results, logrank_results, 
                                 cox_results, save_path=str(summary_path))
    
    # Compile all results
    results = {
        'survival_data': surv_df,
        'km_results': km_results,
        'logrank_results': logrank_results,
        'cox_results': cox_results,
        'multivariate_cox': mv_cox_results,
        'summary': {
            'n_patients': len(surv_df),
            'n_events': surv_df['event'].sum(),
            'median_followup_months': surv_df['survival_months'].median(),
            'logrank_p': logrank_results['overall']['p_value'],
            'significant_difference': logrank_results['overall']['significant']
        }
    }
    
    # Save results summary (convert numpy types to native Python)
    summary_file = config.RESULTS_DIR / "survival_analysis_results.json"
    
    # Helper to convert numpy types
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj
    
    summary_data = {
        'summary': convert_to_native(results['summary']),
        'logrank_overall': {
            'chi_square': float(logrank_results['overall']['test_statistic']),
            'p_value': float(logrank_results['overall']['p_value'])
        },
        'median_survival': {k: float(v) if not np.isnan(v) and not np.isinf(v) else None 
                           for k, v in km_results['median_survival'].items()},
        'hazard_ratios': convert_to_native(cox_results['hazard_ratios']) if cox_results else None
    }
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nResults saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    results = run_survival_analysis()
    plt.show()

