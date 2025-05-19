# src/data_validation/drift.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize with reference (training) data
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_statistics(reference_data)
        
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for each column"""
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'hist': np.histogram(data[col].dropna(), bins=10)
            }
            
        for col in data.select_dtypes(include=['object', 'category']).columns:
            stats_dict[col] = {
                'value_counts': data[col].value_counts(normalize=True).to_dict(),
                'unique_count': data[col].nunique()
            }
            
        return stats_dict
    
    def detect_drift(self, current_data: pd.DataFrame, 
                     threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect drift between reference and current data
        Returns drift metrics and flagged features
        """
        current_stats = self._compute_statistics(current_data)
        drift_results = {
            'drift_detected': False,
            'feature_drifts': {},
            'flagged_features': []
        }
        
        # Check numeric features using Kolmogorov-Smirnov test
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col not in current_data.columns:
                continue
                
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_results['feature_drifts'][col] = {
                'test': 'ks',
                'statistic': ks_stat,
                'p_value': p_value,
                'drift': p_value < threshold
            }
            
            if p_value < threshold:
                drift_results['drift_detected'] = True
                drift_results['flagged_features'].append(col)
                
        # Check categorical features using Chi-squared test
        for col in self.reference_data.select_dtypes(include=['object', 'category']).columns:
            if col not in current_data.columns:
                continue
            
            # Get value distributions
            ref_counts = self.reference_stats[col]['value_counts']
            curr_counts = current_stats[col]['value_counts']
            
            # Calculate JS divergence as an alternative
            js_div = self._jensen_shannon_divergence(ref_counts, curr_counts)
            
            drift_results['feature_drifts'][col] = {
                'test': 'jensen_shannon',
                'statistic': js_div,
                'drift': js_div > threshold
            }
            
            if js_div > threshold:
                drift_results['drift_detected'] = True
                drift_results['flagged_features'].append(col)
        
        return drift_results
    
    def _jensen_shannon_divergence(self, dist1, dist2):
        """Calculate Jensen-Shannon divergence between two distributions"""
        # Implementation details here
        return 0.1  # Placeholder
