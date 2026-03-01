"""
Evaluation Metrics Module.

Comprehensive evaluation utilities for classification models including:
- Confusion matrix visualization
- Classification reports
- ROC curves
- Precision, Recall, F1 scores
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (integer indices)
        y_pred: Predicted labels (integer indices)
        label_names: Optional list of class names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'micro_precision': precision_score(y_true, y_pred, average='micro'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
        'micro_recall': recall_score(y_true, y_pred, average='micro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_dict: bool = False
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Class names
        output_dict: Return as dictionary instead of string
        
    Returns:
        Classification report string or dictionary
    """
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=output_dict,
        digits=4
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Class names
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        save_path: Optional path to save figure
        normalize: Normalize by true label counts
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    label_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_probs: Predicted probabilities
        label_names: Class names
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_classes = len(label_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, (label, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{label} (AUC = {roc_auc:.3f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multi-class Classification', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to: {save_path}")
    
    return fig


def compare_models(
    baseline_metrics: Dict[str, float],
    augmented_metrics: Dict[str, float],
    metric_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison bar chart between baseline and augmented models.
    
    Args:
        baseline_metrics: Metrics from baseline model
        augmented_metrics: Metrics from augmented model
        metric_names: Metrics to compare (uses common keys if None)
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        metric_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    baseline_values = [baseline_metrics.get(m, 0) for m in metric_names]
    augmented_values = [augmented_metrics.get(m, 0) for m in metric_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, augmented_values, width, label='Augmented', color='seagreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison chart to: {save_path}")
    
    return fig


def print_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Classification Metrics"
):
    """Print formatted metrics summary."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    print(f"\n  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"\n  Precision:")
    print(f"    - Micro:          {metrics['micro_precision']:.4f}")
    print(f"    - Macro:          {metrics['macro_precision']:.4f}")
    print(f"    - Weighted:       {metrics['weighted_precision']:.4f}")
    print(f"\n  Recall:")
    print(f"    - Micro:          {metrics['micro_recall']:.4f}")
    print(f"    - Macro:          {metrics['macro_recall']:.4f}")
    print(f"    - Weighted:       {metrics['weighted_recall']:.4f}")
    print(f"\n  F1 Score:")
    print(f"    - Micro:          {metrics['micro_f1']:.4f}")
    print(f"    - Macro:          {metrics['macro_f1']:.4f}")
    print(f"    - Weighted:       {metrics['weighted_f1']:.4f}")
    print(f"\n{'='*50}\n")
