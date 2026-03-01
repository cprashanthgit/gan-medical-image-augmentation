"""
Visualization Utilities.

Helper functions for visualizing images, training progress,
and dataset statistics.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_image_grid(
    images: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    title: str = "Image Grid",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Display images in a grid layout.
    
    Args:
        images: Array of images (N, H, W, C) in [0, 1] range
        rows: Number of rows
        cols: Number of columns
        title: Figure title
        figsize: Figure size (auto-calculated if None)
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = (cols * 2, rows * 2)
    
    fig = plt.figure(figsize=figsize)
    
    num_images = min(rows * cols, len(images))
    
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved image grid to: {save_path}")
    
    return fig


def plot_training_samples(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot side-by-side comparison of real and generated images.
    
    Args:
        real_images: Real training images
        generated_images: GAN-generated images
        num_samples: Number of samples to show
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    for i in range(num_samples):
        axes[0, i].imshow(real_images[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real Images', fontsize=12, loc='left')
        
        axes[1, i].imshow(generated_images[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated Images', fontsize=12, loc='left')
    
    plt.suptitle('Real vs Generated Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    return fig


def show_generated_images(
    output_dir: str,
    epochs: List[int],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Display generated images from different training epochs.
    
    Args:
        output_dir: Directory containing saved images
        epochs: List of epochs to display
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, len(epochs), figsize=figsize)
    
    if len(epochs) == 1:
        axes = [axes]
    
    for ax, epoch in zip(axes, epochs):
        img_path = output_path / f"epoch_{epoch:03d}.png"
        
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(f'Epoch {epoch}', fontsize=12)
        else:
            ax.text(0.5, 0.5, f'Epoch {epoch}\nNot Found',
                   ha='center', va='center', fontsize=10)
        
        ax.axis('off')
    
    plt.suptitle('Training Progress', fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_class_distribution(
    df: pd.DataFrame,
    label_column: str = 'dx',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize class distribution in the dataset.
    
    Args:
        df: DataFrame with labels
        label_column: Column name containing labels
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    class_counts = df[label_column].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    
    bars = axes[0].bar(class_counts.index, class_counts.values, color=colors)
    axes[0].set_xlabel('Class Label', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Class Distribution', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            str(count),
            ha='center', va='bottom', fontsize=9
        )
    
    axes[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    axes[1].set_title('Class Proportions', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution to: {save_path}")
    
    return fig


def plot_loss_curves(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}')
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.set_title(f'Training {metric.title()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to: {save_path}")
    
    return fig


def create_summary_figure(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    training_history: dict,
    metrics: dict,
    label_names: List[str],
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive summary figure with all results.
    
    Args:
        real_images: Sample real images
        generated_images: Sample generated images
        training_history: Training history dict
        metrics: Evaluation metrics dict
        label_names: Class names
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    for i in range(min(4, len(real_images))):
        ax1_sub = fig.add_subplot(gs[0, 0])
        for idx in range(4):
            plt.subplot(1, 4, idx + 1)
            plt.imshow(real_images[idx])
            plt.axis('off')
    ax1.set_title('Real Images', fontsize=12)
    ax1.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary figure to: {save_path}")
    
    return fig
