"""
Classifier Training Module.

Provides utilities for training CNN classifiers with support for
data augmentation, class balancing, and evaluation.
"""

from typing import Optional, Dict, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from ..models.classifier import CNNClassifier, compute_class_weights
from ..config import ClassifierConfig


class ClassifierTrainer:
    """
    Trainer for CNN classifiers with augmentation support.
    
    Handles training, evaluation, and comparison between
    baseline and augmented models.
    """
    
    def __init__(
        self,
        config: ClassifierConfig,
        label_names: List[str],
        output_dir: str = "./outputs/classifier/"
    ):
        """
        Initialize classifier trainer.
        
        Args:
            config: ClassifierConfig with training parameters
            label_names: List of class label names
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.label_names = label_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_model = None
        self.augmented_model = None
        self.baseline_history = None
        self.augmented_history = None
    
    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        use_class_weights: bool = False
    ) -> CNNClassifier:
        """
        Train baseline classifier on original data.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot)
            X_val: Validation images
            y_val: Validation labels
            use_class_weights: Whether to use class weights
            
        Returns:
            Trained CNNClassifier
        """
        print("\n" + "="*60)
        print("Training Baseline Classifier")
        print("="*60)
        
        self.baseline_model = CNNClassifier(
            num_classes=self.config.num_classes,
            learning_rate=self.config.learning_rate,
            dropout_rate=self.config.dropout_rate
        )
        
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_train)
            print(f"Using class weights: {class_weights}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        self.baseline_history = self.baseline_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weights=class_weights,
            callbacks=callbacks
        )
        
        return self.baseline_model
    
    def train_augmented(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        synthetic_images: np.ndarray,
        synthetic_labels: np.ndarray,
        use_class_weights: bool = False
    ) -> CNNClassifier:
        """
        Train classifier on augmented data (original + synthetic).
        
        Args:
            X_train: Original training images
            y_train: Original training labels
            X_val: Validation images
            y_val: Validation labels
            synthetic_images: GAN-generated images
            synthetic_labels: Labels for synthetic images
            use_class_weights: Whether to use class weights
            
        Returns:
            Trained CNNClassifier
        """
        print("\n" + "="*60)
        print("Training Augmented Classifier")
        print("="*60)
        
        X_aug = np.concatenate([X_train, synthetic_images], axis=0)
        y_aug = np.concatenate([y_train, synthetic_labels], axis=0)
        
        print(f"Original data: {X_train.shape[0]} samples")
        print(f"Synthetic data: {synthetic_images.shape[0]} samples")
        print(f"Augmented data: {X_aug.shape[0]} samples")
        
        self.augmented_model = CNNClassifier(
            num_classes=self.config.num_classes,
            learning_rate=self.config.learning_rate,
            dropout_rate=self.config.dropout_rate
        )
        
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_aug)
            print(f"Using class weights: {class_weights}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        self.augmented_history = self.augmented_model.train(
            X_aug, y_aug,
            X_val, y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weights=class_weights,
            callbacks=callbacks
        )
        
        return self.augmented_model
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both baseline and augmented models.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics for both models
        """
        results = {}
        
        if self.baseline_model is not None:
            baseline_loss, baseline_acc = self.baseline_model.evaluate(X_test, y_test)
            results['baseline'] = {
                'loss': baseline_loss,
                'accuracy': baseline_acc
            }
            print(f"\nBaseline Model:")
            print(f"  Test Loss: {baseline_loss:.4f}")
            print(f"  Test Accuracy: {baseline_acc:.4f}")
        
        if self.augmented_model is not None:
            aug_loss, aug_acc = self.augmented_model.evaluate(X_test, y_test)
            results['augmented'] = {
                'loss': aug_loss,
                'accuracy': aug_acc
            }
            print(f"\nAugmented Model:")
            print(f"  Test Loss: {aug_loss:.4f}")
            print(f"  Test Accuracy: {aug_acc:.4f}")
        
        if 'baseline' in results and 'augmented' in results:
            improvement = results['augmented']['accuracy'] - results['baseline']['accuracy']
            print(f"\nAccuracy Improvement: {improvement*100:.2f}%")
        
        return results
    
    def plot_training_comparison(self, save_path: Optional[str] = None):
        """Plot training curves comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if self.baseline_history:
            axes[0, 0].plot(self.baseline_history.history['loss'], label='Train Loss')
            axes[0, 0].plot(self.baseline_history.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Baseline - Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(self.baseline_history.history['accuracy'], label='Train Acc')
            axes[0, 1].plot(self.baseline_history.history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_title('Baseline - Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        if self.augmented_history:
            axes[1, 0].plot(self.augmented_history.history['loss'], label='Train Loss')
            axes[1, 0].plot(self.augmented_history.history['val_loss'], label='Val Loss')
            axes[1, 0].set_title('Augmented - Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(self.augmented_history.history['accuracy'], label='Train Acc')
            axes[1, 1].plot(self.augmented_history.history['val_accuracy'], label='Val Acc')
            axes[1, 1].set_title('Augmented - Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training comparison to: {save_path}")
        
        plt.show()
    
    def save_models(self, prefix: str = "classifier"):
        """Save both models."""
        if self.baseline_model:
            path = self.output_dir / f"{prefix}_baseline.h5"
            self.baseline_model.save(str(path))
            print(f"Saved baseline model to: {path}")
        
        if self.augmented_model:
            path = self.output_dir / f"{prefix}_augmented.h5"
            self.augmented_model.save(str(path))
            print(f"Saved augmented model to: {path}")
