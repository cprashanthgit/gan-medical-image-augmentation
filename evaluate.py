#!/usr/bin/env python3
"""
Evaluation Script for Trained Models.

Load pre-trained models and evaluate them on test data.

Usage:
    python evaluate.py --model_path ./outputs/classifier/dcgan_augmented.h5 --data_dir ./data
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import get_default_config, CLASS_LABELS
from src.data.data_loader import load_metadata, prepare_classifier_data
from src.models.classifier import CNNClassifier
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    generate_classification_report,
    print_metrics_summary,
    compare_models
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model (.h5)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing HAM10000 data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evaluation",
        help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        help="Optional path to second model for comparison"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = get_default_config()
    config.data.image_dir = os.path.join(args.data_dir, "HAM10000_images")
    config.data.metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    
    print("\n" + "="*70)
    print(" Model Evaluation")
    print("="*70)
    
    print("\n[1/3] Loading data...")
    
    df = load_metadata(config.data.metadata_path, config.data.image_dir)
    
    (X_train, X_val, X_test), (y_train, y_val, y_test), label_names = prepare_classifier_data(
        df,
        img_size=config.data.img_size,
        num_classes=config.classifier.num_classes
    )
    
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {label_names}")
    
    print("\n[2/3] Loading model...")
    
    model = keras.models.load_model(args.model_path)
    print(f"Loaded model from: {args.model_path}")
    
    print("\n[3/3] Evaluating...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    metrics = compute_classification_metrics(y_true, y_pred, label_names)
    print_metrics_summary(metrics, title="Evaluation Metrics")
    
    print("\nClassification Report:")
    print(generate_classification_report(y_true, y_pred, label_names))
    
    model_name = os.path.basename(args.model_path).replace('.h5', '')
    
    plot_confusion_matrix(
        y_true, y_pred, label_names,
        title=f"Confusion Matrix - {model_name}",
        save_path=os.path.join(args.output_dir, f"confusion_matrix_{model_name}.png")
    )
    
    plot_roc_curves(
        y_test, y_pred_probs, label_names,
        save_path=os.path.join(args.output_dir, f"roc_curves_{model_name}.png")
    )
    
    if args.compare_with:
        print("\n" + "="*70)
        print(" Model Comparison")
        print("="*70)
        
        model2 = keras.models.load_model(args.compare_with)
        print(f"Loaded comparison model from: {args.compare_with}")
        
        y_pred2 = np.argmax(model2.predict(X_test), axis=1)
        metrics2 = compute_classification_metrics(y_true, y_pred2, label_names)
        
        compare_models(
            metrics, metrics2,
            save_path=os.path.join(args.output_dir, "model_comparison.png")
        )
    
    print("\n" + "="*70)
    print(" Evaluation Complete")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
