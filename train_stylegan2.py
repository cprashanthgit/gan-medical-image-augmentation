#!/usr/bin/env python3
"""
StyleGAN2 Training Pipeline for Skin Lesion Data Augmentation.

This script trains a StyleGAN2-inspired model on the HAM10000 dataset,
generates synthetic images, and trains a CNN classifier to compare
baseline vs augmented performance.

Usage:
    python train_stylegan2.py --data_dir ./data --epochs 100

Author: Prash
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import Config, get_default_config, CLASS_LABELS
from src.data.data_loader import (
    load_metadata,
    load_all_images,
    create_tf_dataset,
    prepare_classifier_data
)
from src.models.stylegan2 import StyleGAN2
from src.models.classifier import CNNClassifier
from src.training.train_gan import GANTrainer
from src.training.train_classifier import ClassifierTrainer
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    generate_classification_report,
    print_metrics_summary
)
from src.utils.visualization import (
    plot_image_grid,
    visualize_class_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train StyleGAN2 for skin lesion data augmentation"
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
        default="./outputs",
        help="Directory for outputs"
    )
    parser.add_argument(
        "--gan_epochs",
        type=int,
        default=100,
        help="Number of GAN training epochs"
    )
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=50,
        help="Number of classifier training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=2000,
        help="Number of synthetic images to generate"
    )
    parser.add_argument(
        "--dlatent_dim",
        type=int,
        default=256,
        help="Dimension of disentangled latent space"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    config = get_default_config()
    config.data.image_dir = os.path.join(args.data_dir, "HAM10000_images")
    config.data.metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    config.gan.epochs = args.gan_epochs
    config.gan.batch_size = args.batch_size
    config.gan.output_dir = os.path.join(args.output_dir, "stylegan2")
    config.stylegan2.dlatent_dim = args.dlatent_dim
    config.classifier.epochs = args.classifier_epochs
    
    print("\n" + "="*70)
    print(" StyleGAN2 Data Augmentation Pipeline")
    print(" Dataset: HAM10000 Skin Lesion Images")
    print("="*70)
    
    print("\n[1/6] Loading and preprocessing data...")
    
    df = load_metadata(config.data.metadata_path, config.data.image_dir)
    print(f"Loaded metadata: {len(df)} samples")
    print(f"Classes: {df['dx'].unique().tolist()}")
    
    all_images = load_all_images(
        df,
        img_size=config.data.img_size,
        normalize_for='gan'
    )
    
    dataset = create_tf_dataset(
        all_images,
        batch_size=config.gan.batch_size,
        buffer_size=config.gan.buffer_size
    )
    
    print("\n[2/6] Initializing StyleGAN2 model...")
    
    stylegan = StyleGAN2(
        latent_dim=config.gan.latent_dim,
        img_size=config.data.img_size,
        channels=config.data.channels,
        dlatent_dim=config.stylegan2.dlatent_dim,
        base_filters=config.stylegan2.base_filters,
        learning_rate=config.gan.learning_rate,
        beta_1=config.gan.beta_1
    )
    
    stylegan.generator.summary()
    
    print("\n[3/6] Training StyleGAN2...")
    
    trainer = GANTrainer(stylegan, config.gan, model_name="stylegan2")
    history = trainer.train(
        dataset,
        epochs=config.gan.epochs,
        save_interval=config.gan.save_interval
    )
    
    trainer.plot_training_history(
        save_path=os.path.join(args.output_dir, "stylegan2_training_history.png")
    )
    
    print("\n[4/6] Generating synthetic images...")
    
    synthetic_images = trainer.generate_synthetic_images(
        args.num_synthetic,
        seed=args.seed
    )
    print(f"Generated {synthetic_images.shape[0]} synthetic images")
    
    plot_image_grid(
        synthetic_images[:16],
        rows=4, cols=4,
        title="StyleGAN2 Generated Skin Lesion Images",
        save_path=os.path.join(args.output_dir, "stylegan2_synthetic_samples.png")
    )
    
    print("\n[5/6] Preparing classifier data...")
    
    (X_train, X_val, X_test), (y_train, y_val, y_test), label_names = prepare_classifier_data(
        df,
        img_size=config.data.img_size,
        num_classes=config.classifier.num_classes
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    majority_class = df['dx'].value_counts().idxmax()
    majority_idx = label_names.index(majority_class)
    
    from tensorflow.keras.utils import to_categorical
    synthetic_labels = to_categorical(
        np.full(args.num_synthetic, majority_idx),
        config.classifier.num_classes
    )
    
    print("\n[6/6] Training and evaluating classifiers...")
    
    classifier_trainer = ClassifierTrainer(
        config.classifier,
        label_names,
        output_dir=os.path.join(args.output_dir, "classifier")
    )
    
    baseline_model = classifier_trainer.train_baseline(
        X_train, y_train,
        X_val, y_val
    )
    
    augmented_model = classifier_trainer.train_augmented(
        X_train, y_train,
        X_val, y_val,
        synthetic_images, synthetic_labels
    )
    
    results = classifier_trainer.evaluate_models(X_test, y_test)
    
    classifier_trainer.plot_training_comparison(
        save_path=os.path.join(args.output_dir, "stylegan2_classifier_comparison.png")
    )
    
    print("\n" + "="*70)
    print(" EVALUATION RESULTS")
    print("="*70)
    
    y_pred_baseline = baseline_model.predict_classes(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n--- Baseline Model ---")
    print(generate_classification_report(y_true, y_pred_baseline, label_names))
    
    plot_confusion_matrix(
        y_true, y_pred_baseline, label_names,
        title="Baseline CNN Confusion Matrix",
        save_path=os.path.join(args.output_dir, "stylegan2_confusion_baseline.png")
    )
    
    y_pred_aug = augmented_model.predict_classes(X_test)
    
    print("\n--- Augmented Model (StyleGAN2) ---")
    print(generate_classification_report(y_true, y_pred_aug, label_names))
    
    plot_confusion_matrix(
        y_true, y_pred_aug, label_names,
        title="StyleGAN2 Augmented CNN Confusion Matrix",
        cmap="Greens",
        save_path=os.path.join(args.output_dir, "stylegan2_confusion_augmented.png")
    )
    
    classifier_trainer.save_models("stylegan2")
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Baseline Accuracy: {results['baseline']['accuracy']:.4f}")
    print(f"Augmented Accuracy: {results['augmented']['accuracy']:.4f}")
    
    improvement = results['augmented']['accuracy'] - results['baseline']['accuracy']
    print(f"Improvement: {improvement*100:+.2f}%")


if __name__ == "__main__":
    main()
