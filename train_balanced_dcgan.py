#!/usr/bin/env python3
"""
Balanced DCGAN Training Pipeline for Skin Lesion Data Augmentation.

This script identifies minority classes in the HAM10000 dataset,
trains a separate DCGAN for each minority class to perfectly balance 
the dataset distribution, and trains a CNN classifier to evaluate the performance.

Usage:
    python train_balanced_dcgan.py --data_dir ./data --gan_epochs 100
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
from src.models.dcgan import DCGAN
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
        description="Train DCGAN per minority class for PERFECT class balancing."
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
        help="Number of GAN training epochs per minority class"
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
        "--target_count",
        type=int,
        default=0,
        help="Target number of images per class. 0 means match the majority class count automatically."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main balanced training pipeline."""
    args = parse_args()
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    config = get_default_config()
    config.data.image_dir = os.path.join(args.data_dir, "HAM10000_images")
    config.data.metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    config.gan.epochs = args.gan_epochs
    config.gan.batch_size = args.batch_size
    config.classifier.epochs = args.classifier_epochs
    config.gan.output_dir = os.path.join(args.output_dir, "balanced_dcgan")
    
    print("\n" + "="*70)
    print(" BALANCED DCGAN Data Augmentation Pipeline")
    print(" Strategy: Train isolated GANs to upsample ALL minority classes.")
    print(" Dataset: HAM10000 Skin Lesion Images")
    print("="*70)
    
    print("\n[1/5] Loading metadata and analyzing distribution...")
    
    df = load_metadata(config.data.metadata_path, config.data.image_dir)
    class_counts = df['dx'].value_counts()
    
    print(f"Loaded metadata: {len(df)} samples")
    print("Original Distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")
        
    visualize_class_distribution(
        df, 
        save_path=os.path.join(args.output_dir, "original_class_distribution.png")
    )
    
    # Identify target count for perfect balancing
    majority_class = class_counts.idxmax()
    majority_count = class_counts.max()
    
    target_count = args.target_count if args.target_count > 0 else majority_count
    print(f"\nTarget image count per class for balancing: {target_count}")
    
    print("\n[2/5] Preparing initial classifier data structures...")
    
    (X_train, X_val, X_test), (y_train, y_val, y_test), label_names = prepare_classifier_data(
        df,
        img_size=config.data.img_size,
        num_classes=config.classifier.num_classes
    )
    
    print(f"Base Training set: {X_train.shape[0]} samples")
    print(f"Base Validation set: {X_val.shape[0]} samples")
    print(f"Base Test set: {X_test.shape[0]} samples")

    # Arrays to collect all synthetically generated images across all minority classes
    all_synthetic_images = []
    all_synthetic_labels = []
    from tensorflow.keras.utils import to_categorical
    
    print("\n[3/5] Iterating through classes to Train GANs & Synthesize Images...")
    for label in label_names:
        count = class_counts[label]
        deficit = target_count - count
        
        if deficit <= 0:
            print(f"\n>>> Skipping '{label}' (Count: {count}). No deficit to fill.")
            continue
            
        print(f"\n{'='*50}")
        print(f" Processing Class: {label} (Deficit: {deficit} images)")
        print(f"{'='*50}")
        
        class_df = df[df['dx'] == label]
        class_images = load_all_images(
            class_df,
            img_size=config.data.img_size,
            normalize_for='gan',
            verbose=False
        )
        
        dataset = create_tf_dataset(
            class_images,
            batch_size=config.gan.batch_size,
            buffer_size=config.gan.buffer_size
        )
        
        dcgan = DCGAN(
            latent_dim=config.gan.latent_dim,
            img_size=config.data.img_size,
            channels=config.data.channels,
            learning_rate=config.gan.learning_rate,
            beta_1=config.gan.beta_1
        )
        
        print(f"--> Training Specialist DCGAN for '{label}' for {config.gan.epochs} epochs...")
        trainer = GANTrainer(dcgan, config.gan, model_name=f"dcgan_{label}")
        
        # Suppress verbose output for individual epochs to keep console clean
        history = trainer.train(
            dataset,
            epochs=config.gan.epochs,
            save_interval=config.gan.save_interval,
            verbose=False 
        )
        
        # Save training history graph for this specific GAN
        trainer.plot_training_history(
            save_path=os.path.join(args.output_dir, f"dcgan_training_history_{label}.png")
        )
        
        print(f"--> Generating {deficit} synthetic '{label}' images...")
        synthetic_images = trainer.generate_synthetic_images(
            deficit,
            seed=args.seed
        )
        
        # Collect generated images
        all_synthetic_images.append(synthetic_images)
        
        # Create matching labels
        label_idx = label_names.index(label)
        synth_labels = to_categorical(
            np.full(deficit, label_idx),
            config.classifier.num_classes
        )
        all_synthetic_labels.append(synth_labels)

        plot_image_grid(
            synthetic_images[:16],
            rows=4, cols=4,
            title=f"Generated {label.upper()} Skin Lesions",
            save_path=os.path.join(args.output_dir, f"synthetic_samples_{label}.png")
        )
        
    print("\n[4/5] Merging perfectly balanced dataset...")
    if len(all_synthetic_images) > 0:
        final_synthetic_images = np.concatenate(all_synthetic_images, axis=0)
        final_synthetic_labels = np.concatenate(all_synthetic_labels, axis=0)
        print(f"Successfully synthesized {final_synthetic_images.shape[0]} total images across all minority classes.")
    else:
        print("No synthetic images were generated (Dataset already balanced).")
        final_synthetic_images = np.array([])
        final_synthetic_labels = np.array([])

    
    print("\n[5/5] Training and evaluating CNN Classifiers...")
    
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
        final_synthetic_images, final_synthetic_labels
    )
    
    results = classifier_trainer.evaluate_models(X_test, y_test)
    
    classifier_trainer.plot_training_comparison(
        save_path=os.path.join(args.output_dir, "balanced_classifier_comparison.png")
    )
    
    print("\n" + "="*70)
    print(" EVALUATION RESULTS")
    print("="*70)
    
    y_pred_probs_base = baseline_model.predict(X_test)
    y_pred_baseline = np.argmax(y_pred_probs_base, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n--- Baseline Model ---")
    print(generate_classification_report(y_true, y_pred_baseline, label_names))
    
    plot_confusion_matrix(
        y_true, y_pred_baseline, label_names,
        title="Baseline CNN Confusion Matrix",
        save_path=os.path.join(args.output_dir, "confusion_matrix_baseline.png")
    )
    
    y_pred_probs_aug = augmented_model.predict(X_test)
    y_pred_aug = np.argmax(y_pred_probs_aug, axis=1)
    
    print("\n--- Augmented Model (Perfectly Balanced) ---")
    print(generate_classification_report(y_true, y_pred_aug, label_names))
    
    plot_confusion_matrix(
        y_true, y_pred_aug, label_names,
        title="Balanced CNN Confusion Matrix",
        cmap="Greens",
        save_path=os.path.join(args.output_dir, "confusion_matrix_balanced_augmented.png")
    )
    
    classifier_trainer.save_models("balanced_dcgan")
    
    print("\n" + "="*70)
    print(" BALANCED PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Baseline Accuracy: {results['baseline']['accuracy']:.4f}")
    print(f"Balanced Augmented Accuracy: {results['augmented']['accuracy']:.4f}")
    
    improvement = results['augmented']['accuracy'] - results['baseline']['accuracy']
    print(f"Improvement: {improvement*100:+.2f}%")


if __name__ == "__main__":
    main()
