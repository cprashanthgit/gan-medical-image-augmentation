#!/usr/bin/env python3
"""
Test Script for GAN Data Augmentation Pipeline
==============================================

This script validates all Python modules without requiring:
- The actual HAM10000 dataset
- GPU hardware
- Long training times

It tests:
1. All imports work correctly
2. Model architectures can be instantiated
3. Forward/backward passes work with dummy data
4. Training steps execute without errors

Usage:
    python test_code.py
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_test(name, passed, error=None):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if error:
        print(f"         Error: {error}")

def run_tests():
    """Run all validation tests."""
    results = {"passed": 0, "failed": 0, "errors": []}
    
    # =========================================================
    # TEST 1: Basic Imports
    # =========================================================
    print_header("TEST 1: Basic Python Imports")
    
    basic_imports = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("tensorflow", "import tensorflow as tf"),
        ("sklearn", "from sklearn.model_selection import train_test_split"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
    ]
    
    for name, import_stmt in basic_imports:
        try:
            exec(import_stmt)
            print_test(name, True)
            results["passed"] += 1
        except Exception as e:
            print_test(name, False, str(e))
            results["failed"] += 1
            results["errors"].append(f"Import {name}: {e}")
    
    # =========================================================
    # TEST 2: Source Module Imports
    # =========================================================
    print_header("TEST 2: Source Module Imports")
    
    module_imports = [
        ("src.config", "from src.config import Config, get_default_config, CLASS_LABELS"),
        ("src.data.data_loader", "from src.data.data_loader import HAM10000Dataset, preprocess_image"),
        ("src.models.dcgan", "from src.models.dcgan import DCGAN, DCGANGenerator, DCGANDiscriminator"),
        ("src.models.stylegan2", "from src.models.stylegan2 import StyleGAN2, StyleGAN2Generator"),
        ("src.models.classifier", "from src.models.classifier import CNNClassifier"),
        ("src.training.train_gan", "from src.training.train_gan import GANTrainer"),
        ("src.training.train_classifier", "from src.training.train_classifier import ClassifierTrainer"),
        ("src.evaluation.metrics", "from src.evaluation.metrics import compute_classification_metrics"),
        ("src.utils.visualization", "from src.utils.visualization import plot_image_grid"),
    ]
    
    for name, import_stmt in module_imports:
        try:
            exec(import_stmt)
            print_test(name, True)
            results["passed"] += 1
        except Exception as e:
            print_test(name, False, str(e))
            results["failed"] += 1
            results["errors"].append(f"Import {name}: {e}")
    
    # =========================================================
    # TEST 3: Configuration
    # =========================================================
    print_header("TEST 3: Configuration System")
    
    try:
        from src.config import Config, get_default_config, DataConfig, GANConfig
        
        config = get_default_config()
        print_test("get_default_config()", True)
        results["passed"] += 1
        
        assert config.data.img_size == 64, "img_size should be 64"
        assert config.gan.latent_dim == 100, "latent_dim should be 100"
        assert config.classifier.num_classes == 7, "num_classes should be 7"
        print_test("Config values correct", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("Configuration", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"Config: {e}")
    
    # =========================================================
    # TEST 4: DCGAN Model Architecture
    # =========================================================
    print_header("TEST 4: DCGAN Model Architecture")
    
    try:
        import tensorflow as tf
        import numpy as np
        from src.models.dcgan import DCGAN, DCGANGenerator, DCGANDiscriminator
        
        # Test Generator
        gen = DCGANGenerator(latent_dim=100, img_size=64, channels=3)
        dummy_noise = tf.random.normal([4, 100])
        gen_output = gen(dummy_noise)
        assert gen_output.shape == (4, 64, 64, 3), f"Generator output shape wrong: {gen_output.shape}"
        print_test("DCGANGenerator forward pass", True)
        results["passed"] += 1
        
        # Test Discriminator
        disc = DCGANDiscriminator(img_size=64, channels=3)
        dummy_images = tf.random.normal([4, 64, 64, 3])
        disc_output = disc(dummy_images)
        assert disc_output.shape == (4, 1), f"Discriminator output shape wrong: {disc_output.shape}"
        print_test("DCGANDiscriminator forward pass", True)
        results["passed"] += 1
        
        # Test full DCGAN
        dcgan = DCGAN(latent_dim=100, img_size=64, channels=3)
        print_test("DCGAN initialization", True)
        results["passed"] += 1
        
        # Test train step
        dummy_batch = tf.random.uniform([4, 64, 64, 3], -1, 1)
        g_loss, d_loss = dcgan.train_step(dummy_batch)
        assert g_loss.numpy() > 0, "Generator loss should be positive"
        assert d_loss.numpy() > 0, "Discriminator loss should be positive"
        print_test("DCGAN train_step execution", True)
        results["passed"] += 1
        
        # Test image generation
        generated = dcgan.generate_images(4, seed=42)
        assert generated.shape == (4, 64, 64, 3), f"Generated shape wrong: {generated.shape}"
        assert tf.reduce_min(generated) >= 0 and tf.reduce_max(generated) <= 1
        print_test("DCGAN image generation", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("DCGAN", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"DCGAN: {e}\n{traceback.format_exc()}")
    
    # =========================================================
    # TEST 5: StyleGAN2 Model Architecture
    # =========================================================
    print_header("TEST 5: StyleGAN2 Model Architecture")
    
    try:
        from src.models.stylegan2 import StyleGAN2, StyleGAN2Generator
        
        # Test Generator
        gen = StyleGAN2Generator(latent_dim=100, img_size=64, channels=3, dlatent_dim=256)
        dummy_noise = tf.random.normal([4, 100])
        gen_output = gen(dummy_noise)
        assert gen_output.shape == (4, 64, 64, 3), f"StyleGAN2 Generator output shape wrong: {gen_output.shape}"
        print_test("StyleGAN2Generator forward pass", True)
        results["passed"] += 1
        
        # Test full StyleGAN2
        stylegan = StyleGAN2(latent_dim=100, img_size=64, channels=3)
        print_test("StyleGAN2 initialization", True)
        results["passed"] += 1
        
        # Test train step
        dummy_batch = tf.random.uniform([4, 64, 64, 3], -1, 1)
        g_loss, d_loss = stylegan.train_step(dummy_batch)
        print_test("StyleGAN2 train_step execution", True)
        results["passed"] += 1
        
        # Test image generation
        generated = stylegan.generate_images(4, seed=42)
        assert generated.shape == (4, 64, 64, 3)
        print_test("StyleGAN2 image generation", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("StyleGAN2", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"StyleGAN2: {e}\n{traceback.format_exc()}")
    
    # =========================================================
    # TEST 6: CNN Classifier
    # =========================================================
    print_header("TEST 6: CNN Classifier")
    
    try:
        from src.models.classifier import CNNClassifier, compute_class_weights
        
        # Test classifier creation
        classifier = CNNClassifier(input_shape=(64, 64, 3), num_classes=7)
        print_test("CNNClassifier initialization", True)
        results["passed"] += 1
        
        # Test forward pass
        dummy_images = np.random.rand(4, 64, 64, 3).astype('float32')
        predictions = classifier.model.predict(dummy_images, verbose=0)
        assert predictions.shape == (4, 7), f"Classifier output shape wrong: {predictions.shape}"
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5), "Softmax outputs should sum to 1"
        print_test("CNNClassifier forward pass", True)
        results["passed"] += 1
        
        # Test training (1 epoch with dummy data, verbose=0 for Windows compatibility)
        dummy_labels = np.eye(7)[np.random.randint(0, 7, 4)]
        classifier.model.fit(
            dummy_images, dummy_labels,
            epochs=1, batch_size=2,
            verbose=0  # Suppress output to avoid Windows encoding issues
        )
        print_test("CNNClassifier training", True)
        results["passed"] += 1
        
        # Test evaluation
        loss, acc = classifier.model.evaluate(dummy_images, dummy_labels, verbose=0)
        print_test("CNNClassifier evaluation", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("CNN Classifier", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"Classifier: {e}\n{traceback.format_exc()}")
    
    # =========================================================
    # TEST 7: Evaluation Metrics
    # =========================================================
    print_header("TEST 7: Evaluation Metrics")
    
    try:
        from src.evaluation.metrics import (
            compute_classification_metrics,
            generate_classification_report,
            plot_confusion_matrix
        )
        
        # Dummy predictions
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1])
        label_names = ['class_0', 'class_1', 'class_2']
        
        metrics = compute_classification_metrics(y_true, y_pred, label_names)
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        print_test("compute_classification_metrics", True)
        results["passed"] += 1
        
        report = generate_classification_report(y_true, y_pred, label_names)
        assert isinstance(report, str)
        print_test("generate_classification_report", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("Evaluation Metrics", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"Metrics: {e}")
    
    # =========================================================
    # TEST 8: Data Loader Functions
    # =========================================================
    print_header("TEST 8: Data Loader Functions")
    
    try:
        from src.data.data_loader import create_tf_dataset, augment_with_synthetic
        
        # Test dataset creation
        dummy_images = np.random.rand(100, 64, 64, 3).astype('float32')
        dataset = create_tf_dataset(dummy_images, batch_size=16)
        
        # Verify batching works
        for batch in dataset.take(1):
            assert batch.shape[0] <= 16
            assert batch.shape[1:] == (64, 64, 3)
        print_test("create_tf_dataset", True)
        results["passed"] += 1
        
        # Test augmentation function
        X_train = np.random.rand(50, 64, 64, 3).astype('float32')
        y_train = np.eye(7)[np.random.randint(0, 7, 50)]
        synthetic = np.random.rand(20, 64, 64, 3).astype('float32')
        synthetic_labels = np.eye(7)[np.random.randint(0, 7, 20)]
        
        X_aug, y_aug = augment_with_synthetic(X_train, y_train, synthetic, synthetic_labels)
        assert X_aug.shape[0] == 70
        assert y_aug.shape[0] == 70
        print_test("augment_with_synthetic", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("Data Loader", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"Data Loader: {e}")
    
    # =========================================================
    # TEST 9: Colab Scripts Syntax Check
    # =========================================================
    print_header("TEST 9: Colab Scripts Syntax Check")
    
    colab_scripts = ['colab_train_dcgan.py', 'colab_train_stylegan2.py']
    
    for script in colab_scripts:
        try:
            script_path = os.path.join(os.path.dirname(__file__), script)
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, script, 'exec')
            print_test(f"{script} syntax valid", True)
            results["passed"] += 1
        except SyntaxError as e:
            print_test(f"{script} syntax", False, str(e))
            results["failed"] += 1
            results["errors"].append(f"{script}: {e}")
        except FileNotFoundError:
            print_test(f"{script}", False, "File not found")
            results["failed"] += 1
    
    # =========================================================
    # TEST 10: End-to-End Mini Pipeline
    # =========================================================
    print_header("TEST 10: End-to-End Mini Pipeline (Dummy Data)")
    
    try:
        import tensorflow as tf
        import numpy as np
        from src.models.dcgan import DCGAN
        from src.models.classifier import CNNClassifier
        from src.data.data_loader import create_tf_dataset
        
        print("  Running mini pipeline with dummy data...")
        
        # Create dummy "dataset"
        num_samples = 32
        dummy_images_gan = np.random.uniform(-1, 1, (num_samples, 64, 64, 3)).astype('float32')
        dummy_images_clf = np.random.uniform(0, 1, (num_samples, 64, 64, 3)).astype('float32')
        dummy_labels = np.eye(7)[np.random.randint(0, 7, num_samples)]
        
        # Step 1: Create GAN and train 2 steps
        dcgan = DCGAN(latent_dim=100)
        dataset = create_tf_dataset(dummy_images_gan, batch_size=8)
        
        for i, batch in enumerate(dataset.take(2)):
            g_loss, d_loss = dcgan.train_step(batch)
        print_test("GAN training steps", True)
        results["passed"] += 1
        
        # Step 2: Generate synthetic images
        synthetic = dcgan.generate_images(8, seed=42).numpy()
        print_test("Synthetic image generation", True)
        results["passed"] += 1
        
        # Step 3: Augment data
        X_aug = np.concatenate([dummy_images_clf[:24], synthetic])
        y_aug = np.concatenate([dummy_labels[:24], np.eye(7)[np.zeros(8, dtype=int)]])
        print_test("Data augmentation", True)
        results["passed"] += 1
        
        # Step 4: Train classifier
        classifier = CNNClassifier(num_classes=7)
        classifier.model.fit(X_aug, y_aug, epochs=1, batch_size=8, verbose=0)
        print_test("Classifier training", True)
        results["passed"] += 1
        
        # Step 5: Evaluate
        preds = np.argmax(classifier.model.predict(dummy_images_clf[:8], verbose=0), axis=1)
        assert len(preds) == 8
        print_test("Classifier prediction", True)
        results["passed"] += 1
        
    except Exception as e:
        print_test("End-to-End Pipeline", False, str(e))
        results["failed"] += 1
        results["errors"].append(f"Pipeline: {e}\n{traceback.format_exc()}")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print_header("TEST SUMMARY")
    
    total = results["passed"] + results["failed"]
    print(f"\n  Total Tests: {total}")
    print(f"  Passed:      {results['passed']}")
    print(f"  Failed:      {results['failed']}")
    print(f"  Success Rate: {results['passed']/total*100:.1f}%")
    
    if results["errors"]:
        print("\n  Errors:")
        for err in results["errors"]:
            print(f"    - {err[:100]}...")
    
    if results["failed"] == 0:
        print("\n  " + "="*50)
        print("  ALL TESTS PASSED! Code is ready to run.")
        print("  " + "="*50)
        return 0
    else:
        print("\n  " + "="*50)
        print("  SOME TESTS FAILED. Please fix errors above.")
        print("  " + "="*50)
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
