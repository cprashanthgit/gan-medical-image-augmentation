"""
GAN Training Module.

Provides a unified trainer class for training DCGAN and StyleGAN2 models
with progress tracking, checkpointing, and image generation.
"""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..models.dcgan import DCGAN
from ..models.stylegan2 import StyleGAN2
from ..config import GANConfig


class GANTrainer:
    """
    Unified trainer for GAN models (DCGAN, StyleGAN2).
    
    Handles training loop, progress tracking, checkpointing,
    and sample image generation during training.
    """
    
    def __init__(
        self,
        model: Union[DCGAN, StyleGAN2],
        config: GANConfig,
        model_name: str = "gan"
    ):
        """
        Initialize GAN trainer.
        
        Args:
            model: DCGAN or StyleGAN2 instance
            config: GANConfig with training parameters
            model_name: Name for saving outputs
        """
        self.model = model
        self.config = config
        self.model_name = model_name
        
        self.output_dir = Path(config.output_dir) / model_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / model_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = tf.random.normal([config.num_examples_to_generate, model.latent_dim])
        
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'epochs': []
        }
    
    def train(
        self,
        dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        save_interval: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the GAN model.
        
        Args:
            dataset: tf.data.Dataset of training images
            epochs: Number of epochs (uses config if None)
            save_interval: Interval for saving samples (uses config if None)
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        save_interval = save_interval or self.config.save_interval
        
        for epoch in range(1, epochs + 1):
            g_losses = []
            d_losses = []
            
            for batch in dataset:
                g_loss, d_loss = self.model.train_step(batch)
                g_losses.append(g_loss.numpy())
                d_losses.append(d_loss.numpy())
            
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['epochs'].append(epoch)
            
            if verbose:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Generator loss:     {avg_g_loss:.4f}")
                print(f"  Discriminator loss: {avg_d_loss:.4f}")
            
            if epoch == 1 or epoch % save_interval == 0 or epoch == epochs:
                self._save_samples(epoch)
        
        self.model.save_weights(str(self.checkpoint_dir / "final"))
        
        if verbose:
            print("Training complete!")
        
        return self.history
    
    def _save_samples(self, epoch: int):
        """Generate and save sample images."""
        predictions = self.model.generator(self.seed, training=False)
        
        predictions = (predictions + 1.0) / 2.0
        predictions = tf.clip_by_value(predictions, 0.0, 1.0)
        
        num_images = self.config.num_examples_to_generate
        grid_size = int(np.sqrt(num_images))
        
        fig = plt.figure(figsize=(grid_size * 2, grid_size * 2))
        
        for i in range(num_images):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(predictions[i])
            plt.axis("off")
        
        save_path = self.output_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Saved samples to: {save_path}")
    
    def generate_synthetic_images(
        self,
        num_images: int,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Generate synthetic images using trained generator.
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            Numpy array of generated images in [0, 1] range
        """
        return self.model.generate_images(num_images, seed).numpy()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['epochs'], self.history['g_loss'], label='Generator')
        axes[0].plot(self.history['epochs'], self.history['d_loss'], label='Discriminator')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('GAN Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['epochs'], self.history['g_loss'], label='Generator')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Generator Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training plot to: {save_path}")
        
        plt.show()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        self.model.load_weights(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")


def train_class_specific_gan(
    model_class: type,
    images: np.ndarray,
    label: str,
    config: GANConfig,
    epochs: int = 100
) -> Union[DCGAN, StyleGAN2]:
    """
    Train a GAN model for a specific class.
    
    Args:
        model_class: DCGAN or StyleGAN2 class
        images: Images for the specific class
        label: Class label name
        config: GANConfig
        epochs: Number of training epochs
        
    Returns:
        Trained GAN model
    """
    print(f"\n{'='*60}")
    print(f"Training GAN for class: {label}")
    print(f"{'='*60}")
    
    model = model_class(
        latent_dim=config.latent_dim,
        img_size=64,
        channels=3,
        learning_rate=config.learning_rate,
        beta_1=config.beta_1
    )
    
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(config.buffer_size)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    trainer = GANTrainer(model, config, model_name=f"gan_{label}")
    trainer.train(dataset, epochs=epochs)
    
    return model
