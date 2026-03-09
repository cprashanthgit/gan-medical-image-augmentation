"""
Deep Convolutional GAN (DCGAN) Implementation.

Architecture based on the paper:
"Unsupervised Representation Learning with Deep Convolutional GANs" (Radford et al., 2016)

This implementation is designed for 64x64 RGB image generation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class DCGANGenerator(keras.Model):
    """
    DCGAN Generator Network.
    
    Takes a latent vector (noise) and generates a 64x64x3 image.
    
    Architecture:
        z (100,) -> Dense -> Reshape (8x8x256) -> 
        ConvTranspose2D blocks -> RGB image (64x64x3)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_size: int = 64,
        channels: int = 3,
        name: str = "dcgan_generator"
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Sequential:
        """Build the generator network."""
        model = keras.Sequential(name="generator_network")
        
        model.add(layers.Dense(
            8 * 8 * 256,
            use_bias=False,
            input_shape=(self.latent_dim,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Reshape((8, 8, 256)))
        
        model.add(layers.Conv2DTranspose(
            128, (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        model.add(layers.Conv2DTranspose(
            64, (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        model.add(layers.Conv2DTranspose(
            self.channels, (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh"
        ))
        
        return model
    
    def call(self, inputs, training=False):
        """Forward pass through generator."""
        return self.model(inputs, training=training)
    
    def generate(self, num_images: int, seed: Optional[int] = None) -> tf.Tensor:
        """
        Generate images from random noise.
        
        Args:
            num_images: Number of images to generate
            seed: Optional random seed
            
        Returns:
            Generated images in [0, 1] range
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        noise = tf.random.normal([num_images, self.latent_dim])
        generated = self(noise, training=False)
        
        generated = (generated + 1.0) / 2.0
        generated = tf.clip_by_value(generated, 0.0, 1.0)
        
        return generated


class DCGANDiscriminator(keras.Model):
    """
    DCGAN Discriminator Network.
    
    Takes a 64x64x3 image and outputs a single logit (real/fake score).
    
    Architecture:
        RGB image (64x64x3) -> Conv2D blocks -> Flatten -> Dense (1)
    """
    
    def __init__(
        self,
        img_size: int = 64,
        channels: int = 3,
        name: str = "dcgan_discriminator"
    ):
        super().__init__(name=name)
        
        self.img_size = img_size
        self.channels = channels
        
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Sequential:
        """Build the discriminator network."""
        model = keras.Sequential(name="discriminator_network")
        
        model.add(layers.Conv2D(
            64, (5, 5),
            strides=(2, 2),
            padding="same",
            input_shape=(self.img_size, self.img_size, self.channels)
        ))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model
    
    def call(self, inputs, training=False):
        """Forward pass through discriminator."""
        return self.model(inputs, training=training)


class DCGAN:
    """
    Complete DCGAN implementation with training utilities.
    
    Combines generator and discriminator with loss functions and optimizers.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_size: int = 64,
        channels: int = 3,
        learning_rate: float = 1e-4,
        beta_1: float = 0.5
    ):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        self.generator = DCGANGenerator(latent_dim, img_size, channels)
        self.discriminator = DCGANDiscriminator(img_size, channels)
        
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=beta_1
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=beta_1
        )
        
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
    
    def discriminator_loss(
        self,
        real_output: tf.Tensor,
        fake_output: tf.Tensor
    ) -> tf.Tensor:
        """Calculate discriminator loss."""
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output: tf.Tensor) -> tf.Tensor:
        """Calculate generator loss."""
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, real_images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Execute one training step.
        
        Args:
            real_images: Batch of real images
            
        Returns:
            Generator loss, Discriminator loss
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)
        
        gen_gradients = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return g_loss, d_loss
    
    def generate_images(
        self,
        num_images: int,
        seed: Optional[int] = None
    ) -> tf.Tensor:
        """Generate images using the trained generator."""
        return self.generator.generate(num_images, seed)
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        self.generator.save_weights(f"{filepath}_generator.weights.h5")
        self.discriminator.save_weights(f"{filepath}_discriminator.weights.h5")
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        self.generator.load_weights(f"{filepath}_generator.weights.h5")
        self.discriminator.load_weights(f"{filepath}_discriminator.weights.h5")
