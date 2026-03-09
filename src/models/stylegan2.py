"""
StyleGAN2-inspired Generator Implementation.

This is a lightweight implementation inspired by:
"Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)

Key features:
- Mapping network (z -> w)
- Adaptive Instance Normalization (AdaIN) with style modulation
- Progressive upsampling through style blocks

Note: This is a simplified version for educational purposes, not the full NVIDIA implementation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


def build_mapping_network(
    latent_dim: int = 100,
    dlatent_dim: int = 256,
    num_layers: int = 4
) -> keras.Model:
    """
    Build the mapping network that transforms z -> w.
    
    The mapping network learns a more disentangled representation (w)
    from the initial latent code (z).
    
    Args:
        latent_dim: Dimension of input latent vector z
        dlatent_dim: Dimension of output style vector w
        num_layers: Number of fully connected layers
        
    Returns:
        Keras model for z -> w mapping
    """
    z_input = keras.Input(shape=(latent_dim,), name="z_input")
    
    x = layers.Lambda(
        lambda z: z / (tf.sqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True)) + 1e-8),
        name="latent_normalization"
    )(z_input)
    
    for i in range(num_layers):
        x = layers.Dense(dlatent_dim, activation="linear", name=f"mapping_dense_{i}")(x)
        x = layers.LeakyReLU(0.2, name=f"mapping_lrelu_{i}")(x)
    
    return keras.Model(z_input, x, name="mapping_network")


def adain_layer(
    x: tf.Tensor,
    style: tf.Tensor,
    channels: int,
    name: str
) -> tf.Tensor:
    """
    Adaptive Instance Normalization layer.
    
    Modulates the feature map x using style vector w.
    
    Args:
        x: Feature map (N, H, W, C)
        style: Style vector w (N, dlatent_dim)
        channels: Number of output channels
        name: Layer name prefix
        
    Returns:
        Style-modulated feature map
    """
    gamma = layers.Dense(channels, name=f"{name}_gamma")(style)
    beta = layers.Dense(channels, name=f"{name}_beta")(style)
    
    def _adain(inputs):
        feat, g, b = inputs
        mean, var = tf.nn.moments(feat, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + 1e-8)
        feat_norm = (feat - mean) / std
        
        g = tf.reshape(g, [-1, 1, 1, channels])
        b = tf.reshape(b, [-1, 1, 1, channels])
        
        return feat_norm * g + b
    
    return layers.Lambda(_adain, name=f"{name}_adain")([x, gamma, beta])


def style_block(
    x: tf.Tensor,
    style: tf.Tensor,
    filters: int,
    upsample: bool,
    name: str
) -> tf.Tensor:
    """
    StyleGAN2 synthesis block.
    
    Optionally upsamples, applies convolution, AdaIN, and activation.
    
    Args:
        x: Input feature map
        style: Style vector w
        filters: Number of output filters
        upsample: Whether to upsample (2x)
        name: Block name prefix
        
    Returns:
        Processed feature map
    """
    if upsample:
        x = layers.UpSampling2D(interpolation="nearest", name=f"{name}_upsample")(x)
    
    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        use_bias=False,
        name=f"{name}_conv"
    )(x)
    
    x = adain_layer(x, style, filters, name=name)
    x = layers.LeakyReLU(0.2, name=f"{name}_lrelu")(x)
    
    return x


class StyleGAN2Generator(keras.Model):
    """
    StyleGAN2-inspired Generator.
    
    Features:
    - Mapping network for disentangled latent space
    - Style-based synthesis with AdaIN
    - Progressive resolution increase through style blocks
    
    Output: 64x64x3 RGB images in [-1, 1] range
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_size: int = 64,
        channels: int = 3,
        dlatent_dim: int = 256,
        base_filters: int = 256,
        mapping_layers: int = 4,
        name: str = "stylegan2_generator"
    ):
        super().__init__(name=name)
        
        assert img_size == 64, "This implementation supports 64x64 output only"
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.dlatent_dim = dlatent_dim
        self.base_filters = base_filters
        
        self.mapping_network = build_mapping_network(
            latent_dim, dlatent_dim, mapping_layers
        )
        
        self._build_synthesis_network()
    
    def _build_synthesis_network(self):
        """Build synthesis network layers - all layers pre-built."""
        self.dense_4x4 = layers.Dense(4 * 4 * self.base_filters, name="dense_4x4")
        self.reshape_4x4 = layers.Reshape((4, 4, self.base_filters), name="reshape_4x4")
        
        # Block 8: 4x4 -> 8x8
        self.up_8 = layers.UpSampling2D(interpolation="nearest", name="up_8")
        self.conv_8 = layers.Conv2D(self.base_filters, (3, 3), padding="same", use_bias=False, name="conv_8")
        self.gamma_8 = layers.Dense(self.base_filters, name="gamma_8")
        self.beta_8 = layers.Dense(self.base_filters, name="beta_8")
        self.lrelu_8 = layers.LeakyReLU(0.2, name="lrelu_8")
        
        # Block 16: 8x8 -> 16x16
        self.up_16 = layers.UpSampling2D(interpolation="nearest", name="up_16")
        self.conv_16 = layers.Conv2D(self.base_filters // 2, (3, 3), padding="same", use_bias=False, name="conv_16")
        self.gamma_16 = layers.Dense(self.base_filters // 2, name="gamma_16")
        self.beta_16 = layers.Dense(self.base_filters // 2, name="beta_16")
        self.lrelu_16 = layers.LeakyReLU(0.2, name="lrelu_16")
        
        # Block 32: 16x16 -> 32x32
        self.up_32 = layers.UpSampling2D(interpolation="nearest", name="up_32")
        self.conv_32 = layers.Conv2D(self.base_filters // 4, (3, 3), padding="same", use_bias=False, name="conv_32")
        self.gamma_32 = layers.Dense(self.base_filters // 4, name="gamma_32")
        self.beta_32 = layers.Dense(self.base_filters // 4, name="beta_32")
        self.lrelu_32 = layers.LeakyReLU(0.2, name="lrelu_32")
        
        # Block 64: 32x32 -> 64x64
        self.up_64 = layers.UpSampling2D(interpolation="nearest", name="up_64")
        self.conv_64 = layers.Conv2D(self.base_filters // 8, (3, 3), padding="same", use_bias=False, name="conv_64")
        self.gamma_64 = layers.Dense(self.base_filters // 8, name="gamma_64")
        self.beta_64 = layers.Dense(self.base_filters // 8, name="beta_64")
        self.lrelu_64 = layers.LeakyReLU(0.2, name="lrelu_64")
        
        # To RGB
        self.to_rgb = layers.Conv2D(
            self.channels, (1, 1),
            padding="same",
            activation="tanh",
            name="to_rgb"
        )
    
    def _apply_adain(self, x, gamma, beta, channels):
        """Apply Adaptive Instance Normalization."""
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + 1e-8)
        x_norm = (x - mean) / std
        
        gamma = tf.reshape(gamma, [-1, 1, 1, channels])
        beta = tf.reshape(beta, [-1, 1, 1, channels])
        
        return x_norm * gamma + beta
    
    def call(self, z, training=False):
        """
        Forward pass through StyleGAN2 generator.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            training: Training mode flag
            
        Returns:
            Generated images (batch_size, 64, 64, 3) in [-1, 1]
        """
        w = self.mapping_network(z)
        
        x = self.dense_4x4(w)
        x = self.reshape_4x4(x)
        
        # Block 8
        x = self.up_8(x)
        x = self.conv_8(x)
        x = self._apply_adain(x, self.gamma_8(w), self.beta_8(w), self.base_filters)
        x = self.lrelu_8(x)
        
        # Block 16
        x = self.up_16(x)
        x = self.conv_16(x)
        x = self._apply_adain(x, self.gamma_16(w), self.beta_16(w), self.base_filters // 2)
        x = self.lrelu_16(x)
        
        # Block 32
        x = self.up_32(x)
        x = self.conv_32(x)
        x = self._apply_adain(x, self.gamma_32(w), self.beta_32(w), self.base_filters // 4)
        x = self.lrelu_32(x)
        
        # Block 64
        x = self.up_64(x)
        x = self.conv_64(x)
        x = self._apply_adain(x, self.gamma_64(w), self.beta_64(w), self.base_filters // 8)
        x = self.lrelu_64(x)
        
        x = self.to_rgb(x)
        
        return x
    
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


class StyleGAN2:
    """
    Complete StyleGAN2 implementation with training utilities.
    
    Uses the same discriminator architecture as DCGAN for simplicity.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_size: int = 64,
        channels: int = 3,
        dlatent_dim: int = 256,
        base_filters: int = 256,
        learning_rate: float = 1e-4,
        beta_1: float = 0.5
    ):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        self.generator = StyleGAN2Generator(
            latent_dim=latent_dim,
            img_size=img_size,
            channels=channels,
            dlatent_dim=dlatent_dim,
            base_filters=base_filters
        )
        
        self.discriminator = self._build_discriminator()
        
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.generator_optimizer = keras.optimizers.Adam(learning_rate, beta_1=beta_1)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate, beta_1=beta_1)
    
    def _build_discriminator(self) -> keras.Sequential:
        """Build discriminator network (same as DCGAN)."""
        model = keras.Sequential(name="discriminator")
        
        model.add(layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same",
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
    
    def discriminator_loss(self, real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
        """Calculate discriminator loss."""
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output: tf.Tensor) -> tf.Tensor:
        """Calculate generator loss."""
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, real_images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Execute one training step."""
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)
        
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return g_loss, d_loss
    
    def generate_images(self, num_images: int, seed: Optional[int] = None) -> tf.Tensor:
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
