"""Model architectures for GAN-based data augmentation."""

from .dcgan import DCGANGenerator, DCGANDiscriminator, DCGAN
from .stylegan2 import StyleGAN2Generator, StyleGAN2
from .classifier import CNNClassifier

__all__ = [
    'DCGANGenerator',
    'DCGANDiscriminator', 
    'DCGAN',
    'StyleGAN2Generator',
    'StyleGAN2',
    'CNNClassifier'
]
