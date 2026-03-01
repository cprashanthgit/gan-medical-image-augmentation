"""
Configuration module for GAN-based Data Augmentation Pipeline.

Contains all hyperparameters and settings for data loading, model architecture,
training, and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    image_dir: str = "./data/HAM10000_images/"
    metadata_path: str = "./data/HAM10000_metadata.csv"
    img_size: int = 64
    channels: int = 3
    test_size: float = 0.30
    val_split: float = 0.50
    random_state: int = 42
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.img_size, self.img_size, self.channels)


@dataclass
class GANConfig:
    """Configuration for GAN training."""
    
    latent_dim: int = 100
    batch_size: int = 64
    buffer_size: int = 10000
    epochs: int = 100
    learning_rate: float = 1e-4
    beta_1: float = 0.5
    
    output_dir: str = "./outputs/generated_images/"
    checkpoint_dir: str = "./outputs/checkpoints/"
    
    num_examples_to_generate: int = 16
    save_interval: int = 5


@dataclass
class StyleGAN2Config(GANConfig):
    """Additional configuration specific to StyleGAN2."""
    
    dlatent_dim: int = 256
    mapping_layers: int = 4
    base_filters: int = 256


@dataclass
class ClassifierConfig:
    """Configuration for CNN classifier."""
    
    num_classes: int = 7
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    dropout_rate: float = 0.5
    
    use_class_weights: bool = False


@dataclass
class Config:
    """Master configuration containing all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    stylegan2: StyleGAN2Config = field(default_factory=StyleGAN2Config)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    
    device: str = "GPU"
    seed: int = 42
    
    def __post_init__(self):
        Path(self.gan.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.gan.checkpoint_dir).mkdir(parents=True, exist_ok=True)


# HAM10000 class labels
CLASS_LABELS = [
    'akiec',  # Actinic Keratoses
    'bcc',    # Basal Cell Carcinoma
    'bkl',    # Benign Keratosis
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic Nevi
    'vasc'    # Vascular Lesions
]

CLASS_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()
