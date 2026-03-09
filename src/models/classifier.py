"""
CNN Classifier for Skin Lesion Classification.

A simple but effective CNN architecture for multi-class classification
of dermoscopic images from the HAM10000 dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from typing import Tuple, Optional, Dict
import numpy as np

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha * (1 - p)^gamma * log(p)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
    return focal_loss_fixed

class CNNClassifier:
    """
    CNN Classifier for skin lesion classification.
    
    Architecture:
        Input (64x64x3) -> 
        Conv Block 1 (32 filters) -> MaxPool ->
        Conv Block 2 (64 filters) -> MaxPool ->
        Conv Block 3 (128 filters) -> MaxPool ->
        Flatten -> Dense (128) -> Dropout -> Dense (num_classes)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 3),
        num_classes: int = 7,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.5
    ):
        """
        Initialize CNN classifier.
        
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            learning_rate: Adam optimizer learning rate
            dropout_rate: Dropout probability
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the CNN architecture."""
        inputs = Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = Model(inputs, outputs, name="skin_lesion_classifier")
        
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
            metrics=["accuracy"]
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        class_weights: Optional[Dict[int, float]] = None,
        callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        Train the classifier.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Optional class weights for imbalanced data
            callbacks: Optional Keras callbacks
            
        Returns:
            Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for input images.
        
        Args:
            X: Input images
            
        Returns:
            Class probabilities
        """
        return self.model.predict(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Get predicted class indices for input images.
        
        Args:
            X: Input images
            
        Returns:
            Predicted class indices
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def save(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "CNNClassifier":
        """Load model from file."""
        instance = cls.__new__(cls)
        instance.model = keras.models.load_model(
            filepath,
            custom_objects={'focal_loss_fixed': categorical_focal_loss(alpha=0.25, gamma=2.0)}
        )
        instance.history = None
        return instance


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        y_train: Training labels (one-hot encoded)
        
    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils import class_weight
    
    y_train_int = np.argmax(y_train, axis=1)
    
    classes = np.unique(y_train_int)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_int
    )
    
    return dict(zip(classes, weights))


def create_enhanced_cnn(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 7,
    use_batch_norm: bool = True
) -> Model:
    """
    Create an enhanced CNN with batch normalization.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs, name="enhanced_skin_lesion_classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
