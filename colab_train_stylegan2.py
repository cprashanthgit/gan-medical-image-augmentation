"""
StyleGAN2 Training for Google Colab
====================================
This is a standalone script that can be run directly in Google Colab.

Usage in Colab:
    1. Upload this file to Colab or copy-paste into a cell
    2. Run: !python colab_train_stylegan2.py
"""

# ============================================================
# SETUP & IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    IMAGE_DIR = "/content/HAM10000_images/"
    METADATA_PATH = "/content/drive/MyDrive/HAM10000_metadata.csv"
    OUTPUT_DIR = "/content/stylegan_outputs/"
    
    IMG_SIZE = 64
    CHANNELS = 3
    
    LATENT_DIM = 100
    DLATENT_DIM = 256
    BASE_FILTERS = 256
    GAN_EPOCHS = 100
    GAN_BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    
    NUM_CLASSES = 7
    CLASSIFIER_EPOCHS = 50
    CLASSIFIER_BATCH_SIZE = 32
    
    NUM_SYNTHETIC = 2000
    SEED = 42

config = Config()
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

# Uncomment in Colab:
# mount_drive()

# ============================================================
# DATA EXTRACTION
# ============================================================

def extract_data():
    import zipfile
    base_path = "/content/drive/MyDrive/"
    zip_files = ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]
    
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    for z in zip_files:
        zip_path = os.path.join(base_path, z)
        if os.path.exists(zip_path):
            print(f"Extracting {z}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.IMAGE_DIR)
    print(f"Done! Total images: {len(os.listdir(config.IMAGE_DIR))}")

# Uncomment to extract:
# extract_data()

# ============================================================
# DATA LOADING
# ============================================================

def load_metadata():
    df = pd.read_csv(config.METADATA_PATH)
    df.columns = df.columns.str.strip()
    df['path'] = df['image_id'].apply(lambda x: os.path.join(config.IMAGE_DIR, f"{x}.jpg"))
    return df

def preprocess_image_gan(path):
    img = load_img(path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img = img_to_array(img)
    return (img / 127.5) - 1.0

def preprocess_image_classifier(path):
    img = load_img(path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img = img_to_array(img)
    return img / 255.0

def load_all_images(df, for_gan=True):
    images = []
    preprocess_fn = preprocess_image_gan if for_gan else preprocess_image_classifier
    for i, path in enumerate(df['path']):
        images.append(preprocess_fn(path))
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{len(df)} images...")
    return np.array(images, dtype='float32')

# ============================================================
# STYLEGAN2 MODEL
# ============================================================

def build_mapping_network():
    """Mapping network: z -> w"""
    z_input = keras.Input(shape=(config.LATENT_DIM,), name="z_input")
    
    # Normalize latent
    x = layers.Lambda(
        lambda z: z / (tf.sqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True)) + 1e-8)
    )(z_input)
    
    for i in range(4):
        x = layers.Dense(config.DLATENT_DIM, activation="linear")(x)
        x = layers.LeakyReLU(0.2)(x)
    
    return keras.Model(z_input, x, name="mapping_network")

def adain(x, style, channels, name):
    """Adaptive Instance Normalization."""
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

def style_block(x, style, filters, upsample, name):
    """Style block with optional upsampling."""
    if upsample:
        x = layers.UpSampling2D(interpolation="nearest")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = adain(x, style, filters, name=name)
    x = layers.LeakyReLU(0.2)(x)
    return x

def make_stylegan2_generator():
    """Build StyleGAN2 generator."""
    z_input = keras.Input(shape=(config.LATENT_DIM,), name="z")
    
    mapping = build_mapping_network()
    w = mapping(z_input)
    
    # Start from 4x4
    x = layers.Dense(4 * 4 * config.BASE_FILTERS)(w)
    x = layers.Reshape((4, 4, config.BASE_FILTERS))(x)
    
    # Progressive upsampling: 4->8->16->32->64
    x = style_block(x, w, config.BASE_FILTERS, upsample=True, name="g_block_8")
    x = style_block(x, w, config.BASE_FILTERS // 2, upsample=True, name="g_block_16")
    x = style_block(x, w, config.BASE_FILTERS // 4, upsample=True, name="g_block_32")
    x = style_block(x, w, config.BASE_FILTERS // 8, upsample=True, name="g_block_64")
    
    # To RGB
    x = layers.Conv2D(3, (1, 1), padding="same", activation="tanh")(x)
    
    return keras.Model(z_input, x, name="stylegan2_generator")

def make_discriminator():
    """Build discriminator (same as DCGAN)."""
    model = keras.Sequential(name="discriminator")
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                            input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)))
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

# Initialize models
generator = make_stylegan2_generator()
discriminator = make_discriminator()

# Loss and optimizers
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, config.LATENT_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
    
    gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return g_loss, d_loss

def generate_and_save_images(epoch, seed):
    predictions = generator(seed, training=False)
    predictions = (predictions + 1.0) / 2.0
    predictions = tf.clip_by_value(predictions, 0.0, 1.0)
    
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis("off")
    plt.savefig(f"{config.OUTPUT_DIR}epoch_{epoch:03d}.png")
    plt.close()

# ============================================================
# CNN CLASSIFIER
# ============================================================

def make_cnn_classifier():
    inputs = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "="*60)
    print(" StyleGAN2 Data Augmentation Pipeline - Google Colab")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_metadata()
    print(f"Loaded {len(df)} samples")
    
    # Prepare GAN dataset
    print("\n[2/6] Preparing GAN dataset...")
    all_images = load_all_images(df, for_gan=True)
    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.GAN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Train StyleGAN2
    print("\n[3/6] Training StyleGAN2...")
    seed = tf.random.normal([16, config.LATENT_DIM])
    
    for epoch in range(1, config.GAN_EPOCHS + 1):
        g_losses, d_losses = [], []
        for batch in dataset:
            g_loss, d_loss = train_step(batch)
            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())
        
        print(f"Epoch {epoch}/{config.GAN_EPOCHS} - G_loss: {np.mean(g_losses):.4f}, D_loss: {np.mean(d_losses):.4f}")
        
        if epoch == 1 or epoch % 5 == 0 or epoch == config.GAN_EPOCHS:
            generate_and_save_images(epoch, seed)
    
    # Generate synthetic images
    print("\n[4/6] Generating synthetic images...")
    noise = tf.random.normal([config.NUM_SYNTHETIC, config.LATENT_DIM])
    synthetic_images = generator(noise, training=False)
    synthetic_images = (synthetic_images + 1.0) / 2.0
    synthetic_images = tf.clip_by_value(synthetic_images, 0.0, 1.0).numpy()
    
    # Prepare classifier data
    print("\n[5/6] Preparing classifier data...")
    X_real = load_all_images(df, for_gan=False)
    label_names = sorted(df['dx'].unique())
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y_real = np.array([label_to_idx[label] for label in df['dx']])
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_real, y_real, test_size=0.30, stratify=y_real, random_state=config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=config.SEED)
    
    y_train_cat = to_categorical(y_train, config.NUM_CLASSES)
    y_val_cat = to_categorical(y_val, config.NUM_CLASSES)
    y_test_cat = to_categorical(y_test, config.NUM_CLASSES)
    
    majority_class = df['dx'].value_counts().idxmax()
    majority_idx = label_to_idx[majority_class]
    synthetic_labels = to_categorical(np.full(config.NUM_SYNTHETIC, majority_idx), config.NUM_CLASSES)
    
    # Train classifiers
    print("\n[6/6] Training classifiers...")
    
    print("\n--- Baseline CNN ---")
    baseline_cnn = make_cnn_classifier()
    baseline_cnn.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                     epochs=config.CLASSIFIER_EPOCHS, batch_size=config.CLASSIFIER_BATCH_SIZE)
    _, baseline_acc = baseline_cnn.evaluate(X_test, y_test_cat)
    
    print("\n--- Augmented CNN ---")
    X_aug = np.concatenate([X_train, synthetic_images], axis=0)
    y_aug = np.concatenate([y_train_cat, synthetic_labels], axis=0)
    
    aug_cnn = make_cnn_classifier()
    aug_cnn.fit(X_aug, y_aug, validation_data=(X_val, y_val_cat),
                epochs=config.CLASSIFIER_EPOCHS, batch_size=config.CLASSIFIER_BATCH_SIZE)
    _, aug_acc = aug_cnn.evaluate(X_test, y_test_cat)
    
    # Results
    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    print(f"Baseline Accuracy:  {baseline_acc:.4f}")
    print(f"Augmented Accuracy: {aug_acc:.4f}")
    print(f"Improvement:        {(aug_acc - baseline_acc)*100:+.2f}%")
    
    y_pred = np.argmax(aug_cnn.predict(X_test), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=label_names))

if __name__ == "__main__":
    main()
