"""
vqvae module (TensorFlow / Keras)
---------------------------------
Minimal VQ-VAE for 256x256 RGB tiles by default (configurable).

API:
- get_vqvae(imsize=256, latent_dim=512, num_embeddings=16) -> keras.Model
- VQVAETrainer(train_variance, imsize=256, latent_dim=512, num_embeddings=16)
- encode_to_codes(model, images) -> (codes_h, codes_w, code_indices[int32])
- decode_from_codes(model, code_indices) -> reconstructed RGB tiles in [-0.5, 0.5] range
"""
from typing import Tuple
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True, name="embeddings_vqvae",
        )

    def call(self, x):
        inp_shape = tf.shape(x)
        flat = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flat)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, inp_shape)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)
        return x + tf.stop_gradient(quantized - x)

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        return tf.argmin(distances, axis=1)

def get_encoder(imsize: int = 256, latent_dim: int = 512):
    inputs = keras.Input(shape=(imsize, imsize, 3))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)   # /2
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)        # /4
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)       # /8
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)       # /16
    x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)       # /32
    out = layers.Conv2D(latent_dim, 1, activation="relu", padding="valid")(x)        # (im/32, im/32, latent_dim)
    return keras.Model(inputs, out, name="encoder")

def get_decoder(imsize: int = 256, latent_dim: int = 512):
    hw = imsize // 32
    latent_inputs = keras.Input(shape=(hw, hw, latent_dim))
    x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    out = layers.Conv2DTranspose(3, 3, strides=2, padding="same")(x)  # linear output
    return keras.Model(latent_inputs, out, name="decoder")

def get_vqvae(imsize: int = 256, latent_dim: int = 512, num_embeddings: int = 16) -> keras.Model:
    vq = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    enc = get_encoder(imsize, latent_dim)
    dec = get_decoder(imsize, latent_dim)
    inputs = keras.Input(shape=(imsize, imsize, 3))
    z = enc(inputs)
    qz = vq(z)
    xhat = dec(qz)
    return keras.Model(inputs, xhat, name="vq_vae")

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, imsize=256, latent_dim=512, num_embeddings=16, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = float(train_variance)
        self.imsize = imsize
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.vqvae = get_vqvae(imsize, latent_dim, num_embeddings)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.vq_loss_tracker]

    def train_step(self, x):
        import tensorflow as tf  # ensure tf is in local scope
        with tf.GradientTape() as tape:
            xhat = self.vqvae(x)
            recon_loss = tf.reduce_mean((x - xhat) ** 2) / self.train_variance
            total = recon_loss + tf.add_n(self.vqvae.losses)
        grads = tape.gradient(total, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(tf.add_n(self.vqvae.losses))
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

def encode_to_codes(vqvae: keras.Model, images: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """images in [-0.5, 0.5], shape (B, H, W, 3). Returns (Hc, Wc, codes[int32] of shape (B, Hc, Wc))"""
    enc = vqvae.get_layer("encoder")
    vq_layer = vqvae.get_layer("vector_quantizer")
    z = enc.predict(images, verbose=0)
    B, Hc, Wc, D = z.shape
    flat = z.reshape(-1, D)
    codes = vq_layer.get_code_indices(flat).numpy().reshape(B, Hc, Wc)
    return Hc, Wc, codes.astype(np.int32)

def decode_from_codes(vqvae: keras.Model, code_indices: np.ndarray) -> np.ndarray:
    """code_indices: (B, Hc, Wc) int32 -> returns reconstructed images in [-0.5, 0.5]"""
    dec = vqvae.get_layer("decoder")
    vq_layer = vqvae.get_layer("vector_quantizer")
    emb = vq_layer.embeddings.numpy()  # (D, K)
    B, Hc, Wc = code_indices.shape
    flat = code_indices.reshape(-1)
    quant = emb[:, flat].T  # (B*Hc*Wc, D)
    quant = quant.reshape(B, Hc, Wc, emb.shape[0]).astype(np.float32)
    xhat = dec.predict(quant, verbose=0)
    return xhat
