import tensorflow as tf
import numpy as np
from transformer import *
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################
   
    

########################################################################################
# Define the function to load embeddings

# Use the embedding layer in your Keras model
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, embed_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.embed_size = embed_size
        
        # Load the embedding matrix
        self.embedding_matrix = self.load_embedding('glove.42B.300d.txt')

        # Define the Keras embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        )

        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Dense(512),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Dense(8),
                                            tf.keras.layers.ReLU()])
        
        self.softmax = tf.keras.layers.Softmax()
    def load_embedding(embedding_file):
    # Load the embedding file
        with open(embedding_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract the embedding matrix
        embedding_matrix = np.zeros((len(lines), 300))
        for i, line in enumerate(lines):
            parts = line.strip().split(' ')
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)

        # Return the embedding matrix
        return embedding_matrix

    def call(self, post, mask):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)

        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits

        encoded_post = self.encoding(post)

        masked_attention = self.self_atten(encoded_post, attention_mask=mask)
        norm1 = self.layer_norm(encoded_post + masked_attention)
        attention = self.self_atten(norm1)
        norm2 = self.layer_norm(norm1 + attention)
        ff_layer = self.feed_forward(norm2)
        norm3 = self.layer_norm(ff_layer)
        logits = self.layer_norm(norm3)
        probs = self.softmax(logits)

        return probs
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
        }


