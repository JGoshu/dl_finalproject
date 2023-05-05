import tensorflow as tf
import numpy as np
from transformer import *
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################
   
    

########################################################################################
# taking in text

# Define the function to load embeddings

# Use the embedding layer in your Keras model
class TransformerEncoder(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        # Load the embedding matrix
        self.embedding_matrix = self.load_embedding()

        # Define the Keras embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        )
        # Define english embedding layer:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        # Define decoder layer that handles language context:     
        self.encoder = TransformerBlock(self.hidden_size)

        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size, activation="leaky_relu")
    def load_embedding(self):
        # Load the embedding file
        with open('data/fake.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract the embedding matrix
        embedding_matrix = np.zeros((len(lines), 300))
        for i, line in enumerate(lines):
            parts = line.strip().split(' ')
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)

        # Return the embedding matrix
        return embedding_matrix

    def call(self, post):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)

        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        print("post: ", post)
        encoded_post = self.encoding(post)
        decoder_output = self.encoder(encoded_post)
        probs = self.classifier(decoder_output)

        return probs
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
        }


