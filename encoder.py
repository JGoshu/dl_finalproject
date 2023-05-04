import tensorflow as tf
import numpy as np
from transformer import *
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################
   
    

########################################################################################

class TransformerEncoder(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.text_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size*2, activation="leaky_relu"),
            tf.keras.layers.Dense(self.hidden_size, activation="leaky_relu"),
        ])
        # Define english embedding layer:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        # Define decoder layer that handles language context:     
        self.encoder = TransformerBlock(self.hidden_size, False)

        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size, activation="leaky_relu")

    def call(self, post):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)

        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits

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


