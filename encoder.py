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
    def __init__(self, vocab_size, hidden_size, window_size, embedding, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        # Load the embedding matrix
        #TODO
        self.embedding_matrix = embedding
        
        # Define the Keras embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        )
        # Define english embedding layer:
       
        # Define decoder layer that handles language context:     
        self.encoder = TransformerBlock(self.hidden_size)

        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size, activation="leaky_relu")

    def call(self, post):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)

        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        print("post: ", post)
        embedding = self.embedding_layer(post)
        encoded = self.encoder(embedding)
        probs = self.classifier(encoded)

        return probs
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
        }


