import tensorflow as tf
import numpy as np
from transformer import *
from encoder import TransformerEncoder
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################
   
    

########################################################################################
# Define the function to load embeddings

# Use the embedding layer in your Keras model
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, embed_size, embedding, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.embed_size = embed_size
        self.encoding = TransformerEncoder(vocab_size=vocab_size, hidden_size=hidden_size, window_size=window_size, embedding=embedding)
        # Load the embedding matrix
        self.embedding_matrix = embedding

        # Define the Keras embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        )

        self.self_atten = AttentionHead(embed_size, embed_size, True)
        self.self_context_atten =  AttentionHead(embed_size, embed_size, False)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Dense(512),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Dense(8),
                                            tf.keras.layers.ReLU()])
        
        self.softmax = tf.keras.layers.Softmax()
    def load_embedding(self):
    # Load the embedding file
        data_dict = {}
        with open('data/fake.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract the embedding matrix
        embedding_matrix = np.zeros((len(lines), 300))
        for i, line in enumerate(lines):
            parts = line.strip().split(' ')
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)
            data_dict[parts[0]] = i
        # Return the embedding matrix
        data_dict['<pad>'] = -1
        return embedding_matrix, data_dict



    def call(self, post, context_sequence):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)

        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits



        # full sequence
        

        #contextseq!!
        
        encoded_post = self.encoding(post)

        attention1 = self.self_atten(encoded_post, encoded_post, encoded_post)
        norm1 = self.layer_norm(encoded_post + attention1)
        attention2 = self.self_context_atten(context_sequence, context_sequence, norm1)
        norm2 = self.layer_norm(norm1 + attention2)
        ff1 = self.feed_forward(norm2)
        norm3 = self.layer_norm(norm2 + ff1)
        norm4 = self.layer_norm(norm3)
        return norm4
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
        }


