# import tensorflow as tf
# import numpy as np
# import hyperparameters as hp
# from transformer import *
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################
   
    

########################################################################################
# taking in text

# Define the function to load embeddings

# Use the embedding layer in your Keras model
# class TransformerEncoder(tf.keras.layers.Layer):
#     def __init__(self, vocab_size, hidden_size, window_size, embedding, embed_size, **kwargs):
#         super(TransformerEncoder, self).__init__(**kwargs)
#         self.vocab_size  = vocab_size
#         self.hidden_size = hidden_size
#         self.window_size = window_size
#         self.embedding_matrix = embedding
#         self.embedding_layer = tf.keras.layers.Embedding(
#             input_dim=self.embedding_matrix.shape[0],
#             output_dim=self.embedding_matrix.shape[1],
#             weights=[self.embedding_matrix],
#             trainable=True
#         )
#         self.positional_embedding = TokenAndPositionEmbedding(self.embedding_layer, hp.maxlen, vocab_size, hp.embed_size)
#         self.transformerblock = TransformerBlock(embed_size=embed_size,  is_decoder=False)
#     def call(self, x):
#         # TODO:
#         # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK
#         # 2) Pass the captions through your positional encoding layer
#         # 3) Pass the english embeddings and the image sequences to the decoder
#         # 4) Apply dense layer(s) to the decoder out to generate logits

#         embedding = self.positional_embedding(x)
#         x = self.transformerblock(embedding)
      
#         return x, embedding
    
    
