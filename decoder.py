# import tensorflow as tf
# import numpy as np
# from transformer import *
# from encoder import TransformerEncoder
# # except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

# ########################################################################################
   
    

# ########################################################################################
# # Define the function to load embeddings

# # Use the embedding layer in your Keras model
# class TransformerDecoder(tf.keras.layers.Layer):
#     def __init__(self, vocab_size, hidden_size, window_size, embed_size, embedding, **kwargs):
#         super(TransformerDecoder, self).__init__(**kwargs)
#         self.vocab_size  = vocab_size
#         self.hidden_size = hidden_size
#         self.window_size = window_size
#         self.embed_size = embed_size
        
        
#     def call(self, x, labels):
#         #NOTE: embed?
#         # x =self.positional_embedding(x)
#         x = self.transformer_block(x, context_sequence=tf.cast(labels, np.float32), is_decoder=True)
#         x = np.expand_dims(x, axis=0)
#         x= self.pooling(x)
#         print("X: ", x)
#         x = self.classifier(x)
#         print("CLASSY X: ", x)
#         return x
    
#     def get_config(self):
#         return {
#             "vocab_size": self.vocab_size,
#             "hidden_size": self.hidden_size,
#             "window_size": self.window_size,
#         }


